import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from croniter import croniter
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.utils.models import (
    SnowflakeJob,
    SnowflakeConnection,
    SnowflakeDatabase,
    SnowflakeSchema,
    SnowflakeCrawlAudit,
    SnowflakeQueryRecord,
    InformationSchemacolumns,
    ColumnLevelLineage
)
from app.services.lineage_builder.lineage_builder import lineage_builder
import snowflake.connector
from app.vector_db import upsert_lineage_embeddings_by_batch

logger = logging.getLogger("snowflake_crawler")


def _due_to_run(cron_expr: str, last_run: Optional[datetime], now: datetime) -> bool:
    try:
        # For first run (last_run is None), use a time in the past to ensure we get the next scheduled time
        start_time = last_run or (now - timedelta(minutes=1))
        itr = croniter(cron_expr, start_time)
        next_time = itr.get_next(datetime)
        return next_time <= now
    except Exception:
        logger.warning("Invalid cron expression: %s", cron_expr)
        return False


def _fetch_delta_query_history(conn: SnowflakeConnection, since: datetime) -> tuple[list[dict], list[dict]]:
    sf_conn = snowflake.connector.connect(
        user=conn.username,
        password=conn.password,
        account=conn.account,
        warehouse=conn.warehouse,
        role=conn.role,
    )
    try:
        cursor = sf_conn.cursor()
        # Limit to selected databases/schemas if configured
        selected_db_ids = [db.id for db in conn.databases if db.is_selected]
        selected_schemas = []
        for db in conn.databases:
            if db.is_selected:
                for sc in db.schemas:
                    if sc.is_selected:
                        selected_schemas.append((db.database_name, sc.schema_name))

        where_parts = ["start_time > to_timestamp_tz(%(since)s)"]
        if selected_schemas:
            # Build OR conditions for selected schemas
            schema_conditions = []
            for i, (dbname, sname) in enumerate(selected_schemas):
                schema_conditions.append(f"(database_name=%(db_name_{i})s AND schema_name=%(schema_name_{i})s)")
            where_parts.append("(" + " OR ".join(schema_conditions) + ")")

        query_access_history_query = f"""
            WITH ranked_queries AS (
                SELECT query_text,
                       query_id,
                       query_type,
                       start_time,
                       end_time,
                       database_id,
                       database_name,
                       schema_id,
                       schema_name,
                       session_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY query_text
                           ORDER BY start_time DESC
                       ) AS rn
                FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
                WHERE {" AND ".join(where_parts)}
                  AND QUERY_TYPE IN ('INSERT', 'MERGE', 'CREATE_VIEW')
                  AND execution_status = 'SUCCESS'
            )
            SELECT rq.query_id,
                   rq.start_time,
                   rq.end_time,
                   ah.base_objects_accessed,
                   ah.objects_modified,
                   rq.query_text,
                   rq.query_type,
                   rq.database_id,
                   rq.database_name,
                   rq.schema_id,
                   rq.schema_name,
                   rq.session_id
            FROM SNOWFLAKE.ACCOUNT_USAGE.ACCESS_HISTORY ah
            JOIN ranked_queries rq
              ON ah.query_id = rq.query_id
            WHERE rq.rn = 1
            ORDER BY rq.start_time DESC;
        """

        query_information_schema_columns = """
            SELECT TABLE_CATALOG as table_catalog,
                   TABLE_SCHEMA as table_schema,
                   TABLE_NAME as table_name,
                   COLUMN_NAME as column_name,
                   DATA_TYPE as data_type,
                   ORDINAL_POSITION as ordinal_position
            FROM SNOWFLAKE.ACCOUNT_USAGE.COLUMNS
            WHERE DELETED IS NULL;
        """
        
        # Build parameter dictionary
        bind_params = {"since": since.isoformat()}
        if selected_schemas:
            for i, (dbname, sname) in enumerate(selected_schemas):
                bind_params[f"db_name_{i}"] = dbname
                bind_params[f"schema_name_{i}"] = sname
        
        cursor.execute(query_access_history_query, bind_params)
        cols = [c[0].lower() for c in cursor.description]
        rows = [dict(zip(cols, r)) for r in cursor.fetchall()]

        cursor.execute(query_information_schema_columns)
        cols2 = [c[0].lower() for c in cursor.description]
        column_info_rows = [dict(zip(cols2, r)) for r in cursor.fetchall()]

        return rows, column_info_rows
    finally:
        try:
            sf_conn.close()
        except Exception:
            pass

def run_crawl_for_connection(db: Session, job: SnowflakeJob, now: datetime) -> None:
    job = db.merge(job)
    conn: SnowflakeConnection = (
        db.query(SnowflakeConnection).filter(SnowflakeConnection.id == job.connection_id, SnowflakeConnection.is_active == True).first()
    )
    if not conn:
        logger.error("❌ Connection not found for job: %s", job.connection_id)
        return
    since = job.last_run_time or (now - timedelta(days=30))
    logger.info(f"since: {since}")
    batch_id = uuid.uuid4()
    
    audit = SnowflakeCrawlAudit(
        org_id=conn.org_id,
        batch_id=batch_id,
        connection_id=conn.id,
        scheduled_at=now,
        status="running",
    )
    db.add(audit)
    db.flush()

    try:
        rows, column_info_rows = _fetch_delta_query_history(conn, since)
        
        max_end = since
        to_insert = []
        for r in rows:
            end_time = r.get("end_time") or r.get("start_time")
            if end_time and isinstance(end_time, str):
                try:
                    end_time = datetime.fromisoformat(end_time)
                except Exception:
                    end_time = None
            if end_time and end_time > max_end:
                max_end = end_time
            rec = SnowflakeQueryRecord(
                org_id=conn.org_id,
                batch_id=batch_id,
                connection_id=conn.id,
                query_id=r.get("query_id"),
                query_text=r.get("query_text"),
                database_name=r.get("database_name"),
                database_id=r.get("database_id"),
                schema_name=r.get("schema_name"),
                schema_id=r.get("schema_id"),
                query_type=r.get("query_type"),
                start_time=r.get("start_time"),
                end_time=r.get("end_time"),
                session_id=r.get("session_id"),
                base_objects_accessed=r.get("base_objects_accessed"),
                objects_modified=r.get("objects_modified")
            )
            to_insert.append(rec)
        
        if to_insert:
            db.bulk_save_objects(to_insert)

        if len(to_insert) > 0:
            logger.info("📊 Crawl completed: %d rows fetched, watermark: %s", 
                       len(to_insert), max_end.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            logger.info("ℹ️  Crawl completed: No new data found, watermark: %s", 
                       max_end.strftime("%Y-%m-%d %H:%M:%S"))

        # ---- Delete old InformationSchemacolumns first ----
        deleted_count = (
            db.query(InformationSchemacolumns)
            .filter(
                InformationSchemacolumns.org_id == conn.org_id,
                InformationSchemacolumns.connection_id == conn.id,
            )
            .delete(synchronize_session=False)
        )
        logger.info("🧹 Deleted %d old InformationSchemacolumns rows for org_id=%s, conn_id=%s",
                    deleted_count, conn.org_id, conn.id)
        db.flush()  # ensure deletion is executed before insert
         
        # ---- Insert InformationSchemacolumns ----
        column_objects = []
        for c in column_info_rows:
            col_rec = InformationSchemacolumns(
                org_id=conn.org_id,
                connection_id=conn.id,
                table_catalog=c.get("table_catalog"),
                table_schema=c.get("table_schema"),
                table_name=c.get("table_name"),
                column_name=c.get("column_name"),
                data_type=c.get("data_type"),
                ordinal_position=c.get("ordinal_position")
            )
            column_objects.append(col_rec)
        
        if column_objects:
            db.bulk_save_objects(column_objects)

        if len(column_objects) > 0:
            logger.info("📊 Crawl completed: %d rows fetched, watermark: %s", 
                       len(column_objects), max_end.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            logger.info("ℹ️  Crawl completed: No new data found, watermark: %s", 
                       max_end.strftime("%Y-%m-%d %H:%M:%S"))
            
        job.last_run_time = max_end
        audit.status = "success"
        audit.query_history_rows_fetched = len(to_insert)
        audit.information_schema_columns_rows_fetched = len(column_objects)
        audit.finished_at = datetime.now(timezone.utc)
        db.commit()
        
        # Only run lineage builder if new data was fetched in this batch
        # This prevents processing the same queries multiple times with different batch_ids
        # Unprocessed queries from previous batches will be handled when new data arrives
        # (the watermark tracks the last processed timestamp, so they'll be included in future runs)
        if len(to_insert) > 0:
            logger.info("🔄 New query history fetched (%d rows), starting lineage builder", len(to_insert))
            try:
                logger.info("🔄 Starting lineage builder for org_id: %s, conn_id: %s, batch_id: %s", 
                           conn.org_id, conn.id, batch_id)
                lineage_builder(conn.org_id, conn.id, batch_id)
                logger.info("✅ Lineage builder completed successfully")

                # --- Vector embeddings sync for this batch ---
                # Create a new database session for embedding sync to avoid idle-in-transaction timeout
                # The previous session may have been idle during the long-running lineage_builder
                embed_db: Session = SessionLocal()
                try:
                    # Count active/inactive lineage rows for this batch
                    active_count = embed_db.query(ColumnLevelLineage).filter(
                        ColumnLevelLineage.org_id == conn.org_id,
                        ColumnLevelLineage.connection_id == conn.id,
                        ColumnLevelLineage.batch_id == batch_id,
                        ColumnLevelLineage.is_active == 1,
                    ).count()

                    inactive_count = embed_db.query(ColumnLevelLineage).filter(
                        ColumnLevelLineage.org_id == conn.org_id,
                        ColumnLevelLineage.connection_id == conn.id,
                        ColumnLevelLineage.batch_id == batch_id,
                        ColumnLevelLineage.is_active == 0,
                    ).count()

                    collection_name = f"org_{conn.org_id}"
                    logger.info(
                        "🧠 Starting embedding upsert for collection=%s org_id=%s batch_id=%s (active=%d, inactive=%d)",
                        collection_name,
                        conn.org_id,
                        batch_id,
                        active_count,
                        inactive_count,
                    )

                    # Perform upsert + delete by batch
                    upsert_lineage_embeddings_by_batch(
                        org_id=str(conn.org_id),
                        batch_id=str(batch_id),
                    )

                    logger.info(
                        "✅ Embedding sync completed for collection=%s org_id=%s batch_id=%s",
                        collection_name,
                        conn.org_id,
                        batch_id,
                    )
                except Exception as embed_err:
                    logger.error(
                        "❌ Embedding sync failed for org_id=%s batch_id=%s: %s",
                        conn.org_id,
                        batch_id,
                        str(embed_err),
                    )
                finally:
                    embed_db.close()
            except Exception as lineage_error:
                logger.error("❌ Lineage builder failed: %s", str(lineage_error))
                import traceback
                logger.error("Lineage builder traceback: %s", traceback.format_exc())
                # Don't fail the entire crawl if lineage builder fails
                pass
        else:
            logger.info("⏭️  Skipping lineage builder: no new data fetched (%d rows)", len(to_insert))
            # --- Best-effort embedding backfill when no new data ---
            # Render uses ephemeral disk for Chroma; after deploy, embeddings may be empty
            # even though column_level_lineage has data. If we have active lineage rows for
            # this connection, upsert embeddings using the latest batch_id.
            embed_db: Session = SessionLocal()
            try:
                latest_batch = (
                    embed_db.query(ColumnLevelLineage.batch_id)
                    .filter(
                        ColumnLevelLineage.org_id == conn.org_id,
                        ColumnLevelLineage.connection_id == conn.id,
                        ColumnLevelLineage.is_active == 1,
                    )
                    .order_by(ColumnLevelLineage.created_at.desc())
                    .first()
                )
                if latest_batch and latest_batch[0]:
                    latest_batch_id = str(latest_batch[0])
                    active_count = (
                        embed_db.query(ColumnLevelLineage)
                        .filter(
                            ColumnLevelLineage.org_id == conn.org_id,
                            ColumnLevelLineage.connection_id == conn.id,
                            ColumnLevelLineage.batch_id == latest_batch_id,
                            ColumnLevelLineage.is_active == 1,
                        )
                        .count()
                    )
                    if active_count > 0:
                        collection_name = f"org_{conn.org_id}"
                        logger.info(
                            "🧠 Embedding backfill (no new data): collection=%s org_id=%s conn_id=%s batch_id=%s (active=%d)",
                            collection_name,
                            conn.org_id,
                            conn.id,
                            latest_batch_id,
                            active_count,
                        )
                        upsert_lineage_embeddings_by_batch(
                            org_id=str(conn.org_id),
                            batch_id=latest_batch_id,
                        )
                        logger.info(
                            "✅ Embedding backfill completed for collection=%s org_id=%s batch_id=%s",
                            collection_name,
                            conn.org_id,
                            latest_batch_id,
                        )
            except Exception as embed_backfill_err:
                logger.warning(
                    "⚠️  Embedding backfill skipped due to error (org=%s conn=%s): %s",
                    conn.org_id,
                    conn.id,
                    str(embed_backfill_err),
                )
            finally:
                embed_db.close()
                   
    except Exception as e:
        db.rollback()
        audit.status = "failed"
        audit.error_message = str(e)
        audit.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.exception("💥 Crawl failed: %s", str(e))


def polling_worker(stop_event: threading.Event, interval_seconds: int = 60):
    logger.info("🚀 Starting Snowflake crawler worker (interval: %d seconds)", interval_seconds)
    cycle_count = 0
    
    while not stop_event.is_set():
        cycle_count += 1
        start_ts = time.time()
        now = datetime.now(timezone.utc)
        
        # Create a session just for querying jobs
        db: Session = SessionLocal()
        jobs = []
        try:
            jobs = db.query(SnowflakeJob).filter(SnowflakeJob.is_active == True).all()
        except Exception as e:
            logger.exception("❌ Error querying jobs: %s", str(e))
        finally:
            # Close the session immediately after querying jobs
            # This prevents the connection from being idle during long-running crawl operations
            try:
                db.close()
            except Exception as close_err:
                logger.warning("⚠️  Error closing database session: %s", str(close_err))
        
        # Process jobs with separate sessions for each job
        if not jobs:
            logger.debug("⏸️  No active jobs found")
        else:
            due_jobs = 0
            for job in jobs:
                if job.cron_expression and _due_to_run(job.cron_expression, job.last_run_time, now):
                    due_jobs += 1
                    logger.info("⏰ Job due: %s (cron: %s)", str(job.id)[:8], job.cron_expression)
                    try:
                        # Create a fresh session for each job to avoid connection timeouts
                        job_db: Session = SessionLocal()
                        try:
                            run_crawl_for_connection(job_db, job, now)
                        finally:
                            try:
                                job_db.close()
                            except Exception as close_err:
                                logger.warning("⚠️  Error closing job database session: %s", str(close_err))
                    except Exception as job_err:
                        logger.exception("❌ Error processing job %s: %s", str(job.id)[:8], str(job_err))
            
            if due_jobs > 0:
                logger.info("✅ Processed %d due jobs", due_jobs)

        elapsed = time.time() - start_ts
        sleep_for = max(1.0, interval_seconds - elapsed)
        stop_event.wait(sleep_for)
    
    logger.info("🛑 Snowflake crawler worker stopped")


