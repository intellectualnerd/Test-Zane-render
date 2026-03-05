import logging
import json
import re
from datetime import datetime, timezone
from typing import Optional

import requests
from sqlalchemy.orm import Session
from sqlalchemy import and_
import time
import threading
from croniter import croniter

from app.utils.models import (
    DbtCloudConnection,
    DbtManifestNode,
    DbtJob,
    DbtCrawlAudit,
)
from app.database import SessionLocal
from sqlalchemy import text

logger = logging.getLogger("dbt_crawler")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _api_base(base_url: str) -> str:
    base = _normalize_base_url(base_url)
    return base if base.endswith("/api") else f"{base}/api"


def _headers(api_key: str):
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _get_json(url: str, headers: dict) -> Optional[dict]:
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _clean_manifest(raw_manifest: dict) -> dict:
    raw = json.dumps(raw_manifest)
    raw = re.sub(r"[\x00-\x1F\x7F]", " ", raw)
    return json.loads(raw)


def _extract_manifest_nodes(manifest: dict, run_id: str, finished_at: Optional[str]):
    rows = []
    child_map = manifest.get("child_map", {})
    for section in ["nodes", "sources"]:
        for uid, meta in manifest.get(section, {}).items():
            downstream = child_map.get(uid, [])
            downstream_structured = [{"unique_id": d, "run_id": run_id} for d in downstream]
            rows.append({
                "unique_id": uid,
                "run_id": str(run_id),
                "database": meta.get("database"),
                "schema": meta.get("schema"),
                "name": meta.get("name"),
                "package_name": meta.get("package_name"),
                "path": meta.get("path"),
                "original_file_path": meta.get("original_file_path"),
                "resource_type": meta.get("resource_type"),
                "raw_code": meta.get("raw_code"),
                "compiled_code": meta.get("compiled_code"),
                "downstream_models": downstream_structured,
                "last_successful_run_at": finished_at,
            })
    return rows


def _update_lineage_with_dbt_paths(db: Session, org_id: str) -> dict:
    """Match dbt manifest nodes with column_level_lineage and update dbt_model_file_path."""
    # Check if column_level_lineage has any data
    lineage_count = db.execute(text("SELECT COUNT(*) FROM column_level_lineage WHERE org_id = :org_id"), {"org_id": org_id}).scalar()
    if lineage_count == 0:
        logger.info("No data found in column_level_lineage for org %s", org_id)
        return {"matched": 0, "updated": 0, "total_lineage": 0}
    
    logger.info("Found %d rows in column_level_lineage for org %s", lineage_count, org_id)
    
    # Get all dbt manifest nodes for this org
    dbt_nodes = db.query(DbtManifestNode).filter(
        DbtManifestNode.org_id == org_id,
        DbtManifestNode.database.isnot(None),
        DbtManifestNode.schema.isnot(None),
        DbtManifestNode.name.isnot(None),
        DbtManifestNode.original_file_path.isnot(None)
    ).all()
    
    if not dbt_nodes:
        logger.info("No dbt manifest nodes found with required fields for org %s", org_id)
        return {"matched": 0, "updated": 0, "total_lineage": lineage_count}
    
    logger.info("Found %d dbt manifest nodes for matching", len(dbt_nodes))
    
    matched_count = 0
    updated_count = 0
    
    for node in dbt_nodes:
        # Match by case-insensitive database, schema, table
        result = db.execute(text("""
            UPDATE column_level_lineage 
            SET dbt_model_file_path = :file_path
            WHERE org_id = :org_id 
            AND LOWER(target_database) = LOWER(:database)
            AND LOWER(target_schema) = LOWER(:schema) 
            AND LOWER(target_table) = LOWER(:table)
            AND (dbt_model_file_path IS NULL OR dbt_model_file_path != :file_path)
        """), {
            "org_id": org_id,
            "database": node.database,
            "schema": node.schema,
            "table": node.name,
            "file_path": node.original_file_path
        })
        
        if result.rowcount > 0:
            matched_count += 1
            updated_count += result.rowcount
            logger.info("Matched dbt node %s (%s.%s.%s) -> %d lineage rows updated with path: %s", 
                       node.unique_id, node.database, node.schema, node.name, 
                       result.rowcount, node.original_file_path)
    
    logger.info("Lineage update completed: %d dbt nodes matched, %d total lineage rows updated", 
               matched_count, updated_count)
    
    return {"matched": matched_count, "updated": updated_count, "total_lineage": lineage_count}


def sync_dbt_metadata(db: Session, connection_id: str, job_id: Optional[str] = None) -> dict:
    """Fetch dbt manifest for the latest successful run and load nodes.

    Clears existing `dbt_manifest_nodes` for (org_id, connection_id), then inserts fresh rows.
    """
    conn: DbtCloudConnection | None = db.query(DbtCloudConnection).filter(
        DbtCloudConnection.id == connection_id,
        DbtCloudConnection.is_active == True,
    ).first()
    if not conn:
        raise ValueError("Connection not found or inactive")

    org_id = conn.org_id
    account_id = conn.account_id
    api = _api_base(conn.base_url)
    headers = _headers(conn.api_key)

    # Only fetch runs to locate last successful run; do not persist runs
    runs_url = f"{api}/v2/accounts/{account_id}/runs/?limit=50"
    runs_payload = _get_json(runs_url, headers) or {}
    runs = runs_payload.get("data", []) if isinstance(runs_payload, dict) else []

    # Last successful manifest
    def _is_success(status_val) -> bool:
        # dbt Cloud v2 may return a string or numeric; 10 is success code
        if status_val is None:
            return False
        if isinstance(status_val, (int, float)):
            return int(status_val) == 10
        return str(status_val).lower() == "success"

    # pick latest by finished_at among successful
    successful = [r for r in runs if _is_success(r.get("status")) and r.get("finished_at")]
    last_success = None
    if successful:
        last_success = max(successful, key=lambda r: r.get("finished_at") or "")

    # Start audit
    audit = DbtCrawlAudit(
        job_id=job_id,
        connection_id=connection_id,
        status="running",
    )
    db.add(audit)
    db.flush()

    # Clear previous manifest nodes for this connection/org
    db.query(DbtManifestNode).filter(and_(DbtManifestNode.org_id == org_id, DbtManifestNode.connection_id == connection_id)).delete(synchronize_session=False)
    if last_success:
        run_id = str(last_success.get("id"))
        finished_at = last_success.get("finished_at")
        manifest_url = f"{api}/v2/accounts/{account_id}/runs/{run_id}/artifacts/manifest.json"
        manifest = _get_json(manifest_url, headers)
        if manifest:
            manifest = _clean_manifest(manifest)
            rows = _extract_manifest_nodes(manifest, run_id, finished_at)
            inserted = 0
            for row in rows:
                db.add(DbtManifestNode(
                    org_id=org_id,
                    connection_id=connection_id,
                    run_id=row["run_id"],
                    unique_id=row["unique_id"],
                    database=row.get("database"),
                    schema=row.get("schema"),
                    name=row.get("name"),
                    package_name=row.get("package_name"),
                    path=row.get("path"),
                    original_file_path=row.get("original_file_path"),
                    resource_type=row.get("resource_type"),
                    raw_code=row.get("raw_code"),
                    compiled_code=row.get("compiled_code"),
                    downstream_models=row.get("downstream_models"),
                    last_successful_run_at=finished_at,
                    synced_at=datetime.now(timezone.utc),
                ))
                inserted += 1
            logger.info("Inserted %s manifest nodes for run %s", inserted, run_id)
            audit.status = "success"
            audit.nodes_inserted = inserted
        else:
            logger.warning("No manifest found for run %s", run_id)
            audit.status = "failed"
            audit.error_message = f"No manifest for run {run_id}"

    audit.finished_at = datetime.now(timezone.utc)
    db.commit()
    
    # Update lineage with dbt file paths
    lineage_result = _update_lineage_with_dbt_paths(db, org_id)
    
    count = db.query(DbtManifestNode).filter(and_(DbtManifestNode.org_id == org_id, DbtManifestNode.connection_id == connection_id)).count()
    return {
        "nodes": count, 
        "picked_run_id": last_success.get("id") if last_success else None,
        "lineage_matched": lineage_result["matched"],
        "lineage_updated": lineage_result["updated"],
        "total_lineage": lineage_result["total_lineage"]
    }


def _due_to_run(cron_expr: str, base_dt: datetime, now: datetime) -> bool:
    try:
        itr = croniter(cron_expr, base_dt)
        next_time = itr.get_next(datetime)
        return next_time <= now
    except Exception:
        logger.warning("Invalid cron expression: %s", cron_expr)
        return False


def polling_worker(stop_event: threading.Event, interval_seconds: int = 600):
    logger.info("🚀 Starting dbt crawler worker (interval: %d seconds)", interval_seconds)
    while not stop_event.is_set():
        start_ts = time.time()
        now = datetime.now(timezone.utc)
        db: Session = SessionLocal()
        try:
            jobs = db.query(DbtJob).filter(DbtJob.is_active == True).all()
            for job in jobs:
                base_dt = job.last_run_time or job.created_at or now
                if job.cron_expression and _due_to_run(job.cron_expression, base_dt, now):
                    logger.info("⏰ dbt job due: %s (cron: %s)", str(job.id)[:8], job.cron_expression)
                    try:
                        res = sync_dbt_metadata(db, job.connection_id, str(job.id))
                        job.last_run_time = now
                        db.commit()
                        logger.info("✅ dbt sync done: %s", res)
                    except Exception as e:
                        db.rollback()
                        logger.exception("❌ dbt sync failed: %s", str(e))
        except Exception as e:
            logger.exception("❌ dbt worker error: %s", str(e))
        finally:
            db.close()

        elapsed = time.time() - start_ts
        sleep_for = max(1.0, interval_seconds - elapsed)
        stop_event.wait(sleep_for)
    logger.info("🛑 dbt crawler worker stopped")

