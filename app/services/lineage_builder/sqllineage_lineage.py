import pandas as pd
import subprocess
import csv
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import Session
from sqllineage.runner import LineageRunner
import os
import re
from collections import defaultdict
from dotenv import load_dotenv
import ast
from sqlglot import parse_one, exp
import logging
import sys
import os
from datetime import datetime
import uuid

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils.models import (
    SnowflakeQueryRecord,
    InformationSchemacolumns,
    LineageLoadWatermark,
    ColumnLevelLineage
    # FilterClauseColumnLineage
)

# Logging Setup
logger = logging.getLogger("lineage.sqllineage_lineage")

# Suppress third-party library logging
logging.getLogger("sqllineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot").setLevel(logging.WARNING)
logging.getLogger("sqlglot.lineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot.optimizer").setLevel(logging.WARNING)


# Database connection
def get_pg_engine():
    try:
        load_dotenv()
        engine = create_engine(
            os.getenv('DATABASE_URL')
        )
        return engine
    except Exception as e:
        logger.error("Failed to create database engine: %s", e, exc_info=True)
        raise

# Query to fetch data
def fetch_query_access_history_and_information_schema_columns(engine, org_id, conn_id, batch_id):
    with Session(engine) as session:
        # Get last processed timestamp (or default to very old date)
        last_processed = (
            session.query(func.max(LineageLoadWatermark.last_processed_at))
            .filter_by(connection_id=conn_id, org_id=org_id)
            .scalar()
        ) or datetime(1900, 1, 1)

        logger.info("Filtering queries with last_processed timestamp: %s (org_id=%s, conn_id=%s, batch_id=%s)", 
                   last_processed, org_id, conn_id, batch_id)
        # Fetch query history based on watermark (do not limit by batch_id)
        # This prevents reprocessing of already-processed queries and avoids duplicate lineage across batches
        query_history = (
            session.query(SnowflakeQueryRecord)
            .filter_by(connection_id=conn_id, org_id=org_id)
            .filter(SnowflakeQueryRecord.created_at > last_processed)
            .all()
        )

        # Fetch information schema columns
        info_schema = (
            session.query(InformationSchemacolumns)
            .filter_by(connection_id=conn_id, org_id=org_id)
            .all()
        )

        # Fetch historical column level lineage
        historical_column_level_lineage = (
            session.query(ColumnLevelLineage)
            .filter_by(connection_id=conn_id, org_id=org_id, is_active=1, is_filter_clause=0)
            .all()
        )

        # Fetch historical filter clause column lineage
        historical_filter_clause_column_lineage = (
            session.query(ColumnLevelLineage)
            .filter_by(connection_id=conn_id, org_id=org_id, is_active=1, is_filter_clause=1)
            .all()
        )

        # Convert ORM objects → DataFrame
        if query_history:
            query_history_df = pd.DataFrame([r.__dict__ for r in query_history])
            query_history_df = query_history_df.drop("_sa_instance_state", axis=1, errors="ignore")
        else:
            query_history_df = pd.DataFrame(columns=[col.name for col in SnowflakeQueryRecord.__table__.columns])

        if info_schema:
            info_schema_df = pd.DataFrame([r.__dict__ for r in info_schema])
            info_schema_df = info_schema_df.drop("_sa_instance_state", axis=1, errors="ignore")
        else:
            info_schema_df = pd.DataFrame(columns=[col.name for col in InformationSchemacolumns.__table__.columns])

        if historical_column_level_lineage:
            historical_column_level_lineage_df = pd.DataFrame([r.__dict__ for r in historical_column_level_lineage])
            historical_column_level_lineage_df = historical_column_level_lineage_df.drop("_sa_instance_state", axis=1, errors="ignore")
        else:
            historical_column_level_lineage_df = pd.DataFrame(columns=[col.name for col in ColumnLevelLineage.__table__.columns])

        if historical_filter_clause_column_lineage:
            historical_filter_clause_column_lineage_df = pd.DataFrame([r.__dict__ for r in historical_filter_clause_column_lineage])
            historical_filter_clause_column_lineage_df = historical_filter_clause_column_lineage_df.drop("_sa_instance_state", axis=1, errors="ignore")
        else:
            historical_filter_clause_column_lineage_df = pd.DataFrame(columns=[col.name for col in ColumnLevelLineage.__table__.columns])

        return query_history_df, info_schema_df, historical_column_level_lineage_df, historical_filter_clause_column_lineage_df


# Function to safely parse and convert to string
def parse_and_convert_to_sql(query):
    try:
        parsed = parse_one(query, read="snowflake")
        return parsed.sql()
    except Exception as e:
        return logger.error("ERROR: %s", e, exc_info=True)


# Detect and Replace Named Parameters
def detect_and_replace_named_parameters(query: str, static_value: str = "null"):
    try:
        # Protect string literals inside single quotes
        string_literals = re.findall(r"'[^']*'", query)
        protected_literals = {s: f"__STRING_LITERAL_{i}__" for i, s in enumerate(string_literals)}
        
        for orig, placeholder in protected_literals.items():
            query = query.replace(orig, placeholder)

        # Regex to match :param but not ::type
        param_pattern = r"(?<!:):([a-zA-Z_][a-zA-Z0-9_]*)"
        matches = re.findall(param_pattern, query)

        if matches:
            query = re.sub(param_pattern, static_value, query)

        # Restore string literals
        for orig, placeholder in protected_literals.items():
            query = query.replace(placeholder, orig)

        return query
    except Exception as e:
        logger.error("Error in detect_and_replace_named_parameters: %s", e, exc_info=True)
        return query


# Combining Temp View with Merge/Insert Query
def combine_queries_by_session(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df['query_text'] = df['query_text'].apply(parse_and_convert_to_sql).str.strip()
        df['query_type'] = df['query_type'].str.upper().str.strip()

        create_view_df = df[df['query_type'] == 'CREATE_VIEW'].copy()
        merge_df = df[df['query_type'] == 'MERGE'].copy()
        insert_df = df[df['query_type'] == 'INSERT'].copy()

        # Track used merge/insert query_ids
        used_merge_ids = set()
        used_insert_ids = set()

        combined_queries = []

        for _, create_row in create_view_df.iterrows():
            try:
                combined_query_id = []
                create_query = create_row['query_text']
                session_id = create_row['session_id']
                base_objects_accessed = create_row['base_objects_accessed']
                database_name = create_row['database_name']
                schema_name = create_row['schema_name']

                combined_query_id.append(create_row['query_id'])

                # Extract view name
                match = re.search(
                    r"CREATE\s+OR\s+REPLACE\s+TEMPORARY\s+VIEW\s+([^\s]+)",
                    create_query.strip(),
                    flags=re.IGNORECASE
                )
                if not match:
                    continue

                view_name = match.group(1).strip()

                combined_sql = None
                query_type = None

                #First, try to find matching MERGE INTO
                merge_candidates = merge_df[merge_df['session_id'] == session_id]
                for _, merge_row in merge_candidates.iterrows():
                    merge_query = merge_row['query_text']
                    if view_name.lower() in merge_query.lower():
                        combined_sql = create_query.strip() + ";\n" + merge_query.strip()
                        combined_query_id.append(merge_row['query_id'])
                        used_merge_ids.add(merge_row['query_id'])
                        objects_modified = merge_row['objects_modified']
                        query_type = 'create_view_and_merge'
                        break

                #If no MERGE match, try matching INSERT INTO
                if not combined_sql:
                    insert_candidates = insert_df[insert_df['session_id'] == session_id]
                    for _, insert_row in insert_candidates.iterrows():
                        insert_query = insert_row['query_text']
                        if view_name.lower() in insert_query.lower():
                            combined_sql = create_query.strip() + ";\n" + insert_query.strip()
                            combined_query_id.append(insert_row['query_id'])
                            used_insert_ids.add(insert_row['query_id'])
                            objects_modified = insert_row['objects_modified']
                            query_type = 'create_view_and_insert'
                            break

                if combined_sql:
                    combined_queries.append({
                        'query_id': combined_query_id,
                        'query_text': combined_sql,
                        'query_type': query_type,
                        'session_id': session_id,
                        'base_objects_accessed': base_objects_accessed,
                        'objects_modified': objects_modified,
                        'database_name': database_name,
                        'schema_name': schema_name
                    })
            except Exception as inner_e:
                logger.error(
                            "Error processing create view row (query_id=%s): %s",
                            create_row.get("query_id"),
                            inner_e,
                            exc_info=True
                        )


        #Add unused MERGE and INSERT queries as it is
        unused_merge_df = merge_df[~merge_df['query_id'].isin(used_merge_ids)].copy()
        unused_insert_df = insert_df[~insert_df['query_id'].isin(used_insert_ids)].copy()

        unused_merge_df['query_type'] = 'standalone_merge'
        unused_insert_df['query_type'] = 'standalone_insert'

        combined_df = pd.DataFrame(combined_queries)
        final_df = pd.concat([combined_df, unused_merge_df, unused_insert_df], ignore_index=True)

        final_df['start_time'] = final_df['start_time'].dt.tz_localize(None)
        return final_df

    except Exception as e:
        logging.error("An error occurred while processing queries: %s", e, exc_info=True)
        return pd.DataFrame()


# Function to find the final target by traversing the chain
def find_all_final_sources(start, mapping, visited=None):
    try:
        if visited is None:
            visited = set()
        visited.add(start)

        if start not in mapping:
            return [start]

        roots = set()
        for src in mapping[start]:
            if src not in visited:
                sub_roots = find_all_final_sources(src, mapping, visited.copy())
                roots.update(sub_roots)

        return list(roots)
    except Exception as e:
        logger.error("Error in find_all_final_sources (start=%s): %s", start, e, exc_info=True)
        return []

# Parse fully qualified column names
def parse_full_column(qualified_col):
    try:
        parts = qualified_col.split(".")
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
        elif len(parts) == 3:
            return None, parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return None, None, parts[0], parts[1]
        else:
            return None, None, None, qualified_col
    except Exception as e:
        logger.error("Error parsing qualified column: %s", qualified_col, exc_info=True)
        return (None, None, None, qualified_col)


def sanitize_identifier(identifier: str) -> str:
    """
    Convert Snowflake-style quoted identifiers into safe names.
    Example:
      "row#" -> rownum_row
      "First Name" -> first_name
      "123abc" -> col_123abc
    """
    s = identifier.strip('"')  # remove quotes
    s = s.replace(" ", "_")    # spaces → underscores
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s)  # replace special chars
    if re.match(r'^\d', s):   # if starts with number
        s = "col_" + s
    return s.lower()

def preprocess_sql(sql: str) -> str:
    """
    Dynamically clean Snowflake SQL for sqlglot parsing.
    """
    sql = re.sub(r'array\s*\(([^)]+)\)', r'\1', sql, flags=re.IGNORECASE)
    sql = re.sub(r'""([^""]+)""', lambda m: sanitize_identifier(m.group(1)), sql)
    return sql


def resolve_cte_column_source_issue_recursively(cte_name: str, column_name: str, sql: str) -> list[dict] | None:
    try:
        # Preprocess before parsing
        sql = preprocess_sql(sql)
        parsed = parse_one(sql, read="snowflake")

        # Collect all CTEs
        def get_all_cte_definitions(parsed_expr):
            return {cte.alias_or_name: cte.this for cte in parsed_expr.find_all(exp.CTE)}

        # Collect all subqueries with alias
        def get_all_subquery_definitions(parsed_expr):
            subquery_map = {}
            for subq in parsed_expr.find_all(exp.Subquery):
                alias = subq.alias_or_name
                if alias:
                    subquery_map[alias] = subq.this
            return subquery_map

        # Get all underlying physical tables
        def extract_source_tables_recursive(expr, cte_map, visited=None):
            visited = visited or set()
            tables = []
            for table in expr.find_all(exp.Table):
                table_name = table.name
                if table_name in cte_map and table_name not in visited:
                    visited.add(table_name)
                    tables += extract_source_tables_recursive(
                        cte_map[table_name], cte_map, visited
                    )
                elif table_name not in cte_map:
                    tables.append(
                        {
                            "catalog": table.catalog or "",
                            "db": table.db or "",
                            "table": table.name,
                        }
                    )
            return tables

        # Helper function to extract column lineage from a column expression
        def extract_column_lineage(column_expr, cte_expr, cte_map, current_cte, visited):
            # unwrap nested casts/functions until we reach a column
            while (
                hasattr(column_expr, "this")
                and not isinstance(column_expr, exp.Column)
            ):
                column_expr = column_expr.this

            if isinstance(column_expr, exp.Column):
                table_prefix = column_expr.table
                if table_prefix:
                    for table in cte_expr.find_all(exp.Table):
                        if (
                            table.alias_or_name == table_prefix
                            or table.name == table_prefix
                        ):
                            if table.name in cte_map and table.name not in visited:
                                return find_column_sources(
                                    cte_map[table.name],
                                    cte_map,
                                    column_expr.name,
                                    table.name,
                                    visited | {current_cte},
                                )
                            else:
                                return [
                                    {
                                        "source_database": table.catalog or "",
                                        "source_schema": table.db or "",
                                        "source_table": table.name,
                                        "source_column": column_expr.name,
                                    }
                                ]
                else:
                    tables = extract_source_tables_recursive(
                        cte_expr, cte_map, visited={current_cte}
                    )
                    return [
                        {
                            "source_database": t["catalog"],
                            "source_schema": t["db"],
                            "source_table": t["table"],
                            "source_column": column_expr.name,
                        }
                        for t in tables
                    ]
            return []

        # Helper function to search for columns in filter clauses
        def search_columns_in_filter_clauses(select_expr, cte_expr, cte_map, target_col, current_cte, visited):
            results = []

            # Search in WHERE clause
            if select_expr.args.get("where"):
                where_expr = select_expr.args["where"]
                for col in where_expr.find_all(exp.Column):
                    if col.name.lower() == target_col.lower():
                        lineage = extract_column_lineage(col, cte_expr, cte_map, current_cte, visited)
                        if lineage:
                            results.extend(lineage)

            # Search in HAVING clause
            if select_expr.args.get("having"):
                having_expr = select_expr.args["having"]
                for col in having_expr.find_all(exp.Column):
                    if col.name.lower() == target_col.lower():
                        lineage = extract_column_lineage(col, cte_expr, cte_map, current_cte, visited)
                        if lineage:
                            results.extend(lineage)

            # Search in GROUP BY clause
            if select_expr.args.get("group"):
                group_expr = select_expr.args["group"]
                for col in group_expr.find_all(exp.Column):
                    if col.name.lower() == target_col.lower():
                        lineage = extract_column_lineage(col, cte_expr, cte_map, current_cte, visited)
                        if lineage:
                            results.extend(lineage)

            # Search in ORDER BY clause
            if select_expr.args.get("order"):
                order_expr = select_expr.args["order"]
                for col in order_expr.find_all(exp.Column):
                    if col.name.lower() == target_col.lower():
                        lineage = extract_column_lineage(col, cte_expr, cte_map, current_cte, visited)
                        if lineage:
                            results.extend(lineage)

            # Search in JOIN conditions
            for join in select_expr.find_all(exp.Join):
                if join.args.get("on"):
                    on_expr = join.args["on"]
                    for col in on_expr.find_all(exp.Column):
                        if col.name.lower() == target_col.lower():
                            lineage = extract_column_lineage(col, cte_expr, cte_map, current_cte, visited)
                            if lineage:
                                results.extend(lineage)

            return results

        # Find column lineage
        def find_column_sources(cte_expr, cte_map, target_col, current_cte, visited=None):
            visited = visited or set()

            for select in cte_expr.find_all(exp.Select):
                # First, search in SELECT expressions (existing functionality)
                for expression in select.expressions:
                    alias = expression.alias_or_name

                    # Match alias or raw column
                    if (alias and alias.lower() == target_col.lower()) or (
                        isinstance(expression, exp.Column)
                        and expression.name.lower() == target_col.lower()
                    ):
                        column_expr = (
                            expression.this if isinstance(expression, exp.Alias) else expression
                        )
                        lineage = extract_column_lineage(column_expr, cte_expr, cte_map, current_cte, visited)
                        if lineage:
                            return lineage

                # Then, search in filter clauses (new functionality)
                filter_results = search_columns_in_filter_clauses(select, cte_expr, cte_map, target_col, current_cte, visited)
                if filter_results:
                    return filter_results

            # 🔄 Recurse into other CTEs or subqueries if column not found directly
            for sub_cte in [
                t.name
                for t in cte_expr.find_all(exp.Table)
                if t.name in cte_map and t.name not in visited
            ]:
                result = find_column_sources(
                    cte_map[sub_cte], cte_map, target_col, sub_cte, visited | {current_cte}
                )
                if result:
                    return result

            return []

        # Main execution
        cte_map = get_all_cte_definitions(parsed)
        subquery_map = get_all_subquery_definitions(parsed)

        # Merge CTEs + subqueries into one map
        all_query_map = {**cte_map, **subquery_map}

        # If cte_name is None, search across all CTEs/subqueries
        if cte_name is None:
            for current_cte_name, cte_expr in all_query_map.items():
                result = find_column_sources(cte_expr, all_query_map, column_name, current_cte_name)
                if result:
                    return result
            return None

        # If specific CTE name is provided
        if cte_name not in all_query_map:
            return None

        result = find_column_sources(all_query_map[cte_name], all_query_map, column_name, cte_name)

        return result if result else None

    except Exception as e:
        print(f"Error: {e}")
        return None
        
# Find Relevant table name for given column
def find_column_table_name(qualified_names, column_name, base_objects_accessed):
    try:
        column_name_lower = column_name.lower()
        qualified_names_lower = [q.lower() for q in qualified_names]
        
        for table in base_objects_accessed:
            object_name = table['objectName']
            object_name_lower = object_name.lower()
            
            if object_name_lower in qualified_names_lower:
                for col in table['columns']:
                    if col['columnName'].lower() == column_name_lower:
                        return object_name  # return original table name 

        return None
    except Exception as e:
        logger.error("Error in find_column_table_name: %s", e, exc_info=True)
        return None


# Parse lineage output and write to CSV
def parse_lineage_text(query_id, cleaned_query, query_type, session_id, base_objects_accessed, database_name, schema_name, information_schema_columns_df, lineage_text):
    rows = []
    try:
        for line in lineage_text.strip().splitlines():
            line = line.strip()

            if "<-" in line:
                parts = [part.strip() for part in line.split("<-")]
                target_column = parts[0]
                source_column = parts[-1]
                dependency_score = 1
            else:
                target_column = line.strip()
                source_column = None
                dependency_score = 0


            rows.append({
            "target_column": target_column,
            "source_column": source_column,
            "dependency_score": dependency_score
        })


        rows_df = pd.DataFrame(rows)
        rows_df['source_column_lower'] = rows_df['source_column'].astype(str).str.lower()
        rows_df['target_column_lower'] = rows_df['target_column'].astype(str).str.lower()

        # Build a dictionary mapping target → source
        target_to_sources = defaultdict(list)
        for _, row in rows_df.iterrows():
            target_col = row['target_column_lower']
            source_col = row['source_column_lower']
            if pd.notna(source_col):
                target_to_sources[target_col].append(source_col)

        # Find root targets
        all_sources = set(rows_df['source_column_lower'])
        all_targets = set(rows_df['target_column_lower'])
        root_targets = all_targets - all_sources

        # Construct final source → final target mapping
        collapsed_edges = []
        for root in root_targets:
            final_sources = find_all_final_sources(root, target_to_sources)
            for final_source in final_sources:
                src_db, src_schema, src_table, src_column = parse_full_column(final_source)
                tgt_db, tgt_schema, tgt_table, tgt_column = parse_full_column(root)

                if tgt_db is None:
                    tgt_db = database_name.lower()
                if tgt_schema is None:
                    tgt_schema = schema_name.lower()

                if src_db is None or src_schema is None:
                    for obj in base_objects_accessed:
                        object_name = obj.get('objectName', '').strip().lower()

                        if not object_name or '.' not in object_name:
                            continue

                        parts = object_name.split('.')
                        if len(parts) != 3:
                            continue

                        database, schema, table = parts

                        for col in obj.get('columns', []):
                            if str(col.get('columnName', '')).strip().lower() == src_column:
                                if src_table:
                                    if src_table == table:
                                        src_db = database
                                        src_schema = schema
                                else:
                                    src_db = database
                                    src_schema = schema
                                    src_table = table

                if src_column == '*' and tgt_column == '*':
                    if src_db and src_schema and src_table and tgt_db and tgt_schema and tgt_table:
                        source_columns_df = information_schema_columns_df[
                            (information_schema_columns_df['table_catalog'].str.lower() == src_db) &
                            (information_schema_columns_df['table_schema'].str.lower() == src_schema) &
                            (information_schema_columns_df['table_name'].str.lower() == src_table)
                        ]

                        target_columns_df = information_schema_columns_df[
                            (information_schema_columns_df['table_catalog'].str.lower() == tgt_db) &
                            (information_schema_columns_df['table_schema'].str.lower() == tgt_schema) &
                            (information_schema_columns_df['table_name'].str.lower() == tgt_table)
                        ]

                        result_df = pd.merge(source_columns_df, target_columns_df, how='inner', on='ordinal_position',
                                            suffixes=('_src', '_tgt'))

                        if not result_df.empty:
                            for _, row in result_df.iterrows():
                                collapsed_edges.append({
                                    "source_database": src_db,
                                    "source_schema": src_schema,
                                    "source_table": src_table,
                                    "source_column": row['column_name_src'].lower(),
                                    "target_database": tgt_db,
                                    "target_schema": tgt_schema,
                                    "target_table": tgt_table,
                                    "target_column": row['column_name_tgt'].lower(),
                                    "query_id": query_id,
                                    "query_type": query_type,
                                    "session_id": session_id,
                                    "dbt_model_file_path": None,
                                    "dependency_score": dependency_score
                                })
                        else:
                            logger.warning("No column mapping found for * → * between %s and %s for query_id %s", src_table, tgt_table, query_id)
                    else:
                        logger.warning("Incomplete metadata for wildcard mapping (source or target is None) for query_id %s", query_id)
                    continue

                if src_db is None and src_schema is None:
                    cte_result = resolve_cte_column_source_issue_recursively(src_table, src_column, cleaned_query)

                    if cte_result:
                        if len(cte_result) == 1:
                            src_db = cte_result[0]['source_database'].lower()
                            src_schema = cte_result[0]['source_schema'].lower()
                            src_table = cte_result[0]['source_table'].lower()
                        else:
                            qualified_names = [
                                    f"{entry['source_database']}.{entry['source_schema']}.{entry['source_table']}"
                                    for entry in cte_result
                                ]
                            relevant_qualified_table_name = find_column_table_name(qualified_names, src_column, base_objects_accessed)
                            relevant_qualified_table_name_list = relevant_qualified_table_name.split('.')
                            src_db = relevant_qualified_table_name_list[0].lower()
                            src_schema = relevant_qualified_table_name_list[1].lower()
                            src_table = relevant_qualified_table_name_list[2].lower()

                collapsed_edges.append({
                    "source_database": src_db,
                    "source_schema": src_schema,
                    "source_table": src_table,
                    "source_column": src_column,
                    "target_database": tgt_db,
                    "target_schema": tgt_schema,
                    "target_table": tgt_table,
                    "target_column": tgt_column,
                    "query_id": query_id,
                    "query_type": query_type,
                    "session_id": session_id,
                    "dbt_model_file_path": None,
                    "dependency_score": dependency_score
                })

        return collapsed_edges
    
    except Exception as e:
        logger.error("Error parsing lineage for query_id %s: %s", query_id, e, exc_info=True)