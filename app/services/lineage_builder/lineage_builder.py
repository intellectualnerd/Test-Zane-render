
import sys
import os
import numpy as np
import uuid
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import pandas as pd
import ast
import subprocess
import re
from sqlalchemy import create_engine, text, update, func
from sqlalchemy.orm import Session
import uuid
from datetime import datetime, timezone
from collections import defaultdict

# Handle imports for both direct execution and package import
try:
    from .sql_lineage_builder import build_lineage
    from . import sqllineage_lineage
    from .filter_clause_columns import get_dependent_columns
    from app.utils.models import (
        ColumnLevelLineage,
        # FilterClauseColumnLineage,
        LineageLoadWatermark
    )
except ImportError:
    # When running directly, use absolute imports
    from sql_lineage_builder import build_lineage
    import sqllineage_lineage
    from filter_clause_columns import get_dependent_columns
    # Add the app directory to path for direct execution
    import sys
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    from utils.models import (
        ColumnLevelLineage,
        # FilterClauseColumnLineage,
        LineageLoadWatermark
    )

logger = logging.getLogger("lineage")   # Named logger instead of root
logger.setLevel(logging.INFO)

# Set all lineage loggers to INFO level to reduce noise
logging.getLogger("lineage").setLevel(logging.INFO)

# Suppress third-party library logging
logging.getLogger("sqllineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot").setLevel(logging.WARNING)
logging.getLogger("sqlglot.lineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot.optimizer").setLevel(logging.WARNING)
logging.getLogger("sqlglot.optimizer.scope").setLevel(logging.WARNING)

# Additional suppression for potential subquery scope messages
import warnings
warnings.filterwarnings("ignore", message="Unknown subquery scope")

if not logger.handlers:  # prevent duplicate handlers
    file_handler = logging.FileHandler("lineage.log", mode="a", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# Global list to collect all lineage results
all_lineages = []
all_edges = []

def extract_sql_lineage_source_to_target(sql: str) -> dict:
    """
    Enhanced extract SQL lineage showing both source-to-temp-view and temp-view-to-target mappings.
    This version recursively traces through multiple CTE levels to find missing column lineage.
    """
    lineage = build_lineage(sql, dialect="snowflake", enhanced_mode=True)

    # Build a comprehensive mapping graph from all lineage mappings
    all_mappings = lineage.get("source_to_target", [])
    
    # Create a graph structure: target -> [sources]
    lineage_graph = defaultdict(list)
    reverse_graph = defaultdict(list)  # source -> [targets]
    
    for mapping in all_mappings:
        parts = mapping.split(" <- ")
        if len(parts) == 2:
            target = parts[0].lower()
            source = parts[1].lower()
            lineage_graph[target].append(source)
            reverse_graph[source].append(target)
    
    # Separate temp view mappings and target mappings
    temp_view_mappings = []
    target_mappings = []
    
    for mapping in all_mappings:
        parts = mapping.split(" <- ")
        if len(parts) == 2:
            left_side = parts[0].lower()
            right_side = parts[1].lower()

            if "__dbt_tmp" in left_side and "__dbt_tmp" not in right_side:
                temp_view_mappings.append(mapping)
            elif "__dbt_tmp" in right_side and "__dbt_tmp" not in left_side:
                target_mappings.append(mapping)
            else:
                if len(temp_view_mappings) < len(target_mappings):
                    temp_view_mappings.append(mapping)
                else:
                    target_mappings.append(mapping)
    
    # Build source_to_temp_view mapping
    source_to_temp_view = {}
    for mapping in temp_view_mappings:
        parts = mapping.split(" <- ")
        if len(parts) == 2:
            target_col = parts[0].lower()
            source_col = parts[1].lower()
            source_to_temp_view.setdefault(target_col, []).append(source_col)
    
    # Build temp_view_to_target mapping
    temp_view_to_target = {}
    for mapping in target_mappings:
        parts = mapping.split(" <- ")
        if len(parts) == 2:
            target_col = parts[0].lower()
            source_col = parts[1].lower()
            temp_view_to_target.setdefault(target_col, []).append(source_col)
    
    # Memoization cache to avoid recalculating the same columns
    resolution_cache = {}
    
    def resolve_column_recursively(target_col: str, visited: set = None, max_depth: int = 10, is_top_level: bool = False) -> list:
        """
        Recursively resolve a column through the lineage graph.
        This handles cases where columns go through multiple CTE levels.
        Uses memoization to avoid redundant calculations.
        """
        # Check cache first (only if this is a top-level call)
        if is_top_level and target_col in resolution_cache:
            return resolution_cache[target_col]
        
        if visited is None:
            visited = set()
            is_top_level = True
        
        if target_col in visited or max_depth <= 0:
            return []
        
        visited.add(target_col)
        sources = []
        
        # Check the general lineage graph first (most comprehensive)
        if target_col in lineage_graph:
            for intermediate_col in lineage_graph[target_col]:
                if intermediate_col not in visited:
                    # Check if this is a base table (not a CTE/temp view)
                    if "__dbt_tmp" not in intermediate_col.lower() and "." in intermediate_col:
                        # Likely a base table, but check if it has further mappings
                        if intermediate_col in reverse_graph:
                            # This column is used by other columns, might be intermediate
                            # Use cache if available, otherwise recurse
                            if intermediate_col in resolution_cache:
                                deeper_sources = resolution_cache[intermediate_col]
                            else:
                                deeper_sources = resolve_column_recursively(intermediate_col, visited.copy(), max_depth - 1, False)
                            if deeper_sources:
                                sources.extend(deeper_sources)
                            else:
                                sources.append(intermediate_col)
                        else:
                            # No reverse mappings, likely a final source
                            sources.append(intermediate_col)
                    else:
                        # It's a temp view/CTE, resolve it recursively
                        # Use cache if available
                        if intermediate_col in resolution_cache:
                            recursive_sources = resolution_cache[intermediate_col]
                        else:
                            recursive_sources = resolve_column_recursively(intermediate_col, visited.copy(), max_depth - 1, False)
                        if recursive_sources:
                            sources.extend(recursive_sources)
        
        # Also check temp_view_to_target and source_to_temp_view for explicit mappings
        if target_col in temp_view_to_target:
            for tmp_col in temp_view_to_target[target_col]:
                if tmp_col not in visited:
                    # Check if tmp_col has mappings in source_to_temp_view
                    if tmp_col in source_to_temp_view:
                        for src_col in source_to_temp_view[tmp_col]:
                            if src_col not in visited:
                                # Check if this source is a base table or needs further resolution
                                if "__dbt_tmp" not in src_col.lower() and "." in src_col:
                                    if src_col in reverse_graph:
                                        # Might have further mappings
                                        if src_col in resolution_cache:
                                            deeper_sources = resolution_cache[src_col]
                                        else:
                                            deeper_sources = resolve_column_recursively(src_col, visited.copy(), max_depth - 1, False)
                                        if deeper_sources:
                                            sources.extend(deeper_sources)
                                        else:
                                            sources.append(src_col)
                                    else:
                                        sources.append(src_col)
                                else:
                                    # Recursively resolve
                                    if src_col in resolution_cache:
                                        recursive_sources = resolution_cache[src_col]
                                    else:
                                        recursive_sources = resolve_column_recursively(src_col, visited.copy(), max_depth - 1, False)
                                    if recursive_sources:
                                        sources.extend(recursive_sources)
                    else:
                        # Recursively resolve the intermediate column
                        if tmp_col in resolution_cache:
                            recursive_sources = resolution_cache[tmp_col]
                        else:
                            recursive_sources = resolve_column_recursively(tmp_col, visited.copy(), max_depth - 1, False)
                        if recursive_sources:
                            sources.extend(recursive_sources)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for src in sources:
            if src not in seen:
                seen.add(src)
                unique_sources.append(src)
        
        # Cache the result if this was a top-level call
        if is_top_level:
            resolution_cache[target_col] = unique_sources
        
        return unique_sources
    
    # Build final lineage with recursive resolution
    final_lineage = {}
    
    # First, process all target mappings (optimized to use cache)
    for target, tmp_list in temp_view_to_target.items():
        sources = []
        for tmp in tmp_list:
            if tmp in source_to_temp_view:
                # Direct mapping found
                for src in source_to_temp_view[tmp]:
                    # Check if this source needs further resolution
                    if "__dbt_tmp" not in src.lower() and "." in src:
                        # Check if it's truly a base table or has further mappings
                        if src in reverse_graph:
                            # Has reverse mappings, might be intermediate
                            # Use cache if available
                            if src in resolution_cache:
                                deeper_sources = resolution_cache[src]
                            else:
                                deeper_sources = resolve_column_recursively(src, None, 10, True)
                            if deeper_sources:
                                sources.extend(deeper_sources)
                            else:
                                sources.append(src)
                        else:
                            sources.append(src)
                    else:
                        # Recursively resolve (will use cache)
                        if src in resolution_cache:
                            recursive_sources = resolution_cache[src]
                        else:
                            recursive_sources = resolve_column_recursively(src, None, 10, True)
                        if recursive_sources:
                            sources.extend(recursive_sources)
            else:
                # Try recursive resolution for missing intermediate mappings (will use cache)
                if tmp in resolution_cache:
                    recursive_sources = resolution_cache[tmp]
                else:
                    recursive_sources = resolve_column_recursively(tmp, None, 10, True)
                if recursive_sources:
                    sources.extend(recursive_sources)
        
        # If still no sources found, try resolving the target directly from lineage graph
        if not sources:
            if target in resolution_cache:
                recursive_sources = resolution_cache[target]
            else:
                recursive_sources = resolve_column_recursively(target, None, 10, True)
            if recursive_sources:
                sources.extend(recursive_sources)
        
        # Remove duplicates
        if sources:
            seen = set()
            unique_sources = []
            for src in sources:
                if src not in seen:
                    seen.add(src)
                    unique_sources.append(src)
            final_lineage[target] = unique_sources
        else:
            final_lineage[target] = None
    
    # Also handle any target columns that weren't in temp_view_to_target
    # but might be in the general lineage graph (handles missing columns)
    for mapping in all_mappings:
        parts = mapping.split(" <- ")
        if len(parts) == 2:
            target_col = parts[0].lower()
            source_col = parts[1].lower()
            
            # Skip if it's a temp view mapping (already handled above)
            if "__dbt_tmp" in target_col and "__dbt_tmp" not in source_col:
                # This is source_to_temp_view mapping, skip for now
                continue
            
            # If target is not in final_lineage, try to resolve it (use cache)
            if target_col not in final_lineage:
                # Check if source is a base table
                if "__dbt_tmp" not in source_col.lower() and "." in source_col:
                    # Check if it has further mappings
                    if source_col in reverse_graph:
                        if source_col in resolution_cache:
                            recursive_sources = resolution_cache[source_col]
                        else:
                            recursive_sources = resolve_column_recursively(source_col, None, 10, True)
                        if recursive_sources:
                            final_lineage[target_col] = recursive_sources
                        else:
                            final_lineage[target_col] = [source_col]
                    else:
                        final_lineage[target_col] = [source_col]
                else:
                    # Source is a CTE/temp view, resolve it recursively (use cache)
                    if source_col in resolution_cache:
                        recursive_sources = resolution_cache[source_col]
                    else:
                        recursive_sources = resolve_column_recursively(source_col, None, 10, True)
                    if recursive_sources:
                        final_lineage[target_col] = recursive_sources
                    else:
                        # Even if recursive resolution fails, keep the mapping
                        final_lineage[target_col] = [source_col]
    
    # Additional pass: for any columns in the lineage graph that weren't captured,
    # try to find their sources by column name matching
    # This helps with cases where intermediate CTEs use star expansions
    # Only process columns that weren't already handled (optimization)
    all_target_cols = set(final_lineage.keys())
    for target_col in lineage_graph.keys():
        if target_col not in all_target_cols:
            # This column wasn't captured, try to resolve it (use cache)
            if target_col in resolution_cache:
                recursive_sources = resolution_cache[target_col]
            else:
                recursive_sources = resolve_column_recursively(target_col, None, 10, True)
            if recursive_sources:
                final_lineage[target_col] = recursive_sources
    
    # Propagate sources from temp view columns to final target columns
    # When we have a temp view column (e.g., customerissue__dbt_tmp.id),
    # we should also add the final target column (e.g., customerissue.id) with the same sources
    for col_name, sources in list(final_lineage.items()):
        if "__dbt_tmp" in col_name.lower() and sources:
            # Extract the final target column name by removing __dbt_tmp
            # Format: database.schema.table__dbt_tmp.column -> database.schema.table.column
            parts = col_name.split(".")
            if len(parts) >= 2:
                table_part = parts[-2]  # Get the table part
                column_part = parts[-1]  # Get the column part
                
                # Remove __dbt_tmp from table name (case-insensitive)
                if "__dbt_tmp" in table_part.lower():
                    # Use case-insensitive replacement
                    final_table = re.sub(r"__dbt_tmp", "", table_part, flags=re.IGNORECASE)
                    # Reconstruct the final target column name
                    if len(parts) == 2:
                        # Format: table__dbt_tmp.column
                        final_target_col = f"{final_table}.{column_part}"
                    elif len(parts) == 3:
                        # Format: schema.table__dbt_tmp.column
                        final_target_col = f"{parts[0]}.{final_table}.{column_part}"
                    elif len(parts) == 4:
                        # Format: database.schema.table__dbt_tmp.column
                        final_target_col = f"{parts[0]}.{parts[1]}.{final_table}.{column_part}"
                    else:
                        # Handle longer paths
                        final_target_col = ".".join(parts[:-2]) + f".{final_table}.{column_part}"
                    
                    # Normalize to lowercase for consistency
                    final_target_col = final_target_col.lower()
                    
                    # Add the final target column with the same sources if it doesn't exist
                    if final_target_col not in final_lineage:
                        final_lineage[final_target_col] = sources
                    elif final_lineage[final_target_col] is None:
                        # If it exists but has no sources, use the temp view sources
                        final_lineage[final_target_col] = sources
                    else:
                        # If it already has sources, merge them (avoid duplicates)
                        existing_sources = set(final_lineage[final_target_col])
                        new_sources = set(sources)
                        merged_sources = list(existing_sources | new_sources)
                        final_lineage[final_target_col] = merged_sources
    
    # Filter out entries where source columns contain __dbt_tmp
    # We don't want temp view columns as sources in the final output
    # Also remove temp view columns themselves from the output
    filtered_lineage = {}
    for target_col, sources in final_lineage.items():
        # Skip temp view columns themselves (don't include them in output)
        if "__dbt_tmp" in target_col.lower():
            continue
        
        if sources is None:
            filtered_lineage[target_col] = None
        else:
            # Filter out sources that contain __dbt_tmp
            filtered_sources = [src for src in sources if "__dbt_tmp" not in src.lower()]
            if filtered_sources:
                filtered_lineage[target_col] = filtered_sources
            else:
                # If all sources were filtered out, set to None
                filtered_lineage[target_col] = None
    
    return filtered_lineage

def consolidate_lineage(all_lineages: list, all_edges: list) -> pd.DataFrame:
    """
    Consolidate dict-style and edge-style lineage into a single DataFrame
    with normalization, deduplication, and filtering.
    """
    all_lineages_records = []

    # Process dict-style lineage (from extract_sql_lineage_source_to_target)
    for lineage_dict, query_id, query_type, session_id in all_lineages:
        for target, sources in lineage_dict.items():
            if sources:
                for src in sources:
                    src_db, src_schema, src_table, src_col = sqllineage_lineage.parse_full_column(src)
                    tgt_db, tgt_schema, tgt_table, tgt_col = sqllineage_lineage.parse_full_column(target)
                    all_lineages_records.append({
                        "source_database": src_db,
                        "source_schema": src_schema,
                        "source_table": src_table,
                        "source_column": src_col,
                        "target_database": tgt_db,
                        "target_schema": tgt_schema,
                        "target_table": tgt_table,
                        "target_column": tgt_col,
                        "query_id": query_id,
                        "query_type": query_type,
                        "session_id": session_id,
                        "dbt_model_file_path": None,
                        "dependency_score": 1
                    })
            else:
                tgt_db, tgt_schema, tgt_table, tgt_col = sqllineage_lineage.parse_full_column(target)
                all_lineages_records.append({
                    "source_database": None,
                    "source_schema": None,
                    "source_table": None,
                    "source_column": None,
                    "target_database": tgt_db,
                    "target_schema": tgt_schema,
                    "target_table": tgt_table,
                    "target_column": tgt_col,
                    "query_id": query_id,
                    "query_type": query_type,
                    "session_id": session_id,
                    "dbt_model_file_path": None,
                    "dependency_score": 1
                })

    if all_lineages_records:
        all_lineages_records_df = pd.DataFrame(all_lineages_records)
        all_lineages_records_df.drop_duplicates(
                    subset=[
                        "source_database",
                        "source_schema",
                        "source_table",
                        "source_column",
                        "target_database",
                        "target_schema",
                        "target_table",
                        "target_column"
                    ],
                    inplace=True
                )
    else:
        all_lineages_records_df = pd.DataFrame()
    
    if all_edges:
        all_edges_records_df = pd.DataFrame(all_edges)

        # Deduplication on source→target
        all_edges_records_df.drop_duplicates(
            subset=[
                "source_database", "source_schema", "source_table", "source_column",
                "target_database", "target_schema", "target_table", "target_column"
            ],
            inplace=True
        )

        # Apply filters
        mask = (
            (all_edges_records_df["source_database"].notna() | all_edges_records_df["source_schema"].notna()) &
            (all_edges_records_df["target_schema"].str.lower().fillna("") != "<default>") &
            (~all_edges_records_df["target_table"].str.lower().fillna("").str.contains("__dbt_tmp")) &
            (all_edges_records_df["source_column"].str.strip().fillna("") != "*")
        )
        all_edges_records_df = all_edges_records_df[mask]

        # Replace unwanted values in source_schema and source_table with None
        all_edges_records_df.loc[
            all_edges_records_df["source_schema"].str.lower().fillna("") == "<default>",
            ["source_schema", "source_database"]
        ] = None

        all_edges_records_df.loc[
            all_edges_records_df["source_table"].str.lower().fillna("").str.contains("__dbt_tmp"),
            ["source_table", "source_schema", "source_database", "source_column"]
        ] = None

    else:
        all_edges_records_df = pd.DataFrame()
        
    # Concatenate the two DataFrames
    df = pd.concat([all_lineages_records_df, all_edges_records_df], ignore_index=True)

    return df


def apply_scd_type2(engine, model_class, current_df: pd.DataFrame, historical_df: pd.DataFrame, org_id: uuid.UUID, batch_id: uuid.UUID, connection_id: uuid.UUID):
    """
    SCD Type 2 style update for column-level lineage.
    """

    # Define key columns (unique lineage edge)
    key_cols = [
        "org_id", "connection_id",
        "source_database", "source_schema", "source_table", "source_column",
        "target_database", "target_schema", "target_table", "target_column"
    ]
    
    target_cols = [
        "org_id", "connection_id",
        "target_database", "target_schema", "target_table"
    ]

    historical_df = historical_df.where(pd.notnull(historical_df), np.nan)

    current_df['org_id'] = current_df['org_id'].astype(str)
    current_df['connection_id'] = current_df['connection_id'].astype(str)
    historical_df['org_id'] = historical_df['org_id'].astype(str)
    historical_df['connection_id'] = historical_df['connection_id'].astype(str)

    # First, deduplicate current_df to avoid inserting duplicates within the same batch
    current_df = current_df.drop_duplicates(subset=key_cols, keep='first')
    
    current_targets = current_df[target_cols].drop_duplicates()

    # Get all active historical records (historical_df should already be filtered to is_active=1 from query)
    # But ensure we're working with all historical records to prevent duplicates across batches
    historical_active_all = historical_df.copy()
    
    # Also get historical records matching current targets for deactivation logic
    historical_active = historical_df[
        historical_df.set_index(target_cols).index.isin(current_targets.set_index(target_cols).index)
    ].copy()

    current_df["is_active"] = 1

    # Check against ALL active historical records to prevent duplicates
    # This ensures we don't insert the same lineage edge if it already exists (regardless of batch_id)
    if not historical_active_all.empty and "id" in historical_active_all.columns:
        merged_all = current_df.merge(
            historical_active_all[key_cols + ["id"]],
            on=key_cols,
            how="left",
            indicator=True,
            suffixes=("", "_hist")
        )
        
        # Only insert records that don't exist in ANY active historical record
        to_insert = merged_all[merged_all["_merge"] == "left_only"].drop(columns=["_merge", "id"], errors="ignore")
    else:
        # If no historical data, all current records are new
        to_insert = current_df.copy()
    
    # For deactivation, check against records matching current targets
    if not historical_active.empty and "id" in historical_active.columns:
        merged = current_df.merge(
            historical_active[key_cols + ["id"]],
            on=key_cols,
            how="outer",
            indicator=True,
            suffixes=("", "_hist")
        )
        # to deactivate → only in history (records that existed but are no longer in current)
        to_deactivate = merged[merged["_merge"] == "right_only"]
    else:
        to_deactivate = pd.DataFrame()
   # Case: target matches but source cols are NULL (static derivations)
    # For these, if already exist in history, don't insert duplicate.
    if not to_insert.empty:
        null_sources = to_insert[
            to_insert[["source_database", "source_schema", "source_table", "source_column"]].isnull().all(axis=1)
        ]
        if not null_sources.empty and not historical_active_all.empty:
            # filter out ones already present in ALL historical records (not just matching targets)
            already_in_history = null_sources.merge(
                historical_active_all,
                on=target_cols + ["target_column"],
                how="inner"
            )
            to_insert = pd.concat([
                to_insert.drop(null_sources.index),
                null_sources.loc[~null_sources.index.isin(already_in_history.index)]
            ]).reset_index(drop=True)

    deactivated_count, inserted_count = 0, 0

    # deactivate old lineage edges
    if not to_deactivate.empty:
        ids_to_update = [
        uuid.UUID(str(x)) for x in to_deactivate["id"].dropna().tolist()
        ]

        if ids_to_update:
            with Session(engine) as session:
                session.execute(
                    update(model_class)
                    .where(model_class.id.in_(ids_to_update))
                    .values(is_active=0, updated_at=func.timezone("UTC", func.now()))
                )
                session.commit()
            deactivated_count = len(ids_to_update)

    # insert new lineage edges
    if not to_insert.empty:
        records = to_insert.drop(columns=["id", "_merge"], errors="ignore").to_dict(orient="records")
        lineage_objects = []
        for rec in records:
            lineage_objects.append(
                model_class(
                    id=uuid.uuid4(),
                    org_id=org_id,
                    batch_id=batch_id,
                    connection_id=connection_id,
                    source_database=rec.get("source_database"),
                    source_schema=rec.get("source_schema"),
                    source_table=rec.get("source_table"),
                    source_column=rec.get("source_column"),
                    target_database=rec.get("target_database"),
                    target_schema=rec.get("target_schema"),
                    target_table=rec.get("target_table"),
                    target_column=rec.get("target_column"),
                    query_id=rec.get("query_id"),
                    query_type=rec.get("query_type"),
                    session_id=rec.get("session_id"),
                    dependency_score=rec.get("dependency_score"),
                    dbt_model_file_path=rec.get("dbt_model_file_path"),
                    is_active=1,
                    is_filter_clause=rec.get("is_filter_clause", 0)
                )
            )
        with Session(engine) as session:
            session.bulk_save_objects(lineage_objects)
            session.commit()
        inserted_count = len(records)

    return deactivated_count, inserted_count


def insert_lineage(engine, model_class, df: pd.DataFrame, org_id: uuid.UUID, batch_id: uuid.UUID, connection_id: uuid.UUID):
 
    if df.empty:
        return 0

    with Session(engine) as session:
        objects_to_insert = []
        for _, row in df.iterrows():
            lineage_obj = model_class(
                id=uuid.uuid4(),
                org_id=org_id,
                batch_id=batch_id,
                connection_id=connection_id,
                source_database=row.get("source_database"),
                source_schema=row.get("source_schema"),
                source_table=row.get("source_table"),
                source_column=row.get("source_column"),
                target_database=row.get("target_database"),
                target_schema=row.get("target_schema"),
                target_table=row.get("target_table"),
                target_column=row.get("target_column"),
                query_id=row.get("query_id"),  # JSONB/list is supported
                query_type=row.get("query_type"),
                session_id=row.get("session_id"),
                dependency_score=row.get("dependency_score"),
                dbt_model_file_path=row.get("dbt_model_file_path"),
                is_filter_clause=row.get("is_filter_clause", 0)
            )
            objects_to_insert.append(lineage_obj)

        if objects_to_insert:
            session.bulk_save_objects(objects_to_insert)
            session.commit()
            return len(objects_to_insert)
    return 0


def lineage_builder(org_id, conn_id, batch_id):
    try:
        logger.info("Starting lineage_builder with org_id=%s, conn_id=%s, batch_id=%s", org_id, conn_id, batch_id)
        pg_engine = sqllineage_lineage.get_pg_engine()
        logger.info("PostgreSQL engine created successfully")
        
        fetch_query_history_df, information_schema_columns_df, historical_column_level_lineage_df, historical_filter_clause_column_lineage_df = sqllineage_lineage.fetch_query_access_history_and_information_schema_columns(pg_engine, org_id, conn_id, batch_id)
        logger.info("fetch_query_history_df, information_schema_columns_df, historical_column_level_lineage_df and  historical_filter_clause_column_lineage_df retrieved")
        
        last_processed_at = fetch_query_history_df["created_at"].max()
        logger.info("Last processed timestamp: %s", last_processed_at)
        
        final_df = sqllineage_lineage.combine_queries_by_session(fetch_query_history_df)
        logger.info("Queries combined by session, processing %d sessions", len(final_df))
        
        final_df['base_objects_accessed'] = final_df['base_objects_accessed'].apply(ast.literal_eval)
        logger.info("Base objects accessed parsed successfully")
        for query_id, query_text, query_type, session_id, base_objects_accessed, database_name, schema_name in final_df[['query_id', 'query_text', 'query_type', 'session_id', 'base_objects_accessed', 'database_name', 'schema_name']].values:
            try:
                cleaned_query = sqllineage_lineage.detect_and_replace_named_parameters(query_text, static_value="null")
                try:
                    final_lineage = extract_sql_lineage_source_to_target(cleaned_query)
                except Exception as e:
                    logging.error(f"[{query_id}] extract_sql_lineage failed: {e}")
                    final_lineage = {}

#             # Check if lineage is useless (all None values or empty)
                if not final_lineage or all(v is None for v in final_lineage.values()):
                    logging.info(f"[{query_id}] Falling back to parse_lineage_text()...")
                    cleaned_query = cleaned_query.upper()
                    lineage_process = subprocess.run(
                        ["sqllineage", "-e", cleaned_query, "-l", "column", "--dialect=snowflake"],
                        capture_output=True,
                        text=True,
                    )

                    lineage_output = lineage_process.stdout
                    if lineage_process.returncode != 0:
                        logging.warning(f"[{query_id}] sqllineage warning: {lineage_process.stderr}")

                    parsed_edges = sqllineage_lineage.parse_lineage_text(
                        query_id,
                        cleaned_query,
                        query_type,
                        session_id,
                        base_objects_accessed,
                        database_name,
                        schema_name,
                        information_schema_columns_df,
                        lineage_output,
                    )
                    if parsed_edges:
                        all_edges.extend(parsed_edges)
                    else:
                        logging.warning(f"[{query_id}] parse_lineage_text returned no edges; skipping")

                # Only collect valid lineage
                if final_lineage and not all(v is None for v in final_lineage.values()):
                    all_lineages.append((final_lineage, query_id, query_type, session_id))
                    # logging.info(f"[{query_id}] Lineage collected.")  # Suppressed verbose logging

            except Exception as loop_err:
                logging.error(f"[{query_id}] Unexpected error while processing query: {loop_err}", exc_info=True)
                # continue to next query without breaking the loop
                continue


        logger.info("Processing lineage consolidation...")
        final_df["query_id"] = final_df["query_id"].apply(
                lambda x: str(x) if isinstance(x, list) else x
            )
        logger.info("Query IDs converted to strings")
       
        # final_df.to_csv("C:/Users/User/Documents/12-09-2025_lineage_final/final_df.csv", index=False)
        # logger.info("final_df saved as csv")

        consolidated_df = consolidate_lineage(all_lineages, all_edges)
        logger.info("Lineage consolidated, %d records in consolidated_df", len(consolidated_df))
        
        # Add required columns for SCD Type 2 processing
        if not consolidated_df.empty:
            consolidated_df["org_id"] = org_id
            consolidated_df["connection_id"] = conn_id
            consolidated_df["batch_id"] = batch_id
            logger.info("Added org_id, connection_id, and batch_id columns to consolidated_df")
        
        consolidated_df["query_id"] = consolidated_df["query_id"].apply(
                lambda x: str(x) if isinstance(x, list) else x
            )
        # consolidated_df.to_csv("C:/Users/User/Documents/12-09-2025_lineage_final/consolidated_df.csv", index=False)
        # logger.info("Consolidated_df saved as csv")
        logger.info("Consolidated query IDs converted to strings")


        filter_clause_df = pd.merge(consolidated_df, final_df, on="query_id", how="inner")
        logger.info("Filter clause DataFrame merged, %d records", len(filter_clause_df))
        
        rows = get_dependent_columns(filter_clause_df)
        logger.info("Dependent columns extracted, %d rows", len(rows) if rows else 0)

        if rows:
            final_filter_clause_df = pd.DataFrame(rows)
            final_filter_clause_df.drop_duplicates(
            subset=[
                "source_database", "source_schema", "source_table", "source_column",
                "target_database", "target_schema", "target_table", "target_column"
            ],
            inplace=True
            )
            mask = ~(
            final_filter_clause_df["source_database"].fillna("").eq("") &
            final_filter_clause_df["source_schema"].fillna("").eq("")
            )

            final_filter_clause_df = final_filter_clause_df[mask]
            
            # Add required columns for SCD Type 2 processing
            if not final_filter_clause_df.empty:
                final_filter_clause_df["org_id"] = org_id
                final_filter_clause_df["connection_id"] = conn_id
                final_filter_clause_df["batch_id"] = batch_id
                logger.info("Added org_id, connection_id, and batch_id columns to final_filter_clause_df")
        else:
            final_filter_clause_df = pd.DataFrame()

        # final_filter_clause_df.to_csv("C:/Users/User/Documents/12-09-2025_lineage_final/final_filter_clause_df.csv", index=False)
        # logger.info("final_filter_clause_df saved as csv")


        if not consolidated_df.empty:
            try:
                logger.info("Starting lineage processing for column lineage...")
                if not historical_column_level_lineage_df.empty:
                    logger.info("Processing with SCD Type 2 for column lineage...")
                    deactivated_column_level_lineage, inserted_column_level_lineage, = apply_scd_type2(pg_engine, ColumnLevelLineage, consolidated_df, historical_column_level_lineage_df, org_id, batch_id, conn_id)
                    logger.info(f"{deactivated_column_level_lineage} records deactivated in ColumnLevelLineage table, "f"{inserted_column_level_lineage} new records inserted in ColumnLevelLineage table.")

                else:
                    logger.info("Processing with direct insert for column lineage...")
                    inserted_count = insert_lineage(
                        pg_engine, ColumnLevelLineage, consolidated_df, org_id=org_id, batch_id=batch_id, connection_id=conn_id
                    )

                    logger.info(f"Inserted {inserted_count} lineage records into column_level_lineage")


                if not final_filter_clause_df.empty:
                    try:
                        logger.info("Starting lineage processing for filter clause column lineage...")
                        if not historical_filter_clause_column_lineage_df.empty:
                            logger.info("Processing with SCD Type 2 for filter clause column lineage...")
                            deactivated_filter_clause_column_lineage, inserted_filter_clause_column_lineage, = apply_scd_type2(pg_engine, ColumnLevelLineage, final_filter_clause_df, historical_filter_clause_column_lineage_df, org_id, batch_id, conn_id)
                            logger.info(f"{deactivated_filter_clause_column_lineage} records deactivated in ColumnLevelLineage table, "f"{inserted_filter_clause_column_lineage} new records inserted in ColumnLevelLineage table.")
                        else:
                            logger.info("Processing with direct insert for filter clause column lineage...")
                            inserted_count_filter_clause = insert_lineage(
                                pg_engine, ColumnLevelLineage, final_filter_clause_df, org_id=org_id, batch_id=batch_id, connection_id=conn_id
                            )

                            logger.info(f"Inserted {inserted_count_filter_clause} lineage records into final_filter_clause_df")
                    except Exception as lineage_error:
                        logger.error("Error during lineage processing: %s", lineage_error)
                        import traceback
                        logger.error("Lineage processing traceback: %s", traceback.format_exc())
                        raise

            

                logger.info("Creating watermark...")
                with Session(pg_engine) as session:
                    watermark = LineageLoadWatermark(
                        org_id=org_id,
                        connection_id=conn_id,
                        batch_id=batch_id,
                        last_processed_at=last_processed_at
                    )
                    session.add(watermark)
                    session.commit()
                    
                    logger.info(f"Updated watermark for batch {batch_id}")
                logger.info("Lineage processing completed successfully")
            except Exception as lineage_error:
                logger.error("Error during lineage processing: %s", lineage_error)
                import traceback
                logger.error("Lineage processing traceback: %s", traceback.format_exc())
                raise
        else:
            logger.info(f"No lineage to process")

    except Exception as e:
        import traceback
        logger.critical("Fatal error in main execution: %s", e)
        logger.critical("Full traceback: %s", traceback.format_exc())
        raise

