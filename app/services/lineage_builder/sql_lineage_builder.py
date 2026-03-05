from typing import Any, Dict, Optional, List, Tuple, Union
from collections import defaultdict

from .schema_resolver import SchemaResolver
from .sqlglot_lineage import SqlParsingResult, sqlglot_lineage
from .sqlglot_utils import get_dialect
import sqlglot


def _extract_merge_mapping(merge: sqlglot.exp.Merge) -> Tuple[str, Optional[str], Dict[str, str]]:
    """Extract MERGE statement mapping information"""
    target_urn = _table_to_urn(merge.this)
    using_expr = merge.args.get("using")
    source_urn: Optional[str] = None
    
    if isinstance(using_expr, sqlglot.exp.Table):
        source_urn = _table_to_urn(using_expr)
    elif isinstance(using_expr, sqlglot.exp.Subquery) and isinstance(using_expr.this, sqlglot.exp.Table):
        source_urn = _table_to_urn(using_expr.this)

    # Build column mapping from MERGE assignments
    colmap: Dict[str, str] = {}
    
    on_sql = None
    try:
        on_expr = merge.args.get("on")
        on_sql = on_expr.sql() if on_expr else None
    except Exception:
        on_sql = None

    for eq in merge.find_all(sqlglot.exp.EQ):
        try:
            # Skip ON clause equalities by string containment heuristic
            if on_sql is not None and eq.sql() in on_sql:
                continue
        except Exception:
            pass

        left = eq.left
        right = eq.right
        tgt_col = None
        
        if isinstance(left, sqlglot.exp.Identifier):
            tgt_col = left.name
        elif isinstance(left, sqlglot.exp.Column) and isinstance(left.this, sqlglot.exp.Identifier):
            tgt_col = left.this.name
            
        if not tgt_col:
            continue

        if isinstance(right, sqlglot.exp.Column) and isinstance(right.this, sqlglot.exp.Identifier):
            src_col = right.this.name
            colmap[tgt_col] = src_col

    whens = merge.args.get("whens")
    if whens:
        for when in getattr(whens, "expressions", []) or []:
            if when.args.get("matched", True):
                # WHEN MATCHED handled via SET assignments above
                continue
            insert_expr = when.args.get("then")
            if not isinstance(insert_expr, sqlglot.exp.Insert):
                continue
            target_columns = getattr(insert_expr.this, "expressions", []) or []
            value_expressions = getattr(insert_expr.expression, "expressions", []) or []
            for target_expr, value_expr in zip(target_columns, value_expressions):
                tgt_col: Optional[str] = None
                if isinstance(target_expr, sqlglot.exp.Identifier):
                    tgt_col = target_expr.name
                elif isinstance(target_expr, sqlglot.exp.Column) and isinstance(target_expr.this, sqlglot.exp.Identifier):
                    tgt_col = target_expr.this.name
                if not tgt_col or tgt_col in colmap:
                    continue

                src_col: Optional[str] = None
                if isinstance(value_expr, sqlglot.exp.Identifier):
                    src_col = value_expr.name
                elif isinstance(value_expr, sqlglot.exp.Column) and isinstance(value_expr.this, sqlglot.exp.Identifier):
                    src_col = value_expr.this.name

                if src_col:
                    colmap[tgt_col] = src_col

    return target_urn, source_urn, colmap


def _table_to_urn(table_expr) -> str:
    """Convert sqlglot table expression to URN format"""
    if isinstance(table_expr.this, sqlglot.exp.Dot):
        # Handle multi-part table names
        parts = []
        exp = table_expr.this
        while isinstance(exp, sqlglot.exp.Dot):
            parts.append(exp.this.name)
            exp = exp.expression
        parts.append(exp.name)
        table_name = ".".join(parts)
    else:
        table_name = table_expr.this.name
    
    # Build URN format: platform://ENV/db.schema.table
    catalog = None
    db = None
    
    if table_expr.catalog:
        if hasattr(table_expr.catalog, 'name'):
            catalog = table_expr.catalog.name
        else:
            catalog = str(table_expr.catalog)
    
    if table_expr.db:
        if hasattr(table_expr.db, 'name'):
            db = table_expr.db.name
        else:
            db = str(table_expr.db)
    
    parts = []
    if catalog:
        parts.append(catalog)
    if db:
        parts.append(db)
    parts.append(table_name)
    
    return ".".join(parts)


def _aggregate_merge_lineage(
    source_lineage: List[Dict[str, Any]],
    target_urn: str,
    colmap: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Aggregate lineage from source to target using MERGE column mapping"""
    by_down_col = {cli["downstream"]["column"]: cli for cli in source_lineage}
    out: List[Dict[str, Any]] = []
    
    for tgt_col, src_col in colmap.items():
        if src_col in by_down_col:
            src_cli = by_down_col[src_col]
            out.append({
                "downstream": {
                    "table": target_urn,
                    "column": tgt_col,
                    "native_column_type": src_cli["downstream"]["native_column_type"],
                },
                "upstreams": src_cli["upstreams"],
                "logic": src_cli["logic"],
            })
    
    return out


def build_lineage(
    sql: str,
    dialect: str,
    *,
    default_db: Optional[str] = None,
    default_schema: Optional[str] = None,
    schema_info_by_table: Optional[Dict[str, Dict[str, str]]] = None,
    enhanced_mode: bool = True,
) -> Dict[str, Any]:
    """
    Parse SQL and return table-level and column-level lineage.

    - dialect: a sqlglot dialect string (e.g., "snowflake", "postgres")
    - schema_info_by_table: optional mapping of fully qualified table name -> { column: type }
      If provided, improves column-level lineage for star expansions and type hints.
    - enhanced_mode: if True, uses enhanced multi-statement processing for complete lineage chains
    """
    resolver = SchemaResolver(platform=dialect)

    # Preload schema info if provided. Use our URN convention used by SchemaResolver
    if schema_info_by_table:
        for fqtn, schema in schema_info_by_table.items():
            urn = f"{dialect}://PROD/{fqtn}"
            resolver.add_raw_schema_info(urn, schema)

    # Support multi-statement SQL (e.g., CREATE VIEW + MERGE). Parse all statements and merge lineage.
    dialect_obj = get_dialect(dialect)
    try:
        statements = sqlglot.parse(sql, dialect=dialect_obj)
    except Exception:
        # Fallback to single-statement parsing path
        statements = [sql]

    if enhanced_mode and len(statements) > 1:
        return _build_lineage_enhanced(statements, dialect_obj, resolver, default_db, default_schema, dialect)
    else:
        return _build_lineage_legacy(statements, dialect_obj, resolver, default_db, default_schema, dialect)


def _build_lineage_enhanced(
    statements: List[sqlglot.exp.Expression],
    dialect_obj: sqlglot.Dialect,
    resolver: SchemaResolver,
    default_db: Optional[str],
    default_schema: Optional[str],
    dialect_name: str,
) -> Dict[str, Any]:
    """Enhanced multi-statement processing with complete lineage chains"""
    
    # Keep lineage for outputs of earlier statements (e.g., temp view)
    output_lineage_by_table: Dict[str, List[Dict[str, Any]]] = {}
    # Also accumulate a combined lineage across statements
    accumulated_lineage: List[Dict[str, Any]] = []
    accumulated_inputs: List[str] = []
    accumulated_outputs: List[str] = []
    last_result: Optional[SqlParsingResult] = None

    for stmt in statements:
        stmt_sql = stmt.sql(dialect=dialect_obj)
        result: SqlParsingResult = sqlglot_lineage(
            stmt_sql,
            schema_resolver=resolver,
            default_db=default_db,
            default_schema=default_schema,
            override_dialect=dialect_name,
        )
        last_result = result

        # Always accumulate input tables for this statement
        if result and result.in_tables:
            accumulated_inputs.extend(result.in_tables)

        # Track lineage for produced tables/views
        if result and result.out_tables and result.column_lineage:
            for t in result.out_tables:
                # cache under multiple equivalent keys to improve matching robustness
                keys = {
                    t,
                    t.lower(),
                    t.upper(),
                    # Also cache without URN prefix for better matching
                    t.split("://")[-1] if "://" in t else t,
                    t.split("://")[-1].lower() if "://" in t else t.lower(),
                    t.split("://")[-1].upper() if "://" in t else t.upper(),
                }
                
                # Special handling for dbt temp tables - also cache with __dbt_tmp suffix
                if "__dbt_tmp" not in t.lower():
                    temp_name = t.lower() + "__dbt_tmp"
                    keys.update({
                        temp_name,
                        temp_name.upper(),
                        temp_name.split("://")[-1] if "://" in temp_name else temp_name,
                        temp_name.split("://")[-1].upper() if "://" in temp_name else temp_name.upper(),
                    })
                
                lineage_dict = _result_to_dict(result)["column_lineage"]
                for k in keys:
                    output_lineage_by_table[k] = lineage_dict
                # accumulate this statement's lineage as well (e.g., source -> temp view)
                accumulated_lineage.extend(lineage_dict)
                accumulated_outputs.extend(result.out_tables)
                
                # Debug logging
                # import logging
                # logging.debug(f"Cached lineage for table {t} under keys: {keys}")
                # logging.debug(f"Cached lineage for table {t}: {len(lineage_dict)} column mappings")

        # Enhanced MERGE processing
        if isinstance(stmt, sqlglot.exp.Merge):
            tgt_urn, src_urn, colmap = _extract_merge_mapping(stmt)
            if src_urn:
                # Try to find cached lineage for the source table
                candidate_keys = [
                    src_urn,
                    src_urn.lower(),
                    src_urn.upper(),
                ]
                hit_key = next((k for k in candidate_keys if k in output_lineage_by_table), None)
                
                if hit_key:
                    agg = _aggregate_merge_lineage(
                        output_lineage_by_table[hit_key],
                        tgt_urn,
                        colmap,
                    )
                    # Return both temp view <- source AND target <- temp view mappings
                    temp_view_mappings = _build_mappings_from_lineage(output_lineage_by_table[hit_key])
                    target_mappings = _build_mappings_from_lineage(agg)
                    all_mappings = temp_view_mappings + target_mappings
                    return {"source_to_target": all_mappings}
                else:
                    # Fallback: build direct lineage from column mapping
                    direct_lineage = []
                    for tgt_col, src_col in colmap.items():
                        direct_lineage.append({
                            "downstream": {
                                "table": tgt_urn,
                                "column": tgt_col,
                                "native_column_type": None,
                            },
                            "upstreams": [
                                {"table": src_urn, "column": src_col}
                            ],
                            "logic": None,
                        })
                    # Return both temp view <- source AND target <- temp view mappings
                    temp_view_mappings = _build_mappings_from_lineage(accumulated_lineage)
                    target_mappings = _build_mappings_from_lineage(direct_lineage)
                    all_mappings = temp_view_mappings + target_mappings
                    return {"source_to_target": all_mappings}

        # Enhanced INSERT processing
        if isinstance(stmt, sqlglot.exp.Insert):
            # If INSERT selects from a temp view we produced earlier, reuse its lineage
            try:
                target_tbl = stmt.this if isinstance(stmt.this, sqlglot.exp.Table) else None
                source_expr = stmt.expression
                if target_tbl and isinstance(source_expr, sqlglot.exp.Query):
                    src_tables = [t for t in source_expr.find_all(sqlglot.exp.Table)]
                    # Debug logging
                    # import logging
                    # logging.debug(f"INSERT statement found {len(src_tables)} source tables: {[str(t) for t in src_tables]}")
                    # logging.debug(f"Available cached lineage keys: {list(output_lineage_by_table.keys())}")
                    
                    # Try to find any source table that matches a cached produced view
                    hit_key = None
                    for t in src_tables:
                        src_urn = _table_to_urn(t)
                        # Create multiple candidate keys for better matching
                        candidate_keys = [
                            src_urn, 
                            src_urn.lower(), 
                            src_urn.upper(),
                            # Also try without the URN prefix
                            src_urn.split("://")[-1] if "://" in src_urn else src_urn,
                            src_urn.split("://")[-1].lower() if "://" in src_urn else src_urn.lower(),
                            src_urn.split("://")[-1].upper() if "://" in src_urn else src_urn.upper(),
                        ]
                        
                        # Special handling for dbt temp tables - try to match with base table name
                        if "__dbt_tmp" in src_urn.lower():
                            base_name = src_urn.lower().replace("__dbt_tmp", "")
                            candidate_keys.extend([
                                base_name,
                                base_name.upper(),
                                base_name.split("://")[-1] if "://" in base_name else base_name,
                                base_name.split("://")[-1].upper() if "://" in base_name else base_name.upper(),
                            ])
                        
                        hit_key = next((k for k in candidate_keys if k in output_lineage_by_table), None)
                        if hit_key:
                            break
                    
                    if hit_key:
                        tgt_urn = _table_to_urn(target_tbl)
                        # Project source (from cached temp view lineage) directly onto target
                        remapped = []
                        for cli in output_lineage_by_table[hit_key]:
                            remapped.append({
                                "downstream": {
                                    "table": tgt_urn,
                                    "column": cli["downstream"]["column"],
                                    "native_column_type": cli["downstream"]["native_column_type"],
                                },
                                "upstreams": cli["upstreams"],
                                "logic": cli["logic"],
                            })
                        # Return both temp view <- source AND target <- temp view mappings
                        temp_view_mappings = _build_mappings_from_lineage(output_lineage_by_table[hit_key])
                        target_mappings = _build_mappings_from_lineage(remapped)
                        all_mappings = temp_view_mappings + target_mappings
                        return {"source_to_target": all_mappings}
            except Exception:
                pass

    # If no enhanced processing was applied, return accumulated lineage
    mappings = _build_mappings_from_lineage(accumulated_lineage)
    return {"source_to_target": mappings}


def _build_lineage_legacy(
    statements: List[sqlglot.exp.Expression],
    dialect_obj: sqlglot.Dialect,
    resolver: SchemaResolver,
    default_db: Optional[str],
    default_schema: Optional[str],
    dialect_name: str,
) -> Dict[str, Any]:
    """Legacy single-statement processing (preserves original behavior)"""
    
    combined_column_lineage: List[Dict[str, Any]] = []

    def _result_to_dict(result: SqlParsingResult) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "in_tables": result.in_tables,
            "out_tables": result.out_tables,
            "query_type": result.query_type.value,
            "column_lineage": None,
            "joins": None,
        }
        if result.column_lineage is not None:
            out["column_lineage"] = [
                {
                    "downstream": {
                        "table": cli.downstream.table,
                        "column": cli.downstream.column,
                        "native_column_type": cli.downstream.native_column_type,
                    },
                    "upstreams": [
                        {"table": u.table, "column": u.column} for u in cli.upstreams
                    ],
                    "logic": (
                        {
                            "is_direct_copy": cli.logic.is_direct_copy,
                            "column_logic": cli.logic.column_logic,
                        }
                        if cli.logic
                        else None
                    ),
                }
                for cli in result.column_lineage
            ]
        if result.joins is not None:
            out["joins"] = [
                {
                    "join_type": j.join_type,
                    "left_tables": j.left_tables,
                    "right_tables": j.right_tables,
                    "on_clause": j.on_clause,
                    "columns_involved": [
                        {"table": c.table, "column": c.column}
                        for c in j.columns_involved
                    ],
                }
                for j in result.joins
            ]
        return out

    for stmt in statements:
        stmt_sql = stmt if isinstance(stmt, str) else stmt.sql(dialect=dialect_obj)
        result: SqlParsingResult = sqlglot_lineage(
            stmt_sql,
            schema_resolver=resolver,
            default_db=default_db,
            default_schema=default_schema,
            override_dialect=dialect_name,
        )
        out = _result_to_dict(result)
        if out.get("column_lineage"):
            combined_column_lineage.extend(out["column_lineage"])  # type: ignore[arg-type]

    mappings = _build_mappings_from_lineage(combined_column_lineage)
    return {"source_to_target": mappings}


def _result_to_dict(result: SqlParsingResult) -> Dict[str, Any]:
    """Convert SqlParsingResult to dictionary format"""
    out: Dict[str, Any] = {
        "in_tables": result.in_tables,
        "out_tables": result.out_tables,
        "query_type": result.query_type.value,
        "column_lineage": None,
        "joins": None,
    }
    if result.column_lineage is not None:
        out["column_lineage"] = [
            {
                "downstream": {
                    "table": cli.downstream.table,
                    "column": cli.downstream.column,
                    "native_column_type": cli.downstream.native_column_type,
                },
                "upstreams": [
                    {"table": u.table, "column": u.column} for u in cli.upstreams
                ],
                "logic": (
                    {
                        "is_direct_copy": cli.logic.is_direct_copy,
                        "column_logic": cli.logic.column_logic,
                    }
                    if cli.logic
                    else None
                ),
            }
            for cli in result.column_lineage
        ]
    if result.joins is not None:
        out["joins"] = [
            {
                "join_type": j.join_type,
                "left_tables": j.left_tables,
                "right_tables": j.right_tables,
                "on_clause": j.on_clause,
                "columns_involved": [
                    {"table": c.table, "column": c.column}
                    for c in j.columns_involved
                ],
            }
            for j in result.joins
        ]
    return out


def _build_mappings_from_lineage(column_lineage: List[Dict[str, Any]]) -> List[str]:
    """Build source_to_target mappings from column lineage"""
    
    def _split_urn(urn: Optional[str]) -> Tuple[str, str, str]:
        if not urn:
            return "", "", ""
        # Format: platform://ENV/db.schema.table
        try:
            _, rest = urn.split("://", 1)
            _, path = rest.split("/", 1)
        except ValueError:
            path = urn
        parts = path.split(".")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return "", parts[0], parts[1]
        if len(parts) == 1:
            return "", "", parts[0]
        # More than 3 segments, take last 3 as db.schema.table
        return parts[-3], parts[-2], parts[-1]

    mappings: List[str] = []
    for cli in column_lineage:
        tgt_urn = cli["downstream"]["table"]
        tgt_col = cli["downstream"]["column"]
        tgt_db, tgt_schema, tgt_table = _split_urn(tgt_urn)
        if not tgt_table or not tgt_col:
            continue
        for up in cli["upstreams"]:
            src_urn = up["table"]
            src_col = up["column"]
            src_db, src_schema, src_table = _split_urn(src_urn)
            tgt = ".".join(filter(None, [tgt_db, tgt_schema, tgt_table, tgt_col]))
            src = ".".join(filter(None, [src_db, src_schema, src_table, src_col]))
            mappings.append(f"{tgt} <- {src}")

    return mappings

