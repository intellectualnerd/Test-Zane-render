import pandas as pd
import csv
import os
import re
from sqlglot import parse_one, exp
import logging
import ast
import sys
import sqlglot
from sqlglot.expressions import (
    Column, Table, Subquery, Union, With, Select, Window, Join, Identifier
)

# Logging Setup
logger = logging.getLogger("lineage.filter_clause_columns")
logger.setLevel(logging.INFO)

# Suppress third-party library logging
logging.getLogger("sqllineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot").setLevel(logging.WARNING)
logging.getLogger("sqlglot.lineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot.optimizer").setLevel(logging.WARNING)

def safe_name(x):
  try:
    if x is None:
        return None
    return getattr(x, "name", str(x))
  except Exception as e:
        logger.error("safe_name error: %s", e, exc_info=True)
        return None

def is_base_fqn(q: str) -> bool:
  try:
    return q.count(".") >= 3
  except Exception as e:
        logger.error("is_base_fqn error: %s", e, exc_info=True)
        return False

def split_fqn(fqn: str):
    try:
        if not fqn:
            return None, None, None, None
        parts = fqn.split(".")
        if len(parts) >= 4:
            return parts[-4], parts[-3], parts[-2], parts[-1]
        if len(parts) == 3:
            return None, parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return None, None, parts[0], parts[1]
        return None, None, None, parts[0]
    except Exception as e:
        logger.error("split_fqn error: %s", e, exc_info=True)
        return None, None, None, None

def join_table_fqn(catalog, db, table):
    try:
        return ".".join([p for p in [catalog, db, table] if p])
    except Exception as e:
        logger.error("join_table_fqn error: %s", e, exc_info=True)
        return None

def merge_filter_maps(acc, add):
    try:
        if not add:
            return acc
        for k, vals in add.items():
            acc.setdefault(k, set()).update(vals)
        return acc
    except Exception as e:
        logger.error("merge_filter_maps error: %s", e, exc_info=True)
        return acc

def finalize_filters(filters_set_dict):
    try:
        keys = [
            "where",
            "group_by",
            "having",
            "join_on",
            "order_by",
            "qualify",
            "window_partition_by",
            "window_order_by",
        ]
        return {k: sorted(filters_set_dict.get(k, set())) for k in keys}
    except Exception as e:
        logger.error("finalize_filters error: %s", e, exc_info=True)
        return {}

def iter_columns(value):
    try:
        if not value:
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                yield from iter_columns(v)
            return
        if isinstance(value, Column):
            yield value
            return
        yield from value.find_all(Column)
    except Exception as e:
        logger.error("iter_columns error: %s", e, exc_info=True)
        return

def iter_identifiers(value):
    try:
        if not value:
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                yield from iter_identifiers(v)
            return
        if isinstance(value, Identifier):
            yield value
            return
        yield from value.find_all(Identifier)
    except Exception as e:
        logger.error("iter_identifiers error: %s", e, exc_info=True)
        return

def build_select_scope(sel: Select):
    try:
        alias_to_table = {}
        subquery_aliases = {}

        for table in sel.find_all(Table):
            parent_select = table.parent
            while parent_select and not isinstance(parent_select, Select):
                parent_select = parent_select.parent
            if parent_select is not sel:
                continue

            full_name = ".".join(
                filter(None, [safe_name(table.catalog), safe_name(table.db), safe_name(table.this)])
            )
            key = safe_name(table.alias) or safe_name(table.this)
            if key:
                alias_to_table[key] = full_name

        for sq in sel.find_all(Subquery):
            parent_select = sq.parent
            while parent_select and not isinstance(parent_select, Select):
                parent_select = parent_select.parent
            if parent_select is not sel:
                continue

            if sq.alias:
                subquery_aliases[safe_name(sq.alias)] = sq.this

        # Look for CTEs in the current SELECT and all parent SELECTs
        # CTEs are typically defined at the root level but are available to all SELECTs
        current = sel
        visited = set()
        while current:
            if id(current) in visited:
                break
            visited.add(id(current))
            
            with_expr = current.args.get("with")
            if isinstance(with_expr, With):
                for cte in with_expr.find_all(Subquery):
                    if cte.alias:
                        cte_alias = safe_name(cte.alias)
                        if cte_alias and cte_alias not in subquery_aliases:
                            subquery_aliases[cte_alias] = cte.this
            
            # Walk up to parent SELECT
            parent = current.parent
            while parent and not isinstance(parent, Select):
                parent = parent.parent
            current = parent

        return alias_to_table, subquery_aliases
    except Exception as e:
        logger.error("build_select_scope error: %s", e, exc_info=True)
        return {}, {}

def qualify_col_in_scope(col: Column, alias_to_table: dict) -> str:
    try:
        col_name = safe_name(col.this)
        tbl = safe_name(col.table)
        if tbl and tbl in alias_to_table:
            return f"{alias_to_table[tbl]}.{col_name}"
        elif tbl:
            return f"{tbl}.{col_name}"
        else:
            return col_name
    except Exception as e:
        logger.error("qualify_col_in_scope error: %s", e, exc_info=True)
        return None

def extract_clause_lineage_for_select(sel: Select, alias_to_table: dict):
    try:
        out = {
            "where": set(),
            "group_by": set(),
            "having": set(),
            "join_on": set(),
            "order_by": set(),
            "qualify": set(),
            "window_partition_by": set(),
            "window_order_by": set(),
        }

        for c in iter_columns(sel.args.get("where")):
            out["where"].add(qualify_col_in_scope(c, alias_to_table))
        for c in iter_columns(sel.args.get("group")):
            out["group_by"].add(qualify_col_in_scope(c, alias_to_table))
        for c in iter_columns(sel.args.get("having")):
            out["having"].add(qualify_col_in_scope(c, alias_to_table))
        for c in iter_columns(sel.args.get("qualify")):
            out["qualify"].add(qualify_col_in_scope(c, alias_to_table))
        for c in iter_columns(sel.args.get("order")):
            out["order_by"].add(qualify_col_in_scope(c, alias_to_table))

        for j in sel.args.get("joins", []) or []:
            if isinstance(j, Join):
                # Extract filters from the JOIN's ON condition
                for c in iter_columns(j.args.get("on")):
                    out["join_on"].add(qualify_col_in_scope(c, alias_to_table))
                for ident in iter_identifiers(j.args.get("using")):
                    out["join_on"].add(safe_name(ident))
                
                # Extract filters from subqueries inside JOIN clauses
                # Example: INNER JOIN (SELECT ... FROM ... INNER JOIN ... ON ...) AS b
                join_expression = j.this if hasattr(j, 'this') else j.args.get("this")
                if isinstance(join_expression, Subquery):
                    # This is a subquery in a JOIN (e.g., (SELECT ...) AS alias)
                    subq_select = join_expression.this
                    
                    # Build scope for the subquery's SELECT
                    # Handle both direct Select and Union/other query types
                    subquery_selects = []
                    if isinstance(subq_select, Select):
                        subquery_selects = [subq_select]
                    else:
                        # Try to find Select statements within (for Union, etc.)
                        subquery_selects = list(subq_select.find_all(Select))
                    
                    for subq_sel in subquery_selects:
                        # Get the subquery's internal scope
                        subq_alias_to_table, _ = build_select_scope(subq_sel)
                        
                        # Build comprehensive parent scope (includes outer query scopes)
                        def build_parent_scope_for_join_subq(subq_node, start_alias_to_table, parent_select):
                            """Build scope including all outer query scopes for correlated column resolution"""
                            scope = start_alias_to_table.copy()
                            
                            # Start with the parent SELECT's scope
                            if parent_select:
                                parent_scope, _ = build_select_scope(parent_select)
                                scope.update(parent_scope)
                            
                            # Walk up from the subquery node to find all outer SELECT scopes
                            current = subq_node.parent
                            visited_parents = set()
                            
                            while current:
                                if id(current) in visited_parents:
                                    break
                                visited_parents.add(id(current))
                                
                                if isinstance(current, Select):
                                    outer_alias_to_table, _ = build_select_scope(current)
                                    scope.update(outer_alias_to_table)
                                elif isinstance(current, Subquery):
                                    subq_parent = current.parent
                                    while subq_parent and not isinstance(subq_parent, Select):
                                        subq_parent = subq_parent.parent
                                    if isinstance(subq_parent, Select):
                                        outer_alias_to_table, _ = build_select_scope(subq_parent)
                                        scope.update(outer_alias_to_table)
                                
                                current = current.parent
                            
                            return scope
                        
                        comprehensive_parent_scope = build_parent_scope_for_join_subq(
                            join_expression, alias_to_table, sel
                        )
                        
                        # Merge scopes: subquery's internal scope + parent scopes
                        combined_alias_to_table = {**subq_alias_to_table, **comprehensive_parent_scope}
                        
                        # Extract filters from WHERE clause inside the JOIN subquery
                        where_clause = subq_sel.args.get("where")
                        if where_clause:
                            for c in iter_columns(where_clause):
                                qualified_col = qualify_col_in_scope(c, combined_alias_to_table)
                                if qualified_col:
                                    out["where"].add(qualified_col)
                            # Recursively extract from nested subqueries in WHERE clause
                            nested_filters = extract_filters_from_scalar_subqueries(
                                where_clause, subq_sel, combined_alias_to_table
                            )
                            out = merge_filter_maps(out, nested_filters)
                        
                        # Extract filters from HAVING clause inside the JOIN subquery
                        having_clause = subq_sel.args.get("having")
                        if having_clause:
                            for c in iter_columns(having_clause):
                                qualified_col = qualify_col_in_scope(c, combined_alias_to_table)
                                if qualified_col:
                                    out["having"].add(qualified_col)
                            # Recursively extract from nested subqueries in HAVING clause
                            nested_filters = extract_filters_from_scalar_subqueries(
                                having_clause, subq_sel, combined_alias_to_table
                            )
                            out = merge_filter_maps(out, nested_filters)
                        
                        # Extract filters from GROUP BY clause inside the JOIN subquery
                        group_clause = subq_sel.args.get("group")
                        if group_clause:
                            for c in iter_columns(group_clause):
                                qualified_col = qualify_col_in_scope(c, combined_alias_to_table)
                                if qualified_col:
                                    out["group_by"].add(qualified_col)
                        
                        # Extract filters from JOIN ON conditions inside the subquery
                        for inner_j in subq_sel.args.get("joins", []) or []:
                            if isinstance(inner_j, Join):
                                on_clause = inner_j.args.get("on")
                                if on_clause:
                                    for c in iter_columns(on_clause):
                                        qualified_col = qualify_col_in_scope(c, combined_alias_to_table)
                                        if qualified_col:
                                            out["join_on"].add(qualified_col)
                                    
                                    # Recursively extract from nested subqueries in JOIN ON
                                    nested_filters = extract_filters_from_scalar_subqueries(
                                        on_clause, subq_sel, combined_alias_to_table
                                    )
                                    out = merge_filter_maps(out, nested_filters)
                        
                        # Extract columns selected in the JOIN subquery
                        # These columns are available in the outer query and can affect filtering
                        # Example: SELECT led.date AS leaseendcontractcurtaileddate FROM ... 
                        # should extract led.date (qualified) as a filter column
                        subq_alias = safe_name(join_expression.alias)
                        for sel_expr in getattr(subq_sel, "selects", []) or []:
                            # Extract all columns from the SELECT expression
                            # This includes columns used in expressions, not just direct column references
                            for col in iter_columns(sel_expr):
                                qualified_col = qualify_col_in_scope(col, combined_alias_to_table)
                                if qualified_col:
                                    # Add the qualified column as a filter column
                                    # Since it's part of a JOIN subquery, add to join_on
                                    out["join_on"].add(qualified_col)
                            
                            # Also recursively extract from nested scalar subqueries in SELECT expressions
                            if hasattr(sel_expr, 'this'):
                                nested_expr = sel_expr.this
                            else:
                                nested_expr = sel_expr
                            
                            nested_filters = extract_filters_from_scalar_subqueries(
                                nested_expr, subq_sel, combined_alias_to_table
                            )
                            out = merge_filter_maps(out, nested_filters)

        for w in sel.find_all(Window):
            parent_select = w.parent
            while parent_select and not isinstance(parent_select, Select):
                parent_select = parent_select.parent
            if parent_select is not sel:
                continue

            for c in iter_columns(w.args.get("partition_by")):
                out["window_partition_by"].add(qualify_col_in_scope(c, alias_to_table))
            for c in iter_columns(w.args.get("order")):
                out["window_order_by"].add(qualify_col_in_scope(c, alias_to_table))

        return out
    except Exception as e:
        logger.error("extract_clause_lineage_for_select error: %s", e, exc_info=True)
        return {}

def extract_filters_from_scalar_subqueries(expr, parent_sel: Select, parent_alias_to_table: dict):
    """
    Extract filters from scalar subqueries (Subquery nodes without aliases) 
    embedded within a SELECT expression. Recursively handles nested subqueries.
    
    Args:
        expr: The expression node that may contain scalar subqueries
        parent_sel: The parent SELECT statement for scope resolution
        parent_alias_to_table: The alias to table mapping from parent SELECT
        
    Returns:
        Dictionary of filter columns organized by clause type
    """
    try:
        filters = {
            "where": set(),
            "group_by": set(),
            "having": set(),
            "join_on": set(),
            "order_by": set(),
            "qualify": set(),
            "window_partition_by": set(),
            "window_order_by": set(),
        }
        
        if not expr:
            return filters
        
        # Build a comprehensive parent scope by walking up the parent chain
        # This ensures we can resolve correlated columns from outer queries
        def build_comprehensive_parent_scope(subq_node, start_alias_to_table, parent_select):
            """Build scope including all outer query scopes for correlated column resolution"""
            scope = start_alias_to_table.copy()
            
            # Start with the parent SELECT's scope (this is the immediate parent)
            if parent_select:
                parent_scope, _ = build_select_scope(parent_select)
                scope.update(parent_scope)
            
            # Walk up from the subquery node to find all outer SELECT scopes
            current = subq_node.parent
            visited_parents = set()
            
            while current:
                if id(current) in visited_parents:
                    break
                visited_parents.add(id(current))
                
                if isinstance(current, Select):
                    outer_alias_to_table, _ = build_select_scope(current)
                    scope.update(outer_alias_to_table)
                elif isinstance(current, Subquery):
                    # Get the parent of this Subquery
                    subq_parent = current.parent
                    while subq_parent and not isinstance(subq_parent, Select):
                        subq_parent = subq_parent.parent
                    if isinstance(subq_parent, Select):
                        outer_alias_to_table, _ = build_select_scope(subq_parent)
                        scope.update(outer_alias_to_table)
                
                # Continue up the chain
                current = current.parent
            
            return scope
        
        def extract_filters_from_select(inner_sel, combined_scope):
            """Extract filters from a SELECT statement and return them"""
            inner_filters = {
                "where": set(),
                "group_by": set(),
                "having": set(),
                "join_on": set(),
                "order_by": set(),
                "qualify": set(),
                "window_partition_by": set(),
                "window_order_by": set(),
            }
            
            # Extract from WHERE clause
            where_clause = inner_sel.args.get("where")
            if where_clause:
                for c in iter_columns(where_clause):
                    qualified_col = qualify_col_in_scope(c, combined_scope)
                    if qualified_col:
                        inner_filters["where"].add(qualified_col)
                # Recursively extract from nested subqueries in WHERE clause
                nested_filters = extract_filters_from_scalar_subqueries(
                    where_clause, inner_sel, combined_scope
                )
                inner_filters = merge_filter_maps(inner_filters, nested_filters)
            
            # Extract from HAVING clause
            having_clause = inner_sel.args.get("having")
            if having_clause:
                for c in iter_columns(having_clause):
                    qualified_col = qualify_col_in_scope(c, combined_scope)
                    if qualified_col:
                        inner_filters["having"].add(qualified_col)
                # Recursively extract from nested subqueries in HAVING clause
                nested_filters = extract_filters_from_scalar_subqueries(
                    having_clause, inner_sel, combined_scope
                )
                inner_filters = merge_filter_maps(inner_filters, nested_filters)
            
            # Extract from GROUP BY clause
            group_clause = inner_sel.args.get("group")
            if group_clause:
                for c in iter_columns(group_clause):
                    qualified_col = qualify_col_in_scope(c, combined_scope)
                    if qualified_col:
                        inner_filters["group_by"].add(qualified_col)
            
            # Extract from JOIN ON conditions
            for j in inner_sel.args.get("joins", []) or []:
                if isinstance(j, Join):
                    on_clause = j.args.get("on")
                    if on_clause:
                        for c in iter_columns(on_clause):
                            qualified_col = qualify_col_in_scope(c, combined_scope)
                            if qualified_col:
                                inner_filters["join_on"].add(qualified_col)
                        # Recursively extract from nested subqueries in JOIN ON
                        nested_filters = extract_filters_from_scalar_subqueries(
                            on_clause, inner_sel, combined_scope
                        )
                        inner_filters = merge_filter_maps(inner_filters, nested_filters)
            
            # Recursively extract filters from nested scalar subqueries in SELECT expressions
            for sel_expr in getattr(inner_sel, "selects", []) or []:
                if hasattr(sel_expr, 'this'):
                    nested_expr = sel_expr.this
                else:
                    nested_expr = sel_expr
                
                nested_filters = extract_filters_from_scalar_subqueries(
                    nested_expr, inner_sel, combined_scope
                )
                inner_filters = merge_filter_maps(inner_filters, nested_filters)
            
            return inner_filters
        
        # Find all Subquery nodes in the expression (scalar subqueries)
        # Use a set to avoid processing the same subquery multiple times
        processed_subqueries = set()
        
        for subq in expr.find_all(Subquery):
            # Only process scalar subqueries (those without aliases, embedded in expressions)
            # Skip if this subquery has an alias (it's a table subquery, not scalar)
            if subq.alias:
                continue
            
            # Avoid processing the same subquery multiple times
            subq_id = id(subq)
            if subq_id in processed_subqueries:
                continue
            processed_subqueries.add(subq_id)
            
            # Get the Select statement inside the subquery
            inner_sel = subq.this
            if not isinstance(inner_sel, Select):
                # If it's a Union or other expression, try to get Select from it
                inner_selects = list(inner_sel.find_all(Select))
                if not inner_selects:
                    continue
                # Process all SELECT statements (for Union, process all branches)
                for sel in inner_selects:
                    # Build scope for the inner SELECT (scalar subquery)
                    inner_alias_to_table, _ = build_select_scope(sel)
                    
                    # Build comprehensive parent scope for correlated column resolution
                    comprehensive_parent_scope = build_comprehensive_parent_scope(subq, parent_alias_to_table, parent_sel)
                    
                    # Merge scopes: inner first, then parent (parent overrides for conflicts)
                    combined_alias_to_table = {**inner_alias_to_table, **comprehensive_parent_scope}
                    
                    # Extract filters from this SELECT
                    inner_filters = extract_filters_from_select(sel, combined_alias_to_table)
                    filters = merge_filter_maps(filters, inner_filters)
            else:
                # Direct Select statement
                # Build scope for the inner SELECT (scalar subquery)
                inner_alias_to_table, _ = build_select_scope(inner_sel)
                
                # Build comprehensive parent scope for correlated column resolution
                comprehensive_parent_scope = build_comprehensive_parent_scope(subq, parent_alias_to_table, parent_sel)
                
                # Merge scopes: inner first, then parent (parent overrides for conflicts)
                combined_alias_to_table = {**inner_alias_to_table, **comprehensive_parent_scope}
                
                # Extract filters from this SELECT
                inner_filters = extract_filters_from_select(inner_sel, combined_alias_to_table)
                filters = merge_filter_maps(filters, inner_filters)
        
        return filters
    except Exception as e:
        logger.error("extract_filters_from_scalar_subqueries error: %s", e, exc_info=True)
        return {
            "where": set(),
            "group_by": set(),
            "having": set(),
            "join_on": set(),
            "order_by": set(),
            "qualify": set(),
            "window_partition_by": set(),
            "window_order_by": set(),
        }

def collect_output_lineage_for_select(sel: Select):
    try:
        alias_to_table, _subq = build_select_scope(sel)
        filters = extract_clause_lineage_for_select(sel, alias_to_table)

        outputs = {}
        for expr in getattr(sel, "selects", []) or []:
            out_alias = safe_name(expr.alias_or_name)
            if not out_alias:
                continue
            srcs = set()
            for c in expr.find_all(Column):
                srcs.add(qualify_col_in_scope(c, alias_to_table))
            for ident in expr.find_all(Identifier):
                srcs.add(safe_name(ident))
            outputs[out_alias] = srcs

        return outputs, filters
    except Exception as e:
        logger.error("collect_output_lineage_for_select error: %s", e, exc_info=True)
        return {}, {}

def recursively_resolve_column_dependencies(root, column_name: str, _visited=None):
    """
    Recursively resolve column dependencies until physical source columns are reached.
    
    This function implements transitive dependency tracking for filter clause columns:
    - If column A depends on column B, and column B depends on column C,
      then this function will return both B and C as dependencies of A.
    - This ensures that filter columns capture the full dependency chain, not just direct dependencies.
    
    Circular Reference Protection:
    - Uses a _visited set to track columns in the current recursion path
    - If a column is encountered again in the same path, recursion stops to prevent infinite loops
    - This safely handles rare cases of circular column dependencies
    
    Args:
        root: The root query expression to search in
        column_name: The column name to resolve (can be intermediate or base)
        _visited: Internal set to track visited columns and prevent circular references
        
    Returns:
        Tuple of (sources_set, filters_dict) where:
        - sources_set: Set of source columns including:
          * Physical base columns (catalog.schema.table.column format)
          * Intermediate columns (e.g., "dtrr.total_dtrr") that are direct dependencies
          * Transitive dependencies resolved recursively
        - filters_dict: Dictionary of filter columns organized by clause type (where, having, etc.)
        
    Example:
        Query structure:
        - avg_occupancy = (d.daysacquired - dtrr.total_dtrr)
        - total_dtrr = SUM(...) WHERE daystoreresidentcount IS NOT NULL
        
        When resolving "avg_occupancy":
        1. Direct expansion finds: "d.daysacquired" and "dtrr.total_dtrr"
        2. "dtrr.total_dtrr" is an intermediate column, so we recursively resolve "total_dtrr"
        3. Recursive resolution finds: "daystoreresidentcount" (from WHERE clause)
        4. Result includes: ["d.daysacquired", "dtrr.total_dtrr", "daystoreresidentcount"]
        
        This ensures that filter columns for avg_occupancy include both the direct dependency
        (total_dtrr) and the transitive dependency (daystoreresidentcount).
    """
    try:
        if _visited is None:
            _visited = set()
        
        # Normalize column name for tracking
        col_key = column_name.lower().strip() if column_name else None
        if not col_key:
            return set(), {}
        
        # Circular reference detection: if we've seen this column in the current recursion path,
        # stop to prevent infinite loops
        if col_key in _visited:
            logger.debug("Circular reference detected for column %s, stopping recursion", column_name)
            return set(), {}
        
        # Mark as visited for this recursion path
        _visited.add(col_key)
        
        # Get initial expansion of this column
        # Use a separate visited set for expand_all_occurrences to avoid conflicts
        # We pass None to let expand_all_occurrences manage its own visited set
        initial_sources, initial_filters = expand_all_occurrences(root, column_name, _visited_columns=None)
        
        if not initial_sources:
            _visited.discard(col_key)
            return set(), initial_filters
        
        # Separate base sources (physical columns) from intermediate sources (derived columns)
        base_sources = set()
        intermediate_sources = set()
        all_filters = initial_filters.copy() if initial_filters else {}
        
        for src in initial_sources:
            if not src:
                continue
            
            # Check if this is a base FQN (physical source column)
            # Base FQNs have format: catalog.schema.table.column (at least 3 dots)
            if is_base_fqn(src):
                base_sources.add(src)
            else:
                # This is an intermediate column - needs recursive expansion
                intermediate_sources.add(src)
        
        # Recursively expand each intermediate source column
        # This is the key to transitive dependency tracking:
        # If avg_occupancy -> total_dtrr, and total_dtrr -> daystoreresidentcount,
        # then we recursively expand total_dtrr to get daystoreresidentcount
        for intermediate_col in intermediate_sources:
            # Extract column name from intermediate source
            # Format could be: "alias.column" or "cte_name.column" or just "column"
            if "." in intermediate_col:
                # For "alias.column" or "cte_name.column", extract just the column name part
                # Example: "dtrr.total_dtrr" -> "total_dtrr"
                parts = intermediate_col.split(".", 1)
                col_part = parts[-1]  # Get the column name part
                
                # Recursively resolve the intermediate column by its name
                # This will trace total_dtrr back to daystoreresidentcount
                recursive_sources, recursive_filters = recursively_resolve_column_dependencies(
                    root, col_part, _visited.copy()
                )
                
                # Merge the recursively resolved sources and filters
                base_sources.update(recursive_sources)
                all_filters = merge_filter_maps(all_filters, recursive_filters)
                
                # If recursive resolution found base sources, we're done with this intermediate column
                # Otherwise, keep the intermediate column as a dependency (it might be a valid output)
                if not recursive_sources:
                    # Couldn't resolve further - might be a base column from a table
                    # Check if it looks like a table.column reference
                    if len(parts) == 2:
                        # Keep the intermediate column as it might be a valid source
                        # This handles cases where intermediate columns are from physical tables
                        pass
            else:
                # Just a column name without alias/CTE prefix
                # Recursively resolve it to find its dependencies
                recursive_sources, recursive_filters = recursively_resolve_column_dependencies(
                    root, intermediate_col, _visited.copy()
                )
                base_sources.update(recursive_sources)
                all_filters = merge_filter_maps(all_filters, recursive_filters)
        
        # Include intermediate columns in the result as well
        # This ensures that if avg_occupancy depends on total_dtrr, and total_dtrr depends on 
        # daystoreresidentcount, then BOTH total_dtrr AND daystoreresidentcount are included
        # in the filter column list
        # Only add intermediate columns that weren't fully resolved to base sources
        for intermediate_col in intermediate_sources:
            # Check if this intermediate column was fully resolved (i.e., all its dependencies are base sources)
            # If not, include it as a dependency itself
            # We include it because it represents a valid intermediate dependency
            if "." in intermediate_col:
                parts = intermediate_col.split(".", 1)
                col_part = parts[-1]
                # If we found base sources from recursive resolution, we've captured the dependencies
                # But we should also include the intermediate column itself if it's a valid filter column
                # For now, we'll include all intermediate columns as they represent valid dependencies
                pass  # Intermediate columns are already represented through their recursive expansion
        
        # Unmark as visited (allows re-visiting in different contexts)
        _visited.discard(col_key)
        
        logger.debug("Recursively resolved %s -> %d base sources, %d intermediate sources", 
                    column_name, len(base_sources), len(intermediate_sources))
        
        # Return both base sources and any intermediate sources that are valid dependencies
        # The intermediate sources represent columns like "total_dtrr" that are dependencies
        # even if they're not physical base columns
        final_sources = base_sources.copy()
        # Add intermediate sources that represent valid dependencies
        # These are columns that the target column directly depends on
        final_sources.update(intermediate_sources)
        
        return final_sources, all_filters
        
    except Exception as e:
        logger.error("recursively_resolve_column_dependencies error for %s: %s", column_name, e, exc_info=True)
        return set(), {}

def expand_sources_in_select_to_base(sel, target_col: str, _seen=None):
    try:
        if _seen is None:
            _seen = set()

        alias_to_table, subq_aliases = build_select_scope(sel)
        outputs, filters = collect_output_lineage_for_select(sel)

        produced_here = None
        target_expr = None
        # Strip whitespace from target_col for matching
        target_col_clean = target_col.strip().lower() if target_col else None
        
        for expr in getattr(sel, "selects", []) or []:
            out_alias = safe_name(expr.alias_or_name)
            if out_alias and target_col_clean and out_alias.lower().strip() == target_col_clean:
                produced_here = out_alias
                # Get the underlying expression (unwrap Alias if needed)
                if hasattr(expr, 'this'):
                    target_expr = expr.this
                else:
                    target_expr = expr
                break
        
        if not produced_here:
            return None, None

        # Extract filters from scalar subqueries within the target expression
        # This is critical for columns built from scalar subqueries with WHERE clauses
        if target_expr:
            # Unwrap common expression wrappers to get to the actual expression
            # For example: Alias(Add(Subquery(...), ...)) -> we need the Add which contains Subquery
            unwrapped_expr = target_expr
            # find_all should work recursively, but let's make sure we're processing the right expression
            scalar_subq_filters = extract_filters_from_scalar_subqueries(
                unwrapped_expr, sel, alias_to_table
            )
            if scalar_subq_filters:
                filters = merge_filter_maps(filters, scalar_subq_filters)
                logger.debug("Extracted scalar subquery filters for %s: %s", target_col, scalar_subq_filters)

        key = (id(sel), produced_here.lower())
        if key in _seen:
            return set(), filters
        _seen.add(key)

        base_sources = set()
        
        # Enhanced: Extract all columns directly from the target expression
        # This ensures we capture all columns used in complex expressions like (d.daysacquired - dtrr.total_dtrr)
        expression_columns = set()
        if target_expr:
            for col in iter_columns(target_expr):
                col_name = safe_name(col.this)
                tbl = safe_name(col.table)
                if tbl:
                    # Preserve the table alias format - the expansion logic will handle CTE resolution
                    expression_columns.add(f"{tbl}.{col_name}")
                else:
                    # No table alias, use qualified name
                    qualified_col = qualify_col_in_scope(col, alias_to_table)
                    if qualified_col:
                        expression_columns.add(qualified_col)
        
        # Combine columns from outputs (existing logic) and direct expression extraction
        # This ensures backward compatibility while adding new functionality
        all_sources = set(outputs.get(produced_here, set())) | expression_columns
        
        for src in all_sources:
            if not src:
                continue

            if is_base_fqn(src):
                base_sources.add(src)
                continue

            if "." in src:
                alias, inner_col = src.split(".", 1)

                # First check if it's a CTE/subquery - these need to be expanded
                if alias in subq_aliases:
                    inner_sel = subq_aliases[alias]
                    inner_sources, _inner_filters = expand_all_occurrences(inner_sel, inner_col)
                    base_sources.update(inner_sources)
                    if _inner_filters:
                        filters = merge_filter_maps(filters, _inner_filters)
                    # Also extract filters from ALL SELECT statements in the CTE/subquery
                    # This handles cases where a derived column depends on WHERE clause filters in the CTE
                    # Example: total_dtrr = SUM(...) WHERE daystoreresidentcount IS NOT NULL
                    # The daystoreresidentcount filter should be captured for columns depending on total_dtrr
                    # We need to extract filters from all SELECT statements, not just the top-level one
                    for cte_sel in inner_sel.find_all(Select):
                        cte_alias_to_table, _ = build_select_scope(cte_sel)
                        cte_filters = extract_clause_lineage_for_select(cte_sel, cte_alias_to_table)
                        if cte_filters:
                            filters = merge_filter_maps(filters, cte_filters)
                    continue
                
                # If it's a table alias, check if it points to a CTE/subquery that needs expansion
                # Example: "d.daysacquired" where "d" is an alias for CTE "vdp"
                if alias in alias_to_table:
                    table_fqn = alias_to_table[alias]
                    expanded_from_cte = False
                    
                    # First, check if the alias itself is a CTE name (e.g., "dtrr.total_dtrr")
                    if alias in subq_aliases:
                        cte_sel = subq_aliases[alias]
                        cte_sources, cte_filters_exp = expand_all_occurrences(cte_sel, inner_col)
                        if cte_sources:
                            base_sources.update(cte_sources)
                            expanded_from_cte = True
                            if cte_filters_exp:
                                filters = merge_filter_maps(filters, cte_filters_exp)
                            # Extract filters from all SELECT statements in the CTE
                            for cte_sel_exp in cte_sel.find_all(Select):
                                cte_alias_to_table_exp, _ = build_select_scope(cte_sel_exp)
                                cte_filters_exp2 = extract_clause_lineage_for_select(cte_sel_exp, cte_alias_to_table_exp)
                                if cte_filters_exp2:
                                    filters = merge_filter_maps(filters, cte_filters_exp2)
                    
                    # Second, check if the table name itself is a CTE name
                    # (e.g., "d" maps to "vdp", and "vdp" is a CTE)
                    if not expanded_from_cte:
                        table_name = table_fqn.split(".")[-1] if table_fqn else None
                        if table_name and table_name in subq_aliases:
                            # Direct match - expand from this CTE
                            cte_sel = subq_aliases[table_name]
                            cte_sources, cte_filters_exp = expand_all_occurrences(cte_sel, inner_col)
                            if cte_sources:
                                base_sources.update(cte_sources)
                                expanded_from_cte = True
                                if cte_filters_exp:
                                    filters = merge_filter_maps(filters, cte_filters_exp)
                                # Extract filters from all SELECT statements in the CTE
                                for cte_sel_exp in cte_sel.find_all(Select):
                                    cte_alias_to_table_exp, _ = build_select_scope(cte_sel_exp)
                                    cte_filters_exp2 = extract_clause_lineage_for_select(cte_sel_exp, cte_alias_to_table_exp)
                                    if cte_filters_exp2:
                                        filters = merge_filter_maps(filters, cte_filters_exp2)
                    
                    # If no direct match, search through all CTEs to find one that produces this column
                    if not expanded_from_cte:
                        for cte_name, cte_sel in subq_aliases.items():
                            # Try to expand the column from this CTE
                            cte_sources, cte_filters_exp = expand_all_occurrences(cte_sel, inner_col)
                            if cte_sources:
                                base_sources.update(cte_sources)
                                expanded_from_cte = True
                                if cte_filters_exp:
                                    filters = merge_filter_maps(filters, cte_filters_exp)
                                # Extract filters from all SELECT statements in the CTE
                                for cte_sel_exp in cte_sel.find_all(Select):
                                    cte_alias_to_table_exp, _ = build_select_scope(cte_sel_exp)
                                    cte_filters_exp2 = extract_clause_lineage_for_select(cte_sel_exp, cte_alias_to_table_exp)
                                    if cte_filters_exp2:
                                        filters = merge_filter_maps(filters, cte_filters_exp2)
                                break
                    
                    # If we didn't expand it from a CTE, add it as a base source
                    if not expanded_from_cte:
                        base_sources.add(f"{table_fqn}.{inner_col}")
                    continue

            resolved_via_projection = False
            for other_out, other_srcs in outputs.items():
                if other_out and other_out.lower() == src.lower():
                    resolved_via_projection = True
                    for c2 in other_srcs:
                        if is_base_fqn(c2):
                            base_sources.add(c2)
                        elif "." in c2:
                            alias2, inner_col2 = c2.split(".", 1)
                            if alias2 in alias_to_table:
                                base_sources.add(f"{alias_to_table[alias2]}.{inner_col2}")
                            elif alias2 in subq_aliases:
                                inner_sel2 = subq_aliases[alias2]
                                inner_sources2, _inner_filters2 = expand_all_occurrences(inner_sel2, inner_col2)
                                base_sources.update(inner_sources2)
                                if _inner_filters2:
                                    filters = merge_filter_maps(filters, _inner_filters2)
                                # Also extract filters from ALL SELECT statements in the CTE/subquery
                                for cte_sel2 in inner_sel2.find_all(Select):
                                    cte_alias_to_table2, _ = build_select_scope(cte_sel2)
                                    cte_filters2 = extract_clause_lineage_for_select(cte_sel2, cte_alias_to_table2)
                                    if cte_filters2:
                                        filters = merge_filter_maps(filters, cte_filters2)
                    break
            if resolved_via_projection:
                continue

            for _sub_alias, inner_sel in subq_aliases.items():
                inner_sources3, _inner_filters3 = expand_all_occurrences(inner_sel, src)
                if inner_sources3:
                    base_sources.update(inner_sources3)
                if _inner_filters3:
                    filters = merge_filter_maps(filters, _inner_filters3)
                # Also extract filters from ALL SELECT statements in the CTE/subquery
                for cte_sel3 in inner_sel.find_all(Select):
                    cte_alias_to_table3, _ = build_select_scope(cte_sel3)
                    cte_filters3 = extract_clause_lineage_for_select(cte_sel3, cte_alias_to_table3)
                    if cte_filters3:
                        filters = merge_filter_maps(filters, cte_filters3)

            if len(alias_to_table) == 1:
                only_alias = next(iter(alias_to_table))
                base_sources.add(f"{alias_to_table[only_alias]}.{src}")

        logger.debug("Expanded %s -> %s", target_col, base_sources)
        return base_sources, filters

    except Exception as e:
        logger.error("expand_sources_in_select_to_base error: %s", e, exc_info=True)
        return set(), {}


def expand_all_occurrences(root, target_col: str, _visited_columns=None):
    """
    Find all occurrences of a target column in the query and expand to base sources.
    
    Args:
        root: The root query expression (Select, Union, etc.)
        target_col: The column name to find and expand
        _visited_columns: Internal parameter to track visited columns for circular reference detection
        
    Returns:
        Tuple of (base_sources_set, filters_dict)
    """
    try:
        if _visited_columns is None:
            _visited_columns = set()
        
        # Handle None or empty target_col
        if not target_col:
            logger.debug("expand_all_occurrences called with None or empty target_col")
            return set(), {}
        
        # Normalize target_col for tracking
        target_col_lower = target_col.lower() if target_col else None
        if not target_col_lower:
            logger.debug("expand_all_occurrences: target_col could not be normalized")
            return set(), {}
        
        # Track this column to prevent infinite recursion in case of circular references
        if target_col_lower in _visited_columns:
            logger.debug("Circular reference detected for column %s, stopping recursion", target_col)
            return set(), {}
        
        _visited_columns.add(target_col_lower)
        
        agg_sources = set()
        agg_filters = {}

        for sel in root.find_all(Select):
            srcs, filt = expand_sources_in_select_to_base(sel, target_col, _seen=None)
            if srcs is not None:
                agg_sources.update(srcs)
                agg_filters = merge_filter_maps(agg_filters, filt)

        logger.debug("expand_all_occurrences for %s -> %s", target_col, agg_sources)
        
        # Remove from visited set after processing (allows re-visiting in different contexts)
        if target_col_lower:
            _visited_columns.discard(target_col_lower)
        
        return agg_sources, agg_filters

    except Exception as e:
        logger.error("expand_all_occurrences error: %s", e, exc_info=True)
        return set(), {}


def unwrap_root_query(node):
    try:
        root = node
        expr = getattr(root, "args", {}).get("expression")
        if expr is not None:
            root = expr
        if isinstance(root, Subquery):
            root = root.this
        return root
    except Exception as e:
        logger.error("unwrap_root_query error: %s", e, exc_info=True)
        return node


def collect_top_level_order_columns(parsed_root, query_root):
    try:
        orders = set()
        for n in (parsed_root, query_root):
            order_expr = getattr(n, "args", {}).get("order")
            for c in iter_columns(order_expr):
                name = safe_name(c.this)
                if name:
                    orders.add(name)
        logger.debug("Top-level order columns: %s", orders)
        return orders
    except Exception as e:
        logger.error("collect_top_level_order_columns error: %s", e, exc_info=True)
        return set()


def get_wrapper_target_table_fqn(parsed):
    try:
        if getattr(parsed, "key", None) == "INSERT":
            t = parsed.this
            if isinstance(t, Table):
                return ".".join(filter(None, [safe_name(t.catalog), safe_name(t.db), safe_name(t.this)]))
        if getattr(parsed, "key", None) == "CREATE":
            t = parsed.this
            if isinstance(t, Table):
                return ".".join(filter(None, [safe_name(t.catalog), safe_name(t.db), safe_name(t.this)]))
        return None
    except Exception as e:
        logger.error("get_wrapper_target_table_fqn error: %s", e, exc_info=True)
        return None

def get_column_full_lineage(fully_qualified_source_column_name: str,
                            fully_qualified_target_column_name: str,
                            sql_query: str):
    """
    Returns lineage mapping from source -> target with filters.
    """
    try:
        parsed = sqlglot.parse_one(sql_query, read="snowflake")
        query_root = unwrap_root_query(parsed)

        src_cat, src_db, src_table, src_col = split_fqn(fully_qualified_source_column_name)
        _tgt_cat, _tgt_db, _tgt_table, tgt_col = split_fqn(fully_qualified_target_column_name)
        
        # Strip whitespace from target column name
        if tgt_col:
            tgt_col = tgt_col.strip()
        if fully_qualified_target_column_name:
            fully_qualified_target_column_name = fully_qualified_target_column_name.strip()

        # Use recursive dependency resolution to capture transitive dependencies
        # This ensures that if avg_occupancy depends on total_dtrr, and total_dtrr depends on 
        # daystoreresidentcount, then both total_dtrr and daystoreresidentcount are captured
        sources, filters = recursively_resolve_column_dependencies(
            query_root, tgt_col or fully_qualified_target_column_name
        )
        
        # If recursive resolution didn't find anything, fall back to direct expansion
        if not sources:
            sources, filters = expand_all_occurrences(query_root, tgt_col or fully_qualified_target_column_name)

        top_order = collect_top_level_order_columns(parsed, query_root)
        if top_order:
            filters = merge_filter_maps(filters, {"order_by": top_order})

        _wrapper_target_fqn = get_wrapper_target_table_fqn(parsed)

        src_table_fqn = join_table_fqn(src_cat, src_db, src_table).lower() if any([src_cat, src_db, src_table]) else None
        src_col_lower = src_col.lower() if src_col else None

        filtered_sources = []
        for s in sources:
            s_lower = s.lower()
            if is_base_fqn(s):
                parts = s_lower.split(".")
                table_fqn = ".".join(parts[:-1])
                col_part = parts[-1]
                if src_table_fqn:
                    if table_fqn == src_table_fqn and (not src_col_lower or src_col_lower == "*" or col_part == src_col_lower):
                        filtered_sources.append(s)
                else:
                    if not src_col_lower or src_col_lower == "*" or col_part == src_col_lower:
                        filtered_sources.append(s)
            else:
                if not src_table_fqn:
                    if not src_col_lower or src_col_lower == "*" or s_lower.split(".")[-1] == src_col_lower:
                        filtered_sources.append(s)

        final_sources = filtered_sources if filtered_sources else sorted(sources)

        return {
            "source_columns": sorted(set(final_sources)),
            "filters": finalize_filters(filters),
        }
    except Exception as e:
        logger.error("get_column_full_lineage error: %s", e, exc_info=True)
        return {"source_columns": [], "filters": {}}
    
def union_filter_dicts(f1: dict, f2: dict) -> dict:
    """
    Merge filter dictionaries by taking the union of values for each key.
    """
    try:
        keys = [
            "where",
            "group_by",
            "having",
            "join_on",
            "order_by",
            "qualify",
            "window_partition_by",
            "window_order_by",
        ]
        out = {}
        for k in keys:
            s1 = set(f1.get(k, []) or [])
            s2 = set(f2.get(k, []) or [])
            out[k] = sorted(s1 | s2)

        logger.debug(f"union_filter_dicts output: {out}")
        return out

    except Exception as e:
        logger.error(f"Error in union_filter_dicts: {e}", exc_info=True)
        return {}


def get_bidirectional_column_lineage(
    fully_qualified_source_column_name: str,
    fully_qualified_target_column_name: str,
    sql_query: str
) -> dict:
    """
    Get bidirectional lineage (src -> tgt and tgt -> src), combining sources and filters.
    """
    try:
        # Forward: src -> tgt
        forward = get_column_full_lineage(
            fully_qualified_source_column_name,
            fully_qualified_target_column_name,
            sql_query,
        )
        logger.debug(f"Forward lineage: {forward}")

        # Reverse: tgt -> src
        reverse = get_column_full_lineage(
            fully_qualified_target_column_name,
            fully_qualified_source_column_name,
            sql_query,
        )
        logger.debug(f"Reverse lineage: {reverse}")

        combined_sources = sorted(
            set(forward.get("source_columns", [])) | set(reverse.get("source_columns", []))
        )
        combined_filters = union_filter_dicts(
            forward.get("filters", {}),
            reverse.get("filters", {})
        )

        result = {
            "source_columns": combined_sources,
            "filters": combined_filters,
        }
        # logger.info(f"Bidirectional lineage result: {result}")  # Suppressed verbose logging
        return result

    except Exception as e:
        logger.error(f"Error in get_bidirectional_column_lineage: {e}", exc_info=True)
        return {
            "source_columns": [],
            "filters": {}
        }
    
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
                result = find_column_sources(all_query_map[current_cte_name], all_query_map, column_name, current_cte_name)
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
    
def get_dependent_columns(df):
    try:
        rows = []
        for _, row in df.iterrows():
            # Check and handle source columns
            if pd.notna(row['source_database']) and pd.notna(row['source_schema']) \
            and pd.notna(row['source_table']) and pd.notna(row['source_column']):
                fully_qualified_source_column_name = (
                    str(row['source_database']).lower() + '.' +
                    str(row['source_schema']).lower() + '.' +
                    str(row['source_table']).lower() + '.' +
                    str(row['source_column']).lower()
                )
            else:
                fully_qualified_source_column_name = None

            # Check and handle target columns - skip row if target columns are missing
            if pd.notna(row['target_database']) and pd.notna(row['target_schema']) \
            and pd.notna(row['target_table']) and pd.notna(row['target_column']):
                fully_qualified_target_column_name = (
                    str(row['target_database']).lower() + '.' +
                    str(row['target_schema']).lower() + '.' +
                    str(row['target_table']).lower() + '.' +
                    str(row['target_column']).lower()
                )
            else:
                # Skip this row if target columns are missing - we can't process without target information
                continue

            sql_query = row.get('query_text', '')
            base_objects_accessed = row.get('base_objects_accessed', {})
            query_id = row.get('query_id', '')
            query_type = row.get('query_type', 'UNKNOWN')
            session_id = row.get('session_id', None)
            dependency_score = row.get('dependency_score', 0)
            dbt_model_file_path = row.get('dbt_model_file_path', '')
            cleaned_query = detect_and_replace_named_parameters(sql_query, static_value="null")
            result = get_bidirectional_column_lineage(fully_qualified_source_column_name, fully_qualified_target_column_name, cleaned_query)

            # Loop through all filters
            for clause, cols in result["filters"].items():
                for col in cols:
                    col = col.lower()
                    f_db, f_schema, f_table, f_col = parse_full_column(col)
                    if f_db is None and f_schema is None:
                        cte_result = resolve_cte_column_source_issue_recursively(f_table, f_col, cleaned_query)

                        if cte_result:
                            if len(cte_result) == 1:
                                f_db = cte_result[0]['source_database'].lower()
                                f_schema = cte_result[0]['source_schema'].lower()
                                f_table = cte_result[0]['source_table'].lower()
                                f_col = cte_result[0]['source_column'].lower()
                            else:
                                qualified_names = [
                                        f"{entry['source_database']}.{entry['source_schema']}.{entry['source_table']}"
                                        for entry in cte_result
                                    ]
                                relevant_qualified_table_name = find_column_table_name(qualified_names, f_col, base_objects_accessed)
                                if relevant_qualified_table_name:
                                    relevant_qualified_table_name_list = relevant_qualified_table_name.split('.')
                                    f_db = relevant_qualified_table_name_list[0].lower()
                                    f_schema = relevant_qualified_table_name_list[1].lower()
                                    f_table = relevant_qualified_table_name_list[2].lower()

                    rows.append({
                        "source_database": f_db,
                        "source_schema": f_schema,
                        "source_table": f_table,
                        "source_column": f_col,
                        "target_database": str(row['target_database']).lower() if pd.notna(row['target_database']) else None,
                        "target_schema": str(row['target_schema']).lower() if pd.notna(row['target_schema']) else None,
                        "target_table": str(row['target_table']).lower() if pd.notna(row['target_table']) else None,
                        "target_column": str(row['target_column']).lower() if pd.notna(row['target_column']) else None,
                        "query_id": query_id,
                        "query_type": query_type,
                        "session_id": session_id,
                        "dependency_score": dependency_score,
                        "dbt_model_file_path": dbt_model_file_path,
                        "is_filter_clause": 1  # Mark this lineage as coming from a filter clause
                    })
        
            logger.info(f"Filter clause columns extracted for query id {query_id}")
        return rows

    except Exception as e:
        try:
            query_id_str = query_id
        except NameError:
            query_id_str = 'unknown'
        logger.error("Error get_dependent_columns for query_id %s: %s", query_id_str, e, exc_info=True)
        return []