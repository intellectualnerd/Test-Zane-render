from ._sqlglot_patch import SQLGLOT_PATCHED

import dataclasses
import functools
import logging
import traceback
from collections import defaultdict
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pydantic.dataclasses
import sqlglot
import sqlglot.errors
import sqlglot.lineage
import sqlglot.optimizer
import sqlglot.optimizer.annotate_types
import sqlglot.optimizer.optimizer
import sqlglot.optimizer.qualify

from ._models import _FrozenModel, _ParserBaseModel, _TableName
from .query_types import get_query_type_of_sql, is_create_table_ddl
from .schema_resolver import SchemaInfo, SchemaResolver, SchemaResolverInterface
from .sql_parsing_common import (
    DIALECTS_WITH_CASE_INSENSITIVE_COLS,
    DIALECTS_WITH_DEFAULT_UPPERCASE_COLS,
    QueryType,
    QueryTypeProps,
)
from .sqlglot_utils import (
    DialectOrStr,
    get_dialect,
    get_query_fingerprint_debug,
    is_dialect_instance,
    parse_statement,
)
from .ordered_set import OrderedSet

assert SQLGLOT_PATCHED

logger = logging.getLogger(__name__)

# Suppress third-party library logging
logging.getLogger("sqllineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot").setLevel(logging.WARNING)
logging.getLogger("sqlglot.lineage").setLevel(logging.WARNING)
logging.getLogger("sqlglot.optimizer").setLevel(logging.WARNING)

Urn = str

SQL_PARSE_RESULT_CACHE_SIZE = 1000
SQL_LINEAGE_TIMEOUT_SECONDS = 10
SQL_PARSER_TRACE = False

_DATE_PART_KEYWORDS = {
    "YEAR",
    "QUARTER",
    "MONTH",
    "WEEK",
    "DAY",
    "HOUR",
    "MINUTE",
    "SECOND",
    "MICROSECOND",
    "MILLISECOND",
    "NANOSECOND",
}


assert len(sqlglot.optimizer.optimizer.RULES) >= 10

_OPTIMIZE_RULES = (
    sqlglot.optimizer.optimizer.qualify,
    sqlglot.optimizer.optimizer.pushdown_projections,
    sqlglot.optimizer.optimizer.unnest_subqueries,
    sqlglot.optimizer.optimizer.quote_identifiers,
)

_DEBUG_TYPE_ANNOTATIONS = False


class _ColumnRef(_FrozenModel):
    table: _TableName
    column: str


class ColumnRef(_FrozenModel):
    table: Urn
    column: str


class _DownstreamColumnRef(_ParserBaseModel):
    table: Optional[_TableName] = None
    column: str
    column_type: Optional[sqlglot.exp.DataType] = None


class DownstreamColumnRef(_ParserBaseModel):
    table: Optional[Urn] = None
    column: str
    column_type: Optional[Any] = None
    native_column_type: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.table, self.column, self.native_column_type))


class ColumnTransformation(_FrozenModel):
    is_direct_copy: bool
    column_logic: str


class _ColumnLineageInfo(_ParserBaseModel):
    downstream: _DownstreamColumnRef
    upstreams: List[_ColumnRef]

    logic: Optional[ColumnTransformation] = None


class ColumnLineageInfo(_ParserBaseModel):
    downstream: DownstreamColumnRef
    upstreams: List[ColumnRef]
    logic: Optional[ColumnTransformation] = pydantic.Field(default=None)

    def __hash__(self) -> int:
        return hash((self.downstream, tuple(self.upstreams), self.logic))


class _JoinInfo(_ParserBaseModel):
    join_type: str
    left_tables: List[_TableName]
    right_tables: List[_TableName]
    on_clause: Optional[str]
    columns_involved: List[_ColumnRef]


class JoinInfo(_ParserBaseModel):
    join_type: str
    left_tables: List[Urn]
    right_tables: List[Urn]
    on_clause: Optional[str]
    columns_involved: List[ColumnRef]


class SqlParsingDebugInfo(_ParserBaseModel):
    confidence: float = 0.0

    tables_discovered: int = pydantic.Field(0, exclude=True)
    table_schemas_resolved: int = pydantic.Field(0, exclude=True)

    generalized_statement: Optional[str] = None

    table_error: Optional[Exception] = pydantic.Field(default=None, exclude=True)
    column_error: Optional[Exception] = pydantic.Field(default=None, exclude=True)

    @property
    def error(self) -> Optional[Exception]:
        return self.table_error or self.column_error

    @pydantic.validator("table_error", "column_error")
    def remove_variables_from_error(cls, v: Optional[Exception]) -> Optional[Exception]:
        if v and v.__traceback__:
            traceback.clear_frames(v.__traceback__)
        return v


class SqlParsingResult(_ParserBaseModel):
    query_type: QueryType = QueryType.UNKNOWN
    query_type_props: QueryTypeProps = {}
    query_fingerprint: Optional[str] = None

    in_tables: List[Urn]
    out_tables: List[Urn]

    column_lineage: Optional[List[ColumnLineageInfo]] = None
    joins: Optional[List[JoinInfo]] = None

    debug_info: SqlParsingDebugInfo = pydantic.Field(
        default_factory=lambda: SqlParsingDebugInfo()
    )

    @classmethod
    def make_from_error(cls, error: Exception) -> "SqlParsingResult":
        return cls(
            in_tables=[],
            out_tables=[],
            debug_info=SqlParsingDebugInfo(
                table_error=error,
            ),
        )


def _extract_table_names(
    iterable: Iterable[sqlglot.exp.Table],
) -> OrderedSet[_TableName]:
    return OrderedSet(_TableName.from_sqlglot_table(table) for table in iterable)


def _table_level_lineage(
    statement: sqlglot.Expression, dialect: sqlglot.Dialect
) -> Tuple[AbstractSet[_TableName], AbstractSet[_TableName]]:
    modified = (
        _extract_table_names(
            expr.this
            for expr in statement.find_all(
                sqlglot.exp.Create,
                sqlglot.exp.Insert,
                sqlglot.exp.Update,
                sqlglot.exp.Delete,
                sqlglot.exp.Merge,
                sqlglot.exp.Alter,
            )
            if isinstance(expr.this, sqlglot.exp.Table)
        )
        | _extract_table_names(
            expr.this.this
            for expr in statement.find_all(
                sqlglot.exp.Create,
                sqlglot.exp.Insert,
            )
            if isinstance(expr.this, sqlglot.exp.Schema)
            and isinstance(expr.this.this, sqlglot.exp.Table)
        )
        | _extract_table_names(
            expr.this
            for expr in ([statement] if isinstance(statement, sqlglot.exp.Drop) else [])
            if isinstance(expr.this, sqlglot.exp.Table)
            and expr.this.this
            and expr.this.name
        )
    )

    tables = (
        _extract_table_names(
            table
            for table in statement.find_all(sqlglot.exp.Table)
            if not isinstance(table.parent, sqlglot.exp.Drop)
        )
        - modified
        - {
            _TableName(database=None, db_schema=None, table=cte.alias_or_name)
            for cte in statement.find_all(sqlglot.exp.CTE)
        }
    )
    if isinstance(statement, sqlglot.exp.Update):
        tables = tables | modified

    return tables, modified


_SupportedColumnLineageTypes = Union[
    sqlglot.exp.Query,
    sqlglot.exp.DerivedTable,
]
_SupportedColumnLineageTypesTuple = (sqlglot.exp.Query, sqlglot.exp.DerivedTable)


class UnsupportedStatementTypeError(TypeError):
    pass


class SqlUnderstandingError(Exception):
    pass


@dataclasses.dataclass
class _ColumnResolver:
    sqlglot_db_schema: sqlglot.MappingSchema
    table_schema_normalized_mapping: Dict[_TableName, Dict[str, str]]
    use_case_insensitive_cols: bool

    def schema_aware_fuzzy_column_resolve(
        self, table: Optional[_TableName], sqlglot_column: str
    ) -> str:
        default_col_name = (
            sqlglot_column.lower() if self.use_case_insensitive_cols else sqlglot_column
        )
        if table:
            return self.table_schema_normalized_mapping[table].get(
                sqlglot_column, default_col_name
            )
        else:
            return default_col_name


def _prepare_query_columns(
    statement: sqlglot.exp.Expression,
    dialect: sqlglot.Dialect,
    table_schemas: Dict[_TableName, SchemaInfo],
    default_db: Optional[str],
    default_schema: Optional[str],
) -> Tuple[sqlglot.exp.Expression, "_ColumnResolver"]:
    is_create_ddl = is_create_table_ddl(statement)
    if (
        not isinstance(
            statement,
            _SupportedColumnLineageTypesTuple,
        )
        and not is_create_ddl
    ):
        raise UnsupportedStatementTypeError(
            f"Can only generate column-level lineage for select-like inner statements, not {type(statement)}"
        )

    use_case_insensitive_cols = is_dialect_instance(
        dialect, DIALECTS_WITH_CASE_INSENSITIVE_COLS
    )

    sqlglot_db_schema = sqlglot.MappingSchema(
        dialect=dialect,
        normalize=False,
    )
    table_schema_normalized_mapping: Dict[_TableName, Dict[str, str]] = defaultdict(
        dict
    )
    for table, table_schema in table_schemas.items():
        normalized_table_schema: SchemaInfo = {}
        for col, col_type in table_schema.items():
            if use_case_insensitive_cols:
                col_normalized = (
                    col.upper()
                    if is_dialect_instance(
                        dialect, DIALECTS_WITH_DEFAULT_UPPERCASE_COLS
                    )
                    else col.lower()
                )
            else:
                col_normalized = col

            table_schema_normalized_mapping[table][col_normalized] = col
            normalized_table_schema[col_normalized] = col_type or "UNKNOWN"

        sqlglot_db_schema.add_table(
            table.as_sqlglot_table(),
            column_mapping=normalized_table_schema,
        )

    if use_case_insensitive_cols:

        def _sqlglot_force_column_normalizer(
            node: sqlglot.exp.Expression,
        ) -> sqlglot.exp.Expression:
            if isinstance(node, sqlglot.exp.Column):
                node.this.set("quoted", False)

            return node

        if SQL_PARSER_TRACE:
            logger.debug(
                "Prior to case normalization sql %s",
                statement.sql(pretty=True, dialect=dialect),
            )
        statement = statement.transform(_sqlglot_force_column_normalizer, copy=False)

    if not is_create_ddl:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Prior to column qualification sql %s",
                statement.sql(pretty=True, dialect=dialect),
            )
        try:
            statement = sqlglot.optimizer.optimizer.optimize(
                statement,
                dialect=dialect,
                schema=sqlglot_db_schema,
                qualify_columns=True,
                validate_qualify_columns=False,
                allow_partial_qualification=True,
                identify=True,
                catalog=default_db,
                db=default_schema,
                rules=_OPTIMIZE_RULES,
            )
        except (sqlglot.errors.OptimizeError, ValueError) as e:
            raise SqlUnderstandingError(
                f"sqlglot failed to map columns to their source tables; likely missing/outdated table schema info: {e}"
            ) from e
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Qualified sql %s", statement.sql(pretty=True, dialect=dialect)
            )

        try:
            statement = sqlglot.optimizer.annotate_types.annotate_types(
                statement, schema=sqlglot_db_schema
            )
        except (sqlglot.errors.OptimizeError, sqlglot.errors.ParseError) as e:
            logger.debug("sqlglot failed to annotate or parse types: %s", e)
        if _DEBUG_TYPE_ANNOTATIONS and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Type annotated sql %s", statement.sql(pretty=True, dialect=dialect)
            )

    return statement, _ColumnResolver(
        sqlglot_db_schema=sqlglot_db_schema,
        table_schema_normalized_mapping=table_schema_normalized_mapping,
        use_case_insensitive_cols=use_case_insensitive_cols,
    )

def _create_table_ddl_cll(
    statement: sqlglot.exp.Expression,
    dialect: sqlglot.Dialect,
    column_resolver: _ColumnResolver,
    output_table: Optional[_TableName],
) -> List[_ColumnLineageInfo]:
    column_lineage: List[_ColumnLineageInfo] = []

    assert output_table is not None, (
        "output_table must be set for create DDL statements"
    )

    create_schema: sqlglot.exp.Schema = statement.this
    sqlglot_columns = create_schema.expressions

    for column_def in sqlglot_columns:
        if not isinstance(column_def, sqlglot.exp.ColumnDef):
            # Ignore things like constraints.
            continue

        output_col = column_resolver.schema_aware_fuzzy_column_resolve(
            output_table, column_def.name
        )
        output_col_type = column_def.args.get("kind")

        column_lineage.append(
            _ColumnLineageInfo(
                downstream=_DownstreamColumnRef(
                    table=output_table,
                    column=output_col,
                    column_type=output_col_type,
                ),
                upstreams=[],
            )
        )

    return column_lineage

def _build_alias_resolution_maps(
    root_scope: sqlglot.optimizer.Scope,
) -> Tuple[Dict[str, _TableName], Dict[str, sqlglot.optimizer.Scope]]:
    alias_map: Dict[str, _TableName] = {}

    cte_map: Dict[str, sqlglot.optimizer.Scope] = {}

    for scope in root_scope.traverse():
        sources = getattr(scope, "sources", {})
        for alias, table_expr in sources.items():
            if isinstance(table_expr, sqlglot.exp.Table):
                table_name = _TableName.from_sqlglot_table(table_expr)
                alias_map[alias.upper()] = table_name
                alias_map[alias.lower()] = table_name
            elif isinstance(table_expr, sqlglot.optimizer.Scope):
                cte_map[alias.upper()] = table_expr
                cte_map[alias.lower()] = table_expr

    cte_sources = getattr(root_scope, "cte_sources", {}) or {}
    for name, scope in cte_sources.items():
        cte_map[name.upper()] = scope
        cte_map[name.lower()] = scope

    return alias_map, cte_map


def _derive_upstreams_via_alias_walk(
    lineage_node: sqlglot.lineage.Node,
    alias_map: Optional[Dict[str, _TableName]],
    cte_map: Optional[Dict[str, sqlglot.optimizer.Scope]],
) -> OrderedSet[_ColumnRef]:
    fallback_upstreams: OrderedSet[_ColumnRef] = OrderedSet()
    if alias_map is None and cte_map is None:
        return fallback_upstreams

    for node in lineage_node.walk():
        expr = node.expression
        
        # Skip string literals (single-quoted values)
        if _is_string_literal(expr):
            continue
        
        # Handle function expressions (like CAST, CONVERT) that may contain column references
        if isinstance(expr, (sqlglot.exp.Cast, sqlglot.exp.Convert)):
            # Extract column reference from the function argument
            func_arg = expr.this
            if isinstance(func_arg, sqlglot.exp.Column):
                column_name = _extract_column_name_from_expression(func_arg.this, preserve_quoted=True)
                if not column_name:
                    column_name = func_arg.name
                
                table_alias = None
                if func_arg.table:
                    if isinstance(func_arg.table, sqlglot.exp.Identifier):
                        table_alias = _extract_column_name_from_identifier(func_arg.table, preserve_quoted=True)
                    elif isinstance(func_arg.table, str):
                        table_alias = func_arg.table
                    else:
                        table_alias = getattr(func_arg.table, "name", None)
                
                # Check if column name itself contains table alias (e.g., "a.hRoommate")
                if column_name and '.' in column_name and not table_alias:
                    potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(column_name)
                    if potential_table_alias:
                        table_alias = potential_table_alias
                        column_name = potential_column
                
                if column_name:
                    if table_alias:
                        resolved = _resolve_alias_to_table(
                            str(table_alias),
                            column_name,
                            alias_map,
                            cte_map,
                        )
                        if resolved:
                            resolved_table, resolved_column = resolved
                            fallback_upstreams.add(
                                _ColumnRef(
                                    table=resolved_table,
                                    column=resolved_column,
                                )
                            )
                    else:
                        # No table alias - try to find column in available sources
                        if alias_map:
                            for source_alias, table_ref in alias_map.items():
                                resolved = _resolve_alias_to_table(
                                    source_alias,
                                    column_name,
                                    alias_map,
                                    cte_map,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    fallback_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break
                        if cte_map:
                            for cte_alias, scope in cte_map.items():
                                resolved = _resolve_alias_to_table(
                                    cte_alias,
                                    column_name,
                                    alias_map,
                                    cte_map,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    fallback_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break
            elif isinstance(func_arg, sqlglot.exp.Identifier):
                # Handle double-quoted identifier directly in CAST (e.g., CAST("a.hRoommate" AS ...))
                identifier_name = _extract_column_name_from_identifier(func_arg, preserve_quoted=True)
                if identifier_name:
                    table_alias = None
                    column_name = identifier_name
                    if '.' in identifier_name:
                        potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(identifier_name)
                        if potential_table_alias:
                            table_alias = potential_table_alias
                            column_name = potential_column
                    
                    if table_alias:
                        resolved = _resolve_alias_to_table(
                            str(table_alias),
                            column_name,
                            alias_map,
                            cte_map,
                        )
                        if resolved:
                            resolved_table, resolved_column = resolved
                            fallback_upstreams.add(
                                _ColumnRef(
                                    table=resolved_table,
                                    column=resolved_column,
                                )
                            )
                    else:
                        # No table alias - try to find column in available sources
                        if alias_map:
                            for source_alias, table_ref in alias_map.items():
                                resolved = _resolve_alias_to_table(
                                    source_alias,
                                    column_name,
                                    alias_map,
                                    cte_map,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    fallback_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break
                        if cte_map:
                            for cte_alias, scope in cte_map.items():
                                resolved = _resolve_alias_to_table(
                                    cte_alias,
                                    column_name,
                                    alias_map,
                                    cte_map,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    fallback_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break
            continue
        
        # Handle Identifier expressions directly (e.g., double-quoted identifiers like "dtDOB")
        # Only process if it's a quoted identifier (double-quoted) that might be a column reference
        if isinstance(expr, sqlglot.exp.Identifier):
            # Check if this identifier is quoted (double-quoted) and not part of a Column expression
            # This handles cases like SELECT "dtDOB" where the identifier is a column reference
            is_quoted_identifier = getattr(expr, "quoted", False)
            is_in_column = isinstance(expr.parent, sqlglot.exp.Column) if hasattr(expr, "parent") else False
            
            if is_quoted_identifier and not is_in_column:
                column_name = _extract_column_name_from_identifier(expr, preserve_quoted=True)
                if column_name:
                    # Try to find column in available sources (no table alias)
                    if alias_map:
                        for source_alias, table_ref in alias_map.items():
                            resolved = _resolve_alias_to_table(
                                source_alias,
                                column_name,
                                alias_map,
                                cte_map,
                            )
                            if resolved:
                                resolved_table, resolved_column = resolved
                                fallback_upstreams.add(
                                    _ColumnRef(
                                        table=resolved_table,
                                        column=resolved_column,
                                    )
                                )
                                break
                    if cte_map:
                        for cte_alias, scope in cte_map.items():
                            resolved = _resolve_alias_to_table(
                                cte_alias,
                                column_name,
                                alias_map,
                                cte_map,
                            )
                            if resolved:
                                resolved_table, resolved_column = resolved
                                fallback_upstreams.add(
                                    _ColumnRef(
                                        table=resolved_table,
                                        column=resolved_column,
                                    )
                                )
                                break
            continue
        
        if isinstance(expr, sqlglot.exp.Alias):
            column_expr = expr.this
            
            # Skip if the underlying expression is a string literal
            if _is_string_literal(column_expr):
                continue
            
            if isinstance(column_expr, sqlglot.exp.Column):
                # Extract column name, preserving double-quoted identifiers
                column_name = _extract_column_name_from_expression(column_expr.this, preserve_quoted=True)
                if not column_name:
                    # Fallback to standard extraction
                    column_name = column_expr.name
                
                # Extract table alias, handling double-quoted identifiers
                table_alias = None
                if column_expr.table:
                    if isinstance(column_expr.table, sqlglot.exp.Identifier):
                        table_alias = _extract_column_name_from_identifier(column_expr.table, preserve_quoted=True)
                    elif isinstance(column_expr.table, str):
                        table_alias = column_expr.table
                    else:
                        table_alias = getattr(column_expr.table, "name", None)
                
                # Check if column name itself contains table alias (e.g., "a.dtDOB")
                if column_name and '.' in column_name and not table_alias:
                    potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(column_name)
                    if potential_table_alias:
                        table_alias = potential_table_alias
                        column_name = potential_column
                
                if column_name:
                    if table_alias:
                        # Try to resolve with table alias
                        resolved = _resolve_alias_to_table(
                            str(table_alias),
                            column_name,
                            alias_map,
                            cte_map,
                        )
                        if resolved:
                            resolved_table, resolved_column = resolved
                            fallback_upstreams.add(
                                _ColumnRef(
                                    table=resolved_table,
                                    column=resolved_column,
                                )
                            )
                    else:
                        # No table alias - try to find column in available sources
                        # This handles cases like "dtDOB" (double-quoted identifier without alias)
                        if alias_map:
                            # Try each table alias to find the column
                            for source_alias, table_ref in alias_map.items():
                                resolved = _resolve_alias_to_table(
                                    source_alias,
                                    column_name,
                                    alias_map,
                                    cte_map,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    fallback_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break  # Found in first matching table
                        if cte_map:
                            # Also check CTEs for the column
                            for cte_alias, scope in cte_map.items():
                                resolved = _resolve_alias_to_table(
                                    cte_alias,
                                    column_name,
                                    alias_map,
                                    cte_map,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    fallback_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break  # Found in first matching CTE
    return fallback_upstreams


def _is_string_literal(expression: sqlglot.exp.Expression) -> bool:
    """
    Check if an expression is a string literal (single-quoted).
    According to ANSI SQL rules, single quotes indicate string literals,
    while double quotes indicate identifiers.
    """
    return isinstance(expression, sqlglot.exp.Literal) and expression.is_string


def _parse_quoted_identifier_with_dot(identifier_str: str) -> Tuple[Optional[str], str]:
    """
    Parse a double-quoted identifier that may contain a dot (e.g., "a.dtDOB").
    Returns (table_alias, column_name).
    If no dot is found, returns (None, identifier_str).
    """
    if not identifier_str:
        return None, ""
    
    # Remove surrounding double quotes if present
    cleaned = identifier_str.strip('"')
    
    # Check if it contains a dot (table.column pattern)
    if '.' in cleaned:
        parts = cleaned.split('.', 1)  # Split only on first dot
        if len(parts) == 2:
            table_alias = parts[0].strip()
            column_name = parts[1].strip()
            return table_alias, column_name
    
    return None, cleaned


def _extract_column_name_from_identifier(
    identifier: sqlglot.exp.Identifier, preserve_quoted: bool = True
) -> str:
    """
    Extract column name from an identifier, preserving double-quoted identifiers.
    Handles cases like "a.dtDOB" where the identifier contains a dot.
    For identifiers with dots, extracts just the column part.
    """
    if identifier is None:
        return ""
    
    # Get the name, preserving quoted identifiers
    name = getattr(identifier, "name", None) or str(identifier)
    
    # If the identifier is quoted (double-quoted) and contains a dot,
    # extract just the column part (after the dot)
    if preserve_quoted and getattr(identifier, "quoted", False):
        if '.' in name:
            _, column_part = _parse_quoted_identifier_with_dot(name)
            return column_part
        return name
    
    # For non-quoted identifiers with dots, extract column part
    if '.' in name:
        _, column_part = _parse_quoted_identifier_with_dot(name)
        return column_part
    
    return name


def _extract_column_name_from_expression(
    expression: sqlglot.exp.Expression, preserve_quoted: bool = True
) -> Optional[str]:
    """
    Extract column name from an expression, properly handling:
    - Double-quoted identifiers (e.g., "a.dtDOB") -> treat as column identifier
    - Single-quoted literals (e.g., 'America/Los_Angeles') -> return None (skip)
    - Regular identifiers -> return name
    """
    # Skip string literals (single-quoted)
    if _is_string_literal(expression):
        return None
    
    if isinstance(expression, sqlglot.exp.Identifier):
        return _extract_column_name_from_identifier(expression, preserve_quoted)
    
    if isinstance(expression, sqlglot.exp.Column):
        # Extract from column's identifier
        col_identifier = expression.this
        if isinstance(col_identifier, sqlglot.exp.Identifier):
            return _extract_column_name_from_identifier(col_identifier, preserve_quoted)
        return getattr(expression, "name", None) or str(col_identifier)
    
    return None


def _is_sequence_like_expression(expression: Optional[sqlglot.exp.Expression]) -> bool:
    if expression is None:
        return False

    for node in expression.walk():
        if isinstance(node, sqlglot.exp.NextValueFor):
            return True
        if isinstance(node, sqlglot.exp.Identifier):
            name = (getattr(node, "name", None) or "").upper()
            if name in {"NEXTVAL", "CURRVAL"}:
                return True
        if isinstance(node, sqlglot.exp.Anonymous):
            func_name = (node.name or "").upper()
            if func_name in {"NEXTVAL", "CURRVAL"}:
                return True
    return False


def _resolve_alias_to_table(
    alias: Optional[str],
    column_name: str,
    alias_map: Optional[Dict[str, _TableName]],
    cte_map: Optional[Dict[str, sqlglot.optimizer.Scope]],
    seen: Optional[Set[str]] = None,
) -> Optional[Tuple[_TableName, str]]:
    if not alias:
        return None

    alias_key = alias.upper()

    if seen is None:
        seen = set()
    if alias_key in seen:
        return None
    seen.add(alias_key)

    def _resolve_table_ref(table_ref: _TableName, column: str) -> Optional[Tuple[_TableName, str]]:
        if (
            not table_ref.database
            and table_ref.table
            and cte_map
            and table_ref.table.upper() in cte_map
        ):
            return _resolve_alias_to_table(
                table_ref.table,
                column,
                alias_map,
                cte_map,
                seen,
            )
        return (table_ref, column)

    if alias_map and alias_key in alias_map:
        table_ref = alias_map[alias_key]
        resolved = _resolve_table_ref(table_ref, column_name)
        if resolved:
            return resolved
        return table_ref, column_name

    if cte_map and alias_key in cte_map:
        scope = cte_map[alias_key]
        alias_resolution_attempted = False
        select_expressions = getattr(getattr(scope, "expression", None), "expressions", []) or []
        target_select_expr: Optional[sqlglot.exp.Expression] = None
        for expr in select_expressions:
            if isinstance(expr, sqlglot.exp.Alias):
                alias_name = expr.alias_or_name
                if alias_name and alias_name.upper() == column_name.upper():
                    if target_select_expr is None:
                        target_select_expr = expr
                    column_expr = expr.this
                    if isinstance(column_expr, sqlglot.exp.Column):
                        col_identifier = column_expr.name
                        table_alias = getattr(column_expr.table, "name", None) or getattr(column_expr.table, "this", None)
                        if table_alias is None and isinstance(column_expr.table, sqlglot.exp.Identifier):
                            table_alias = column_expr.table.name
                        elif isinstance(column_expr.table, str):
                            table_alias = column_expr.table
                        if table_alias and col_identifier:
                            alias_resolution_attempted = True
                            resolved = _resolve_alias_to_table(
                                str(table_alias),
                                col_identifier,
                                alias_map,
                                cte_map,
                                seen,
                            )
                            if resolved:
                                return resolved
                        elif col_identifier:
                            sources_dict = getattr(scope, "sources", {}) or {}
                            for source_alias, table_expr in sources_dict.items():
                                try:
                                    source_columns = scope.source_columns(source_alias)
                                except TypeError:
                                    source_columns = []
                                for source_col in source_columns or []:
                                    source_name = getattr(source_col.this, "name", None) or getattr(source_col, "name", None)
                                    if source_name and source_name.upper() == col_identifier.upper():
                                        alias_resolution_attempted = True
                                        resolved = _resolve_alias_to_table(
                                            str(source_alias),
                                            col_identifier,
                                            alias_map,
                                            cte_map,
                                            seen,
                                        )
                                        if resolved:
                                            return resolved
                                        break
        if target_select_expr is not None:
            value_expr = target_select_expr.this if isinstance(target_select_expr, sqlglot.exp.Alias) else target_select_expr
            if _is_sequence_like_expression(value_expr):
                return None
        expression_table_aliases: Set[str] = set()
        if target_select_expr is not None:
            inspect_expr = target_select_expr.this if isinstance(target_select_expr, sqlglot.exp.Alias) else target_select_expr
            for col_expr in inspect_expr.find_all(sqlglot.exp.Column):
                alias_candidate = getattr(col_expr.table, "name", None) or getattr(col_expr.table, "this", None)
                if alias_candidate is None and isinstance(col_expr.table, sqlglot.exp.Identifier):
                    alias_candidate = col_expr.table.name
                elif isinstance(col_expr.table, str):
                    alias_candidate = col_expr.table
                if alias_candidate:
                    expression_table_aliases.add(alias_candidate.upper())
        for col in scope.columns:
            col_identifier = getattr(col.this, "name", None)
            col_name = col_identifier or getattr(col, "name", None)
            if col_name and col_name.upper() == column_name.upper():
                table_alias = getattr(col.table, "name", None) or getattr(col.table, "this", None)
                if table_alias is None and isinstance(col.table, sqlglot.exp.Identifier):
                    table_alias = col.table.name
                elif isinstance(col.table, str):
                    table_alias = col.table
                upper_alias = table_alias.upper() if isinstance(table_alias, str) else None
                if expression_table_aliases and upper_alias and upper_alias not in expression_table_aliases:
                    continue
                if table_alias:
                    alias_resolution_attempted = True
                    resolved = _resolve_alias_to_table(
                        str(table_alias),
                        col_identifier or column_name,
                        alias_map,
                        cte_map,
                        seen,
                    )
                    if resolved:
                        return resolved
        if target_select_expr is None:
            for expr in select_expressions:
                alias_name = expr.alias_or_name
                if alias_name and alias_name.upper() == column_name.upper():
                    target_select_expr = expr
                    break
        if target_select_expr is not None:
            inspect_expr = target_select_expr.this if isinstance(target_select_expr, sqlglot.exp.Alias) else target_select_expr
            contains_column = isinstance(inspect_expr, sqlglot.exp.Column)
            if not contains_column:
                contains_column = any(True for _ in inspect_expr.find_all(sqlglot.exp.Column))
            if not contains_column:
                return None
        if alias_resolution_attempted:
            return None
        # If not found via direct column match, attempt to inspect sources
        sources = getattr(scope, "sources", {})
        for source_alias, table_expr in sources.items():
            try:
                source_columns = scope.source_columns(source_alias)
            except TypeError:
                source_columns = []
            for col in source_columns or []:
                src_identifier = getattr(col.this, "name", None)
                src_name = src_identifier or getattr(col, "name", None)
                if src_name and src_name.upper() == column_name.upper():
                    if isinstance(table_expr, sqlglot.exp.Table):
                        table_ref = _TableName.from_sqlglot_table(table_expr)
                        resolved = _resolve_table_ref(table_ref, src_name)
                        if resolved:
                            return resolved
        # Fallback to first physical table source when no column match was found
        for _, table_expr in sources.items():
            if isinstance(table_expr, sqlglot.exp.Table):
                table_ref = _TableName.from_sqlglot_table(table_expr)
                resolved = _resolve_table_ref(table_ref, column_name)
                if resolved:
                    return resolved

    return None


def _select_statement_cll(
    statement: _SupportedColumnLineageTypes,
    dialect: sqlglot.Dialect,
    root_scope: sqlglot.optimizer.Scope,
    column_resolver: _ColumnResolver,
    output_table: Optional[_TableName],
    table_name_schema_mapping: Dict[_TableName, SchemaInfo],
    default_db: Optional[str] = None,
    default_schema: Optional[str] = None,
) -> List[_ColumnLineageInfo]:
    column_lineage: List[_ColumnLineageInfo] = []

    alias_table_mapping, cte_scope_mapping = _build_alias_resolution_maps(root_scope)

    try:
        output_columns = [
            (select_col.alias_or_name, select_col) for select_col in statement.selects
        ]
        logger.debug("output columns: %s", [col[0] for col in output_columns])

        for output_col, _original_col_expression in output_columns:
            if not output_col or output_col == "*":
                continue

            lineage_node = sqlglot.lineage.lineage(
                output_col,
                statement,
                dialect=dialect,
                scope=root_scope,
                trim_selects=False,
            )

            direct_raw_col_upstreams = _get_direct_raw_col_upstreams(
                lineage_node,
                dialect,
                default_db,
                default_schema,
                alias_table_mapping=alias_table_mapping,
                cte_scope_mapping=cte_scope_mapping,
            )
            fallback_upstreams = _derive_upstreams_via_alias_walk(
                lineage_node,
                alias_table_mapping,
                cte_scope_mapping,
            )
            if fallback_upstreams and not direct_raw_col_upstreams:
                direct_raw_col_upstreams = fallback_upstreams

            original_col_expression = lineage_node.expression
            if output_col.startswith("_col_"):
                output_col = original_col_expression.this.sql(dialect=dialect)

            output_col = column_resolver.schema_aware_fuzzy_column_resolve(
                output_table, output_col
            )

            output_col_type = None
            if original_col_expression.type:
                output_col_type = original_col_expression.type

            direct_resolved_col_upstreams = {
                _ColumnRef(
                    table=edge.table,
                    column=column_resolver.schema_aware_fuzzy_column_resolve(
                        edge.table, edge.column
                    ),
                )
                for edge in direct_raw_col_upstreams
            }

            value_expr = (
                original_col_expression.this
                if isinstance(original_col_expression, sqlglot.exp.Alias)
                else original_col_expression
            )
            if _is_sequence_like_expression(value_expr):
                direct_resolved_col_upstreams = {
                    edge
                    for edge in direct_resolved_col_upstreams
                    if edge.column
                    and "SEQUENCE" not in edge.column.upper()
                    and edge.column.upper() not in {"NEXTVAL", "CURRVAL"}
                }

            if not direct_resolved_col_upstreams:
                logger.debug(f'  "{output_col}" has no upstreams')
            column_lineage.append(
                _ColumnLineageInfo(
                    downstream=_DownstreamColumnRef(
                        table=output_table,
                        column=output_col,
                        column_type=output_col_type,
                    ),
                    upstreams=sorted(direct_resolved_col_upstreams),
                    logic=_get_column_transformation(lineage_node, dialect),
                )
            )

    except (sqlglot.errors.OptimizeError, ValueError, IndexError) as e:
        raise SqlUnderstandingError(
            f"sqlglot failed to compute some lineage: {e}"
        ) from e

    return column_lineage


class _ColumnLineageWithDebugInfo(_ParserBaseModel):
    column_lineage: List[_ColumnLineageInfo]
    joins: Optional[List[_JoinInfo]] = None

    select_statement: Optional[sqlglot.exp.Expression] = None


def _column_level_lineage(
    statement: sqlglot.exp.Expression,
    dialect: sqlglot.Dialect,
    downstream_table: Optional[_TableName],
    table_name_schema_mapping: Dict[_TableName, SchemaInfo],
    default_db: Optional[str],
    default_schema: Optional[str],
) -> _ColumnLineageWithDebugInfo:
    try:
        select_statement = _try_extract_select(statement)
    except Exception as e:
        raise SqlUnderstandingError(
            f"Failed to extract select from statement: {e}"
        ) from e

    try:
        assert select_statement is not None
        (select_statement, column_resolver) = _prepare_query_columns(
            select_statement,
            dialect=dialect,
            table_schemas=table_name_schema_mapping,
            default_db=default_db,
            default_schema=default_schema,
        )
    except UnsupportedStatementTypeError as e:
        e.args = (f"{e.args[0]} (outer statement type: {type(statement)})",)
        logger.debug(e)
        raise e

    if is_create_table_ddl(select_statement):
        column_lineage = _create_table_ddl_cll(
            select_statement,
            dialect=dialect,
            column_resolver=column_resolver,
            output_table=downstream_table,
        )
        return _ColumnLineageWithDebugInfo(
            column_lineage=column_lineage,
            select_statement=select_statement,
        )

    assert isinstance(select_statement, _SupportedColumnLineageTypesTuple)
    try:
        root_scope = sqlglot.optimizer.build_scope(select_statement)
        if root_scope is None:
            raise SqlUnderstandingError(
                f"Failed to build scope for statement - scope was empty: {statement}"
            )
    except (sqlglot.errors.OptimizeError, ValueError, IndexError) as e:
        raise SqlUnderstandingError(
            f"sqlglot failed to preprocess statement: {e}"
        ) from e

    column_lineage = _select_statement_cll(
        select_statement,
        dialect=dialect,
        root_scope=root_scope,
        column_resolver=column_resolver,
        output_table=downstream_table,
        table_name_schema_mapping=table_name_schema_mapping,
        default_db=default_db,
        default_schema=default_schema,
    )

    joins: Optional[List[_JoinInfo]] = None
    try:
        joins = _list_joins(dialect=dialect, root_scope=root_scope)
        logger.debug("Joins: %s", joins)
    except Exception as e:
        logger.debug("Failed to list joins: %s", e)

    return _ColumnLineageWithDebugInfo(
        column_lineage=column_lineage,
        joins=joins,
        select_statement=select_statement,
    )


def _get_direct_raw_col_upstreams(
    lineage_node: sqlglot.lineage.Node,
    dialect: Optional[sqlglot.Dialect] = None,
    default_db: Optional[str] = None,
    default_schema: Optional[str] = None,
    *,
    alias_table_mapping: Optional[Dict[str, _TableName]] = None,
    cte_scope_mapping: Optional[Dict[str, sqlglot.optimizer.Scope]] = None,
) -> OrderedSet[_ColumnRef]:
    direct_raw_col_upstreams: OrderedSet[_ColumnRef] = OrderedSet()
    sequence_like_columns: Set[Tuple[str, str]] = set()

    for node in lineage_node.walk():
        expression = node.expression

        # Skip string literals (single-quoted values) - these are not columns
        if _is_string_literal(expression):
            continue

        if isinstance(expression, sqlglot.exp.Alias):
            value_expr = expression.this
            
            # Skip if the underlying expression is a string literal
            if _is_string_literal(value_expr):
                continue
            
            # Handle CAST expressions inside Alias (e.g., CAST("hTenant" AS ...) AS htenant)
            if isinstance(value_expr, (sqlglot.exp.Cast, sqlglot.exp.Convert)):
                cast_arg = value_expr.this
                if isinstance(cast_arg, sqlglot.exp.Identifier):
                    # Handle double-quoted identifier in CAST (e.g., CAST("hTenant" AS ...))
                    identifier_name = _extract_column_name_from_identifier(cast_arg, preserve_quoted=True)
                    if identifier_name:
                        column_name = identifier_name
                        table_alias = None
                        if '.' in identifier_name:
                            potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(identifier_name)
                            if potential_table_alias:
                                table_alias = potential_table_alias
                                column_name = potential_column
                        
                        if column_name:
                            if table_alias:
                                resolved = _resolve_alias_to_table(
                                    str(table_alias),
                                    column_name,
                                    alias_table_mapping,
                                    cte_scope_mapping,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    direct_raw_col_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                            else:
                                # No table alias - try to find column in available sources
                                if alias_table_mapping:
                                    for source_alias, table_ref in alias_table_mapping.items():
                                        resolved = _resolve_alias_to_table(
                                            source_alias,
                                            column_name,
                                            alias_table_mapping,
                                            cte_scope_mapping,
                                        )
                                        if resolved:
                                            resolved_table, resolved_column = resolved
                                            direct_raw_col_upstreams.add(
                                                _ColumnRef(
                                                    table=resolved_table,
                                                    column=resolved_column,
                                                )
                                            )
                                            break
                                if cte_scope_mapping:
                                    for cte_alias, scope in cte_scope_mapping.items():
                                        resolved = _resolve_alias_to_table(
                                            cte_alias,
                                            column_name,
                                            alias_table_mapping,
                                            cte_scope_mapping,
                                        )
                                        if resolved:
                                            resolved_table, resolved_column = resolved
                                            direct_raw_col_upstreams.add(
                                                _ColumnRef(
                                                    table=resolved_table,
                                                    column=resolved_column,
                                                )
                                            )
                                            break
                elif isinstance(cast_arg, sqlglot.exp.Column):
                    # Handle Column expression in CAST
                    column_name = _extract_column_name_from_expression(cast_arg.this, preserve_quoted=True)
                    if not column_name:
                        column_name = cast_arg.name
                    
                    table_alias = None
                    if cast_arg.table:
                        if isinstance(cast_arg.table, sqlglot.exp.Identifier):
                            table_alias = _extract_column_name_from_identifier(cast_arg.table, preserve_quoted=True)
                        elif isinstance(cast_arg.table, str):
                            table_alias = cast_arg.table
                        else:
                            table_alias = getattr(cast_arg.table, "name", None)
                    
                    if column_name and '.' in column_name and not table_alias:
                        potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(column_name)
                        if potential_table_alias:
                            table_alias = potential_table_alias
                            column_name = potential_column
                    
                    if column_name:
                        if table_alias:
                            resolved = _resolve_alias_to_table(
                                str(table_alias),
                                column_name,
                                alias_table_mapping,
                                cte_scope_mapping,
                            )
                            if resolved:
                                resolved_table, resolved_column = resolved
                                direct_raw_col_upstreams.add(
                                    _ColumnRef(
                                        table=resolved_table,
                                        column=resolved_column,
                                    )
                                )
                        else:
                            if alias_table_mapping:
                                for source_alias, table_ref in alias_table_mapping.items():
                                    resolved = _resolve_alias_to_table(
                                        source_alias,
                                        column_name,
                                        alias_table_mapping,
                                        cte_scope_mapping,
                                    )
                                    if resolved:
                                        resolved_table, resolved_column = resolved
                                        direct_raw_col_upstreams.add(
                                            _ColumnRef(
                                                table=resolved_table,
                                                column=resolved_column,
                                            )
                                        )
                                        break
                            if cte_scope_mapping:
                                for cte_alias, scope in cte_scope_mapping.items():
                                    resolved = _resolve_alias_to_table(
                                        cte_alias,
                                        column_name,
                                        alias_table_mapping,
                                        cte_scope_mapping,
                                    )
                                    if resolved:
                                        resolved_table, resolved_column = resolved
                                        direct_raw_col_upstreams.add(
                                            _ColumnRef(
                                                table=resolved_table,
                                                column=resolved_column,
                                            )
                                        )
                                        break
            
            if _is_sequence_like_expression(value_expr):
                for col_expr in value_expr.walk():
                    if not isinstance(col_expr, sqlglot.exp.Column):
                        continue
                    # Skip if this column expression is actually a string literal
                    if _is_string_literal(col_expr):
                        continue
                    
                    table_alias_obj = col_expr.table
                    if isinstance(table_alias_obj, sqlglot.exp.Identifier):
                        table_alias = _extract_column_name_from_identifier(table_alias_obj, preserve_quoted=True)
                    elif isinstance(table_alias_obj, str):
                        table_alias = table_alias_obj
                    else:
                        table_alias = getattr(table_alias_obj, "name", None) or getattr(table_alias_obj, "this", None)
                    
                    # Extract column name, preserving double-quoted identifiers
                    column_name = _extract_column_name_from_expression(col_expr.this, preserve_quoted=True)
                    if not column_name:
                        column_name = col_expr.name
                    
                    if table_alias and column_name:
                        sequence_like_columns.add((table_alias.upper(), column_name.upper()))
            continue

        if node.downstream:
            continue

        if isinstance(expression, sqlglot.exp.Table):
            table_ref = _TableName.from_sqlglot_table(node.expression)

            if node.name == "*":
                continue

            # Parse the column name, preserving double-quoted identifiers
            parsed_col = sqlglot.parse_one(node.name)
            
            # Skip if it's a string literal
            if _is_string_literal(parsed_col):
                continue
            
            # Extract column name, preserving double-quoted identifiers
            if isinstance(parsed_col, sqlglot.exp.Column):
                col_identifier = parsed_col.this
                if isinstance(col_identifier, sqlglot.exp.Identifier):
                    normalized_col = _extract_column_name_from_identifier(col_identifier, preserve_quoted=True)
                else:
                    normalized_col = getattr(parsed_col, "name", None) or str(col_identifier)
            elif isinstance(parsed_col, sqlglot.exp.Identifier):
                normalized_col = _extract_column_name_from_identifier(parsed_col, preserve_quoted=True)
            else:
                normalized_col = getattr(parsed_col, "name", None) or str(parsed_col.this) if hasattr(parsed_col, "this") else str(parsed_col)
            
            if hasattr(node, "subfield") and node.subfield:
                normalized_col = f"{normalized_col}.{node.subfield}"

            table_alias_key = node.expression.alias_or_name or table_ref.table
            if table_alias_key and (
                table_alias_key.upper(),
                normalized_col.upper(),
            ) in sequence_like_columns:
                continue

            pivots = node.expression.args.get("pivots") or []
            handled_unpivot = False
            if pivots:
                table_alias = table_ref.table or node.expression.alias_or_name
                for pivot in pivots:
                    if not getattr(pivot, "args", None):
                        continue
                    if not pivot.args.get("unpivot"):
                        continue
                    output_columns = [
                        expr.name
                        for expr in (pivot.args.get("expressions") or [])
                        if isinstance(expr, sqlglot.exp.Column) and expr.name
                    ]
                    if not any(
                        output.upper() == normalized_col.upper()
                        for output in output_columns
                    ):
                        continue
                    handled_unpivot = True
                    fields = pivot.args.get("fields") or []
                    for field in fields:
                        input_expressions = getattr(field, "expressions", []) or []
                        for input_expr in input_expressions:
                            if (
                                isinstance(input_expr, sqlglot.exp.Column)
                                and input_expr.name
                                and table_alias
                            ):
                                resolved = _resolve_alias_to_table(
                                    table_alias,
                                    input_expr.name,
                                    alias_table_mapping,
                                    cte_scope_mapping,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    direct_raw_col_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
            if handled_unpivot:
                continue

            table_alias = table_ref.table
            if table_alias and (
                (cte_scope_mapping and table_alias.upper() in cte_scope_mapping)
                or (alias_table_mapping and table_alias.upper() in alias_table_mapping)
            ):
                resolved = _resolve_alias_to_table(
                    table_alias,
                    normalized_col,
                    alias_table_mapping,
                    cte_scope_mapping,
                )
                if resolved:
                    resolved_table, resolved_column = resolved
                    direct_raw_col_upstreams.add(
                        _ColumnRef(
                            table=resolved_table,
                            column=resolved_column,
                        )
                    )
                    continue

            direct_raw_col_upstreams.add(
                _ColumnRef(table=table_ref, column=normalized_col)
            )
        elif isinstance(node.expression, sqlglot.exp.Placeholder) and node.name != "*":
            try:
                parsed = sqlglot.parse_one(node.name, dialect=dialect)
                
                # Skip if parsed expression is a string literal
                if _is_string_literal(parsed):
                    continue
                
                if isinstance(parsed, sqlglot.exp.Column) and parsed.table:
                    table_ref = _TableName.from_sqlglot_table(
                        sqlglot.parse_one(
                            parsed.table, into=sqlglot.exp.Table, dialect=dialect
                        )
                    )

                    if (
                        not (table_ref.database or table_ref.db_schema)
                        and dialect is not None
                    ):
                        table_ref = table_ref.qualified(
                            dialect=dialect,
                            default_db=default_db,
                            default_schema=default_schema,
                        )

                    # Extract column name, preserving double-quoted identifiers
                    if isinstance(parsed.this, sqlglot.exp.Identifier):
                        column_name = _extract_column_name_from_identifier(parsed.this, preserve_quoted=True)
                    else:
                        column_name = str(parsed.this)
                    direct_raw_col_upstreams.add(
                        _ColumnRef(table=table_ref, column=column_name)
                    )
            except Exception as e:
                logger.debug(
                    f"Failed to parse placeholder column expression: {node.name} with dialect {dialect}. The exception was: {e}",
                    exc_info=True,
                )
        # Handle function expressions (like CAST, CONVERT, etc.) that may contain column references
        elif isinstance(expression, (sqlglot.exp.Cast, sqlglot.exp.Convert, sqlglot.exp.Anonymous)):
            # Walk through the function's arguments to find column references
            # For CAST, the first argument (this) is the expression being cast
            # For other functions, check all arguments
            if isinstance(expression, sqlglot.exp.Cast):
                # CAST(expression AS type) - check the expression being cast
                cast_expr = expression.this
                if cast_expr:
                    # Skip if it's a nested function call without column references (e.g., convert_timezone)
                    # Only process if it's a column reference or identifier
                    if isinstance(cast_expr, (sqlglot.exp.Anonymous, sqlglot.exp.Function)):
                        # For nested functions, walk through their arguments to find column references
                        # This handles cases like CAST(convert_timezone(...) AS datetime) where
                        # convert_timezone might have column references in its arguments
                        for arg in cast_expr.expressions if hasattr(cast_expr, 'expressions') else []:
                            if isinstance(arg, sqlglot.exp.Column):
                                column_name = _extract_column_name_from_expression(arg.this, preserve_quoted=True)
                                if not column_name:
                                    column_name = arg.name
                                
                                table_alias = None
                                if arg.table:
                                    if isinstance(arg.table, sqlglot.exp.Identifier):
                                        table_alias = _extract_column_name_from_identifier(arg.table, preserve_quoted=True)
                                    elif isinstance(arg.table, str):
                                        table_alias = arg.table
                                    else:
                                        table_alias = getattr(arg.table, "name", None)
                                
                                if column_name and table_alias:
                                    resolved = _resolve_alias_to_table(
                                        str(table_alias),
                                        column_name,
                                        alias_table_mapping,
                                        cte_scope_mapping,
                                    )
                                    if resolved:
                                        resolved_table, resolved_column = resolved
                                        direct_raw_col_upstreams.add(
                                            _ColumnRef(
                                                table=resolved_table,
                                                column=resolved_column,
                                            )
                                        )
                        # Skip nested functions that don't contain column references
                        continue
                    
                    # Check if it's a column reference (including double-quoted identifiers)
                    if isinstance(cast_expr, sqlglot.exp.Column):
                        column_name = _extract_column_name_from_expression(cast_expr.this, preserve_quoted=True)
                        if not column_name:
                            column_name = cast_expr.name
                        
                        table_alias = None
                        if cast_expr.table:
                            if isinstance(cast_expr.table, sqlglot.exp.Identifier):
                                table_alias = _extract_column_name_from_identifier(cast_expr.table, preserve_quoted=True)
                            elif isinstance(cast_expr.table, str):
                                table_alias = cast_expr.table
                            else:
                                table_alias = getattr(cast_expr.table, "name", None)
                        
                        # Check if column name itself contains table alias (e.g., "a.hRoommate")
                        if column_name and '.' in column_name and not table_alias:
                            potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(column_name)
                            if potential_table_alias:
                                table_alias = potential_table_alias
                                column_name = potential_column
                        
                        if column_name:
                            if table_alias:
                                resolved = _resolve_alias_to_table(
                                    str(table_alias),
                                    column_name,
                                    alias_table_mapping,
                                    cte_scope_mapping,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    direct_raw_col_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                            else:
                                # No table alias - try to find column in available sources
                                if alias_table_mapping:
                                    for source_alias, table_ref in alias_table_mapping.items():
                                        resolved = _resolve_alias_to_table(
                                            source_alias,
                                            column_name,
                                            alias_table_mapping,
                                            cte_scope_mapping,
                                        )
                                        if resolved:
                                            resolved_table, resolved_column = resolved
                                            direct_raw_col_upstreams.add(
                                                _ColumnRef(
                                                    table=resolved_table,
                                                    column=resolved_column,
                                                )
                                            )
                                            break
                                if cte_scope_mapping:
                                    for cte_alias, scope in cte_scope_mapping.items():
                                        resolved = _resolve_alias_to_table(
                                            cte_alias,
                                            column_name,
                                            alias_table_mapping,
                                            cte_scope_mapping,
                                        )
                                        if resolved:
                                            resolved_table, resolved_column = resolved
                                            direct_raw_col_upstreams.add(
                                                _ColumnRef(
                                                    table=resolved_table,
                                                    column=resolved_column,
                                                )
                                            )
                                            break
                    elif isinstance(cast_expr, sqlglot.exp.Identifier):
                        # Handle double-quoted identifier directly in CAST (e.g., CAST("a.hRoommate" AS ...))
                        identifier_name = _extract_column_name_from_identifier(cast_expr, preserve_quoted=True)
                        if identifier_name:
                            # Check if it contains table alias (e.g., "a.hRoommate")
                            table_alias = None
                            column_name = identifier_name
                            if '.' in identifier_name:
                                potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(identifier_name)
                                if potential_table_alias:
                                    table_alias = potential_table_alias
                                    column_name = potential_column
                            
                            if table_alias:
                                resolved = _resolve_alias_to_table(
                                    str(table_alias),
                                    column_name,
                                    alias_table_mapping,
                                    cte_scope_mapping,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    direct_raw_col_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                            else:
                                # No table alias - try to find column in available sources
                                if alias_table_mapping:
                                    for source_alias, table_ref in alias_table_mapping.items():
                                        resolved = _resolve_alias_to_table(
                                            source_alias,
                                            column_name,
                                            alias_table_mapping,
                                            cte_scope_mapping,
                                        )
                                        if resolved:
                                            resolved_table, resolved_column = resolved
                                            direct_raw_col_upstreams.add(
                                                _ColumnRef(
                                                    table=resolved_table,
                                                    column=resolved_column,
                                                )
                                            )
                                            break
                                if cte_scope_mapping:
                                    for cte_alias, scope in cte_scope_mapping.items():
                                        resolved = _resolve_alias_to_table(
                                            cte_alias,
                                            column_name,
                                            alias_table_mapping,
                                            cte_scope_mapping,
                                        )
                                        if resolved:
                                            resolved_table, resolved_column = resolved
                                            direct_raw_col_upstreams.add(
                                                _ColumnRef(
                                                    table=resolved_table,
                                                    column=resolved_column,
                                                )
                                            )
                                            break
            else:
                # For other functions, walk through all arguments to find column references
                for arg in expression.expressions if hasattr(expression, 'expressions') else []:
                    if isinstance(arg, sqlglot.exp.Column):
                        column_name = _extract_column_name_from_expression(arg.this, preserve_quoted=True)
                        if not column_name:
                            column_name = arg.name
                        
                        table_alias = None
                        if arg.table:
                            if isinstance(arg.table, sqlglot.exp.Identifier):
                                table_alias = _extract_column_name_from_identifier(arg.table, preserve_quoted=True)
                            elif isinstance(arg.table, str):
                                table_alias = arg.table
                            else:
                                table_alias = getattr(arg.table, "name", None)
                        
                        if column_name and table_alias:
                            resolved = _resolve_alias_to_table(
                                str(table_alias),
                                column_name,
                                alias_table_mapping,
                                cte_scope_mapping,
                            )
                            if resolved:
                                resolved_table, resolved_column = resolved
                                direct_raw_col_upstreams.add(
                                    _ColumnRef(
                                        table=resolved_table,
                                        column=resolved_column,
                                    )
                                )
            continue
        
        elif isinstance(node.expression, sqlglot.exp.Alias):
            column_expr = node.expression.this
            
            # Skip if the underlying expression is a string literal
            if _is_string_literal(column_expr):
                continue
            
            # Check if the underlying expression is a function (like CAST) that contains a column reference
            if isinstance(column_expr, (sqlglot.exp.Cast, sqlglot.exp.Convert)):
                # Extract column reference from the function argument
                func_arg = column_expr.this
                if isinstance(func_arg, sqlglot.exp.Column):
                    column_name = _extract_column_name_from_expression(func_arg.this, preserve_quoted=True)
                    if not column_name:
                        column_name = func_arg.name
                    
                    table_alias = None
                    if func_arg.table:
                        if isinstance(func_arg.table, sqlglot.exp.Identifier):
                            table_alias = _extract_column_name_from_identifier(func_arg.table, preserve_quoted=True)
                        elif isinstance(func_arg.table, str):
                            table_alias = func_arg.table
                        else:
                            table_alias = getattr(func_arg.table, "name", None)
                    
                    # Check if column name itself contains table alias (e.g., "a.hRoommate")
                    if column_name and '.' in column_name and not table_alias:
                        potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(column_name)
                        if potential_table_alias:
                            table_alias = potential_table_alias
                            column_name = potential_column
                    
                    if column_name:
                        if table_alias:
                            resolved = _resolve_alias_to_table(
                                str(table_alias),
                                column_name,
                                alias_table_mapping,
                                cte_scope_mapping,
                            )
                            if resolved:
                                resolved_table, resolved_column = resolved
                                direct_raw_col_upstreams.add(
                                    _ColumnRef(
                                        table=resolved_table,
                                        column=resolved_column,
                                    )
                                )
                        else:
                            # No table alias - try to find column in available sources
                            if alias_table_mapping:
                                for source_alias, table_ref in alias_table_mapping.items():
                                    resolved = _resolve_alias_to_table(
                                        source_alias,
                                        column_name,
                                        alias_table_mapping,
                                        cte_scope_mapping,
                                    )
                                    if resolved:
                                        resolved_table, resolved_column = resolved
                                        direct_raw_col_upstreams.add(
                                            _ColumnRef(
                                                table=resolved_table,
                                                column=resolved_column,
                                            )
                                        )
                                        break
                            if cte_scope_mapping:
                                for cte_alias, scope in cte_scope_mapping.items():
                                    resolved = _resolve_alias_to_table(
                                        cte_alias,
                                        column_name,
                                        alias_table_mapping,
                                        cte_scope_mapping,
                                    )
                                    if resolved:
                                        resolved_table, resolved_column = resolved
                                        direct_raw_col_upstreams.add(
                                            _ColumnRef(
                                                table=resolved_table,
                                                column=resolved_column,
                                            )
                                        )
                                        break
                elif isinstance(func_arg, sqlglot.exp.Identifier):
                    # Handle double-quoted identifier directly (e.g., CAST("a.hRoommate" AS ...))
                    identifier_name = _extract_column_name_from_identifier(func_arg, preserve_quoted=True)
                    if identifier_name:
                        table_alias = None
                        column_name = identifier_name
                        if '.' in identifier_name:
                            potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(identifier_name)
                            if potential_table_alias:
                                table_alias = potential_table_alias
                                column_name = potential_column
                        
                        if table_alias:
                            resolved = _resolve_alias_to_table(
                                str(table_alias),
                                column_name,
                                alias_table_mapping,
                                cte_scope_mapping,
                            )
                            if resolved:
                                resolved_table, resolved_column = resolved
                                direct_raw_col_upstreams.add(
                                    _ColumnRef(
                                        table=resolved_table,
                                        column=resolved_column,
                                    )
                                )
                        else:
                            # No table alias - try to find column in available sources
                            if alias_table_mapping:
                                for source_alias, table_ref in alias_table_mapping.items():
                                    resolved = _resolve_alias_to_table(
                                        source_alias,
                                        column_name,
                                        alias_table_mapping,
                                        cte_scope_mapping,
                                    )
                                    if resolved:
                                        resolved_table, resolved_column = resolved
                                        direct_raw_col_upstreams.add(
                                            _ColumnRef(
                                                table=resolved_table,
                                                column=resolved_column,
                                            )
                                        )
                                        break
                            if cte_scope_mapping:
                                for cte_alias, scope in cte_scope_mapping.items():
                                    resolved = _resolve_alias_to_table(
                                        cte_alias,
                                        column_name,
                                        alias_table_mapping,
                                        cte_scope_mapping,
                                    )
                                    if resolved:
                                        resolved_table, resolved_column = resolved
                                        direct_raw_col_upstreams.add(
                                            _ColumnRef(
                                                table=resolved_table,
                                                column=resolved_column,
                                            )
                                        )
                                        break
            
            if isinstance(column_expr, sqlglot.exp.Column):
                # Extract column name, preserving double-quoted identifiers
                column_name = _extract_column_name_from_expression(column_expr.this, preserve_quoted=True)
                if not column_name:
                    column_name = column_expr.name
                
                # Extract table alias, handling double-quoted identifiers
                table_alias = None
                if column_expr.table:
                    if isinstance(column_expr.table, sqlglot.exp.Identifier):
                        table_alias = _extract_column_name_from_identifier(column_expr.table, preserve_quoted=True)
                    elif isinstance(column_expr.table, str):
                        table_alias = column_expr.table
                    else:
                        table_alias = getattr(column_expr.table, "name", None)
                
                # Check if column name itself contains table alias (e.g., "a.dtDOB")
                if column_name and '.' in column_name and not table_alias:
                    potential_table_alias, potential_column = _parse_quoted_identifier_with_dot(column_name)
                    if potential_table_alias:
                        table_alias = potential_table_alias
                        column_name = potential_column
                
                if column_name:
                    if table_alias:
                        # Try to resolve with table alias
                        resolved = _resolve_alias_to_table(
                            str(table_alias),
                            column_name,
                            alias_table_mapping,
                            cte_scope_mapping,
                        )
                        if resolved:
                            resolved_table, resolved_column = resolved
                            direct_raw_col_upstreams.add(
                                _ColumnRef(
                                    table=resolved_table,
                                    column=resolved_column,
                                )
                            )
                    else:
                        # No table alias - try to find column in available sources
                        # This handles cases like "dtDOB" (double-quoted identifier without alias)
                        if alias_table_mapping:
                            # Try each table alias to find the column
                            for source_alias, table_ref in alias_table_mapping.items():
                                resolved = _resolve_alias_to_table(
                                    source_alias,
                                    column_name,
                                    alias_table_mapping,
                                    cte_scope_mapping,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    direct_raw_col_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break  # Found in first matching table
                        if cte_scope_mapping:
                            # Also check CTEs for the column
                            for cte_alias, scope in cte_scope_mapping.items():
                                resolved = _resolve_alias_to_table(
                                    cte_alias,
                                    column_name,
                                    alias_table_mapping,
                                    cte_scope_mapping,
                                )
                                if resolved:
                                    resolved_table, resolved_column = resolved
                                    direct_raw_col_upstreams.add(
                                        _ColumnRef(
                                            table=resolved_table,
                                            column=resolved_column,
                                        )
                                    )
                                    break  # Found in first matching CTE
        else:
            pass

    return direct_raw_col_upstreams


def _is_single_column_expression(
    expression: sqlglot.exp.Expression,
) -> bool:
    if isinstance(expression, sqlglot.exp.Alias):
        expression = expression.this

    return isinstance(expression, sqlglot.exp.Column)


def _get_column_transformation(
    lineage_node: sqlglot.lineage.Node,
    dialect: sqlglot.Dialect,
    parent: Optional[sqlglot.lineage.Node] = None,
) -> ColumnTransformation:
    if not lineage_node.downstream:
        if parent:
            expression = parent.expression
            is_copy = _is_single_column_expression(expression)
        else:
            is_copy = True
            expression = lineage_node.expression
        return ColumnTransformation(
            is_direct_copy=is_copy,
            column_logic=expression.sql(dialect=dialect),
        )

    elif len(lineage_node.downstream) > 1 or not _is_single_column_expression(
        lineage_node.expression
    ):
        return ColumnTransformation(
            is_direct_copy=False,
            column_logic=lineage_node.expression.sql(dialect=dialect),
        )

    else:
        return _get_column_transformation(
            lineage_node=lineage_node.downstream[0],
            dialect=dialect,
            parent=lineage_node,
        )


def _get_join_side_tables(
    target: sqlglot.exp.Expression,
    dialect: sqlglot.Dialect,
    scope: sqlglot.optimizer.Scope,
) -> OrderedSet[_TableName]:
    target_alias_or_name = target.alias_or_name
    if (source := scope.sources.get(target_alias_or_name)) and isinstance(
        source, sqlglot.exp.Table
    ):
        return OrderedSet([_TableName.from_sqlglot_table(source)])

    column = sqlglot.exp.Column(
        this=sqlglot.exp.Star(),
        table=sqlglot.exp.Identifier(this=target.alias_or_name),
    )
    columns_used = _get_raw_col_upstreams_for_expression(
        select=column,
        dialect=dialect,
        scope=scope,
    )
    return OrderedSet(col.table for col in columns_used)


def _get_raw_col_upstreams_for_expression(
    select: sqlglot.exp.Expression,
    dialect: sqlglot.Dialect,
    scope: sqlglot.optimizer.Scope,
) -> OrderedSet[_ColumnRef]:
    if not isinstance(scope.expression, sqlglot.exp.Query):
        return OrderedSet()

    original_expression = scope.expression
    updated_expression = scope.expression.select(select, append=False, copy=True)

    try:
        scope.expression = updated_expression
        node = sqlglot.lineage.to_node(
            column=0,
            scope=scope,
            dialect=dialect,
            trim_selects=False,
        )

        return _get_direct_raw_col_upstreams(node, dialect, None, None)
    finally:
        scope.expression = original_expression


def _list_joins(
    dialect: sqlglot.Dialect,
    root_scope: sqlglot.optimizer.Scope,
) -> List[_JoinInfo]:
    joins: List[_JoinInfo] = []

    scope: sqlglot.optimizer.Scope
    for scope in root_scope.traverse():
        join: sqlglot.exp.Join
        for join in scope.expression.find_all(sqlglot.exp.Join):
            left_side_tables: OrderedSet[_TableName] = OrderedSet()
            from_clause: sqlglot.exp.From
            for from_clause in scope.find_all(sqlglot.exp.From):
                left_side_tables.update(
                    _get_join_side_tables(
                        target=from_clause.this,
                        dialect=dialect,
                        scope=scope,
                    )
                )

            right_side_tables: OrderedSet[_TableName] = OrderedSet()
            if join_target := join.this:
                right_side_tables = _get_join_side_tables(
                    target=join_target,
                    dialect=dialect,
                    scope=scope,
                )

            on_clause: Optional[sqlglot.exp.Expression] = join.args.get("on")
            if on_clause:
                joined_columns = _get_raw_col_upstreams_for_expression(
                    select=on_clause, dialect=dialect, scope=scope
                )

                unique_tables = OrderedSet(col.table for col in joined_columns)
                if not unique_tables:
                    logger.debug(
                        "Skipping join because we couldn't resolve the tables from the join condition: %s",
                        join.sql(dialect=dialect),
                    )
                    continue

                left_side_tables = OrderedSet(left_side_tables & unique_tables)
                right_side_tables = OrderedSet(right_side_tables & unique_tables)
            else:
                joined_columns = OrderedSet()

                if not left_side_tables and not right_side_tables:
                    logger.debug(
                        "Skipping join because we couldn't resolve any tables from the join operands: %s",
                        join.sql(dialect=dialect),
                    )
                    continue
                elif len(left_side_tables | right_side_tables) == 1:
                    logger.debug(
                        "Skipping join because we couldn't resolve enough tables from the join operands: %s",
                        join.sql(dialect=dialect),
                    )
                    continue

            joins.append(
                _JoinInfo(
                    join_type=_get_join_type(join),
                    left_tables=list(left_side_tables),
                    right_tables=list(right_side_tables),
                    on_clause=on_clause.sql(dialect=dialect) if on_clause else None,
                    columns_involved=list(sorted(joined_columns)),
                )
            )

    return joins


def _get_join_type(join: sqlglot.exp.Join) -> str:
    if isinstance(join.this, sqlglot.exp.Lateral):
        if join.this.args.get("cross_apply") is not None:
            return "CROSS APPLY"
        return "LATERAL JOIN"

    if join.args.get("kind") == "STRAIGHT":
        return "STRAIGHT_JOIN"

    components = []
    if method := join.args.get("method"):
        components.append(method)
    if join.args.get("global"):
        components.append("GLOBAL")
    if side := join.args.get("side"):
        components.append(side)
    if kind := join.args.get("kind"):
        components.append(kind)

    components.append("JOIN")
    return " ".join(components)


def _extract_select_from_create(
    statement: sqlglot.exp.Create,
) -> sqlglot.exp.Expression:
    inner = statement.expression

    if inner:
        return inner
    else:
        return statement


_UPDATE_ARGS_NOT_SUPPORTED_BY_SELECT: Set[str] = set(
    sqlglot.exp.Update.arg_types.keys()
) - set(sqlglot.exp.Select.arg_types.keys())
_UPDATE_FROM_TABLE_ARGS_TO_MOVE = {"joins", "laterals", "pivot"}


def _extract_select_from_update(
    statement: sqlglot.exp.Update,
) -> sqlglot.exp.Select:
    statement = statement.copy()

    new_expressions = []
    for expr in statement.expressions:
        if isinstance(expr, sqlglot.exp.EQ) and isinstance(
            expr.left, sqlglot.exp.Column
        ):
            new_expressions.append(
                sqlglot.exp.Alias(
                    this=expr.right,
                    alias=expr.left.this,
                )
            )
        else:
            new_expressions.append(expr)

    extra_args: dict = {}
    original_from = statement.args.get("from")
    if original_from and isinstance(original_from.this, sqlglot.exp.Table):
        for k in _UPDATE_FROM_TABLE_ARGS_TO_MOVE:
            if k in original_from.this.args:
                extra_args[k] = original_from.this.args.get(k)
                original_from.this.set(k, None)

    select_statement = sqlglot.exp.Select(
        **{
            **{
                k: v
                for k, v in statement.args.items()
                if k not in _UPDATE_ARGS_NOT_SUPPORTED_BY_SELECT
            },
            **extra_args,
            "expressions": new_expressions,
        }
    )

    if select_statement.args.get("from"):
        select_statement = select_statement.join(
            statement.this, append=True, join_kind="cross"
        )
    else:
        select_statement = select_statement.from_(statement.this)

    return select_statement


def _try_extract_select(
    statement: sqlglot.exp.Expression,
) -> sqlglot.exp.Expression:
    if isinstance(statement, sqlglot.exp.Merge):
        statement = statement.args["using"]
        if isinstance(statement, sqlglot.exp.Table):
            statement = sqlglot.exp.Select().select("*").from_(statement)
    elif isinstance(statement, sqlglot.exp.Insert):
        statement = statement.expression
    elif isinstance(statement, sqlglot.exp.Update):
        statement = _extract_select_from_update(statement)
    elif isinstance(statement, sqlglot.exp.Create):
        statement = _extract_select_from_create(statement)

    if isinstance(statement, sqlglot.exp.Subquery):
        statement = statement.unnest()

    return statement


def _normalize_db_or_schema(
    db_or_schema: Optional[str],
    dialect: sqlglot.Dialect,
) -> Optional[str]:
    if db_or_schema is None:
        return None
    if is_dialect_instance(dialect, "snowflake"):
        return db_or_schema.upper()
    elif is_dialect_instance(dialect, "mssql"):
        return db_or_schema.lower()
    return db_or_schema


def _simplify_select_into(statement: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
    if not (isinstance(statement, sqlglot.exp.Select) and statement.args.get("into")):
        return statement

    into_expr: sqlglot.exp.Into = statement.args["into"].pop()
    into_table = into_expr.this

    create = sqlglot.exp.Create(
        this=into_table,
        kind="TABLE",
        expression=statement,
    )
    return create


def _sqlglot_lineage_nocache(
    sql: sqlglot.exp.ExpOrStr,
    schema_resolver: SchemaResolverInterface,
    default_db: Optional[str] = None,
    default_schema: Optional[str] = None,
    override_dialect: Optional[DialectOrStr] = None,
) -> SqlParsingResult:
    try:
        return _sqlglot_lineage_inner(
            sql=sql,
            schema_resolver=schema_resolver,
            default_db=default_db,
            default_schema=default_schema,
            override_dialect=override_dialect,
        )
    except Exception as e:
        return SqlParsingResult.make_from_error(e)


def _sqlglot_lineage_inner(
    sql: sqlglot.exp.ExpOrStr,
    schema_resolver: SchemaResolverInterface,
    default_db: Optional[str] = None,
    default_schema: Optional[str] = None,
    override_dialect: Optional[DialectOrStr] = None,
) -> SqlParsingResult:
    if override_dialect:
        dialect = get_dialect(override_dialect)
    else:
        dialect = get_dialect(schema_resolver.platform)

    default_db = _normalize_db_or_schema(default_db, dialect)
    default_schema = _normalize_db_or_schema(default_schema, dialect)

    logger.debug("Parsing lineage from sql statement: %s", sql)
    statement = parse_statement(sql, dialect=dialect)

    def _normalize_misordered_datediff(node: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
        if isinstance(node, sqlglot.exp.DateDiff):
            first_arg = node.args.get("this")
            unit_arg = node.args.get("unit")
            if (
                isinstance(first_arg, sqlglot.exp.Column)
                and not first_arg.table
                and first_arg.name
                and first_arg.name.upper() in _DATE_PART_KEYWORDS
            ):
                desired_unit = first_arg.name.upper()
                if (
                    isinstance(unit_arg, sqlglot.exp.Var)
                    and getattr(unit_arg, "this", None)
                    and unit_arg.this.upper() not in _DATE_PART_KEYWORDS
                ):
                    desired_end: sqlglot.exp.Expression
                    expr_arg = node.args.get("expression")
                    if isinstance(expr_arg, sqlglot.exp.Column) and expr_arg.table:
                        table_ref = (
                            expr_arg.table.copy()
                            if isinstance(expr_arg.table, sqlglot.exp.Expression)
                            else sqlglot.exp.Identifier(
                                this=str(expr_arg.table), quoted=False
                            )
                        )
                        desired_end = sqlglot.exp.Column(
                            this=sqlglot.exp.Identifier(
                                this=unit_arg.this.lower(), quoted=False
                            ),
                            table=table_ref,
                        )
                    else:
                        desired_end = sqlglot.exp.column(unit_arg.this.lower())
                    node.set("this", desired_end)
                    node.set(
                        "unit",
                        sqlglot.exp.Var(this=desired_unit, is_string=True),
                    )
                elif not isinstance(unit_arg, sqlglot.exp.Var):
                    node.set(
                        "unit",
                        sqlglot.exp.Var(this=desired_unit, is_string=True),
                    )
                    node.set("this", unit_arg.copy())
        return node

    statement = statement.transform(_normalize_misordered_datediff, copy=True)

    if isinstance(statement, sqlglot.exp.Command):
        raise UnsupportedStatementTypeError(
            f"Got unsupported syntax for statement: {sql}"
        )

    original_statement, statement = statement, statement.copy()

    statement = _simplify_select_into(statement)

    statement = sqlglot.optimizer.qualify.qualify(
        statement,
        dialect=dialect,
        catalog=default_db,
        db=default_schema,
        qualify_columns=False,
        validate_qualify_columns=False,
        allow_partial_qualification=True,
        identify=False,
    )

    tables, modified = _table_level_lineage(statement, dialect=dialect)

    downstream_table: Optional[_TableName] = None
    if len(modified) == 1:
        downstream_table = next(iter(modified))

    table_name_urn_mapping: Dict[_TableName, str] = {}
    table_name_schema_mapping: Dict[_TableName, SchemaInfo] = {}

    for table in tables | modified:
        qualified_table = table.qualified(
            dialect=dialect, default_db=default_db, default_schema=default_schema
        )

        urn, schema_info = schema_resolver.resolve_table(qualified_table)

        table_name_urn_mapping[qualified_table] = urn
        if schema_info:
            table_name_schema_mapping[qualified_table] = schema_info

        table_name_urn_mapping[table] = urn

    total_tables_discovered = len(tables | modified)
    total_schemas_resolved = len(table_name_schema_mapping)
    debug_info = SqlParsingDebugInfo(
        confidence=(
            0.9
            if total_tables_discovered == total_schemas_resolved
            else 0.2 + 0.3 * total_schemas_resolved / total_tables_discovered
        ),
        tables_discovered=total_tables_discovered,
        table_schemas_resolved=total_schemas_resolved,
    )
    logger.debug(
        f"Resolved {total_schemas_resolved} of {total_tables_discovered} table schemas"
    )
    if SQL_PARSER_TRACE:
        for qualified_table, schema_info in table_name_schema_mapping.items():
            logger.debug(
                "Table name %s resolved to %s with schema %s",
                qualified_table,
                table_name_urn_mapping[qualified_table],
                schema_info,
            )

    column_lineage: Optional[List[_ColumnLineageInfo]] = None
    joins = None
    try:
        column_lineage_debug_info = _column_level_lineage(
            statement,
            dialect=dialect,
            downstream_table=downstream_table,
            table_name_schema_mapping=table_name_schema_mapping,
            default_db=default_db,
            default_schema=default_schema,
        )
        column_lineage = column_lineage_debug_info.column_lineage
        joins = column_lineage_debug_info.joins
    except Exception as e:
        logger.debug(f"Failed to generate column-level lineage: {e}", exc_info=True)
        debug_info.column_error = e

    in_urns = sorted({table_name_urn_mapping[table] for table in tables})
    out_urns = sorted({table_name_urn_mapping[table] for table in modified})
    column_lineage_urns = None
    if column_lineage:
        try:
            column_lineage_urns = [
                _translate_internal_column_lineage(
                    table_name_urn_mapping, internal_col_lineage, dialect=dialect
                )
                for internal_col_lineage in column_lineage
            ]
        except KeyError as e:
            logger.debug(
                f"Failed to translate column lineage to urns: {e}", exc_info=True
            )
            debug_info.column_error = e
    joins_urns = None
    if joins is not None:
        try:
            joins_urns = _translate_internal_joins(
                table_name_urn_mapping, raw_joins=joins, dialect=dialect
            )
        except KeyError as e:
            logger.debug(f"Failed to translate joins to urns: {e}", exc_info=True)

    query_type, query_type_props = get_query_type_of_sql(
        original_statement, dialect=dialect
    )
    query_fingerprint, debug_info.generalized_statement = get_query_fingerprint_debug(
        original_statement, dialect
    )
    return SqlParsingResult(
        query_type=query_type,
        query_type_props=query_type_props,
        query_fingerprint=query_fingerprint,
        in_tables=in_urns,
        out_tables=out_urns,
        column_lineage=column_lineage_urns,
        joins=joins_urns,
        debug_info=debug_info,
    )


def _translate_internal_column_lineage(
    table_name_urn_mapping: Dict[_TableName, str],
    raw_column_lineage: _ColumnLineageInfo,
    dialect: sqlglot.Dialect,
) -> ColumnLineageInfo:
    downstream_urn = None
    if raw_column_lineage.downstream.table:
        downstream_urn = table_name_urn_mapping.get(raw_column_lineage.downstream.table)
    return ColumnLineageInfo(
        downstream=DownstreamColumnRef(
            table=downstream_urn,
            column=raw_column_lineage.downstream.column,
            column_type=None,
            native_column_type=(
                raw_column_lineage.downstream.column_type.sql(dialect=dialect)
                if raw_column_lineage.downstream.column_type
                else None
            ),
        ),
        upstreams=[
            ColumnRef(
                table=table_name_urn_mapping[upstream.table],
                column=upstream.column,
            )
            for upstream in raw_column_lineage.upstreams
            if upstream.table in table_name_urn_mapping
        ],
        logic=raw_column_lineage.logic,
    )


def _translate_internal_joins(
    table_name_urn_mapping: Dict[_TableName, str],
    raw_joins: List[_JoinInfo],
    dialect: sqlglot.Dialect,
) -> List[JoinInfo]:
    joins = []
    for raw_join in raw_joins:
        try:
            joins.append(
                JoinInfo(
                    join_type=raw_join.join_type,
                    left_tables=[
                        table_name_urn_mapping[table] for table in raw_join.left_tables
                    ],
                    right_tables=[
                        table_name_urn_mapping[table] for table in raw_join.right_tables
                    ],
                    on_clause=raw_join.on_clause,
                    columns_involved=[
                        ColumnRef(
                            table=table_name_urn_mapping[col.table],
                            column=col.column,
                        )
                        for col in raw_join.columns_involved
                    ],
                )
            )
        except KeyError:
            continue
    return joins


def sqlglot_lineage(
    sql: sqlglot.exp.ExpOrStr,
    schema_resolver: SchemaResolverInterface,
    default_db: Optional[str] = None,
    default_schema: Optional[str] = None,
    override_dialect: Optional[DialectOrStr] = None,
) -> SqlParsingResult:
    return _sqlglot_lineage_nocache(
        sql, schema_resolver, default_db, default_schema, override_dialect
    )
