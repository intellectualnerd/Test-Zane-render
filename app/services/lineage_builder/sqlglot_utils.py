import functools
import logging
import re
from typing import Dict, Iterable, Optional, Tuple, Union

import sqlglot
import sqlglot.errors
import sqlglot.optimizer.eliminate_ctes

from .fingerprint_utils import generate_hash

logger = logging.getLogger(__name__)
DialectOrStr = Union[sqlglot.Dialect, str]
SQL_PARSE_CACHE_SIZE = 1000
FORMAT_QUERY_CACHE_SIZE = 1000


def _get_dialect_str(platform: str) -> str:
    if platform == "presto-on-hive":
        return "hive"
    elif platform == "mssql":
        return "tsql"
    elif platform == "athena":
        return "trino"
    elif platform == "salesforce":
        return "databricks"
    elif platform in {"mysql", "mariadb"}:
        return "mysql, normalization_strategy = lowercase"
    elif platform == "dremio":
        return "drill"
    else:
        return platform


def get_dialect(platform: DialectOrStr) -> sqlglot.Dialect:
    if isinstance(platform, sqlglot.Dialect):
        return platform
    return sqlglot.Dialect.get_or_raise(_get_dialect_str(platform))


def is_dialect_instance(
    dialect: sqlglot.Dialect, platforms: Union[str, Iterable[str]]
) -> bool:
    platforms = [platforms] if isinstance(platforms, str) else list(platforms)
    dialects = [get_dialect(platform) for platform in platforms]
    if any(isinstance(dialect, dialect_class.__class__) for dialect_class in dialects):
        return True
    return False


@functools.lru_cache(maxsize=SQL_PARSE_CACHE_SIZE)
def _parse_statement(
    sql: sqlglot.exp.ExpOrStr, dialect: sqlglot.Dialect
) -> sqlglot.Expression:
    statement: sqlglot.Expression = sqlglot.maybe_parse(
        sql, dialect=dialect, error_level=sqlglot.ErrorLevel.IMMEDIATE
    )
    return statement


def parse_statement(
    sql: sqlglot.exp.ExpOrStr, dialect: sqlglot.Dialect
) -> sqlglot.Expression:
    return _parse_statement(sql, dialect).copy()


def _expression_to_string(
    expression: sqlglot.exp.ExpOrStr, platform: DialectOrStr
) -> str:
    if isinstance(expression, str):
        return expression
    return expression.sql(dialect=get_dialect(platform))


PLACEHOLDER_BACKWARD_FINGERPRINT_NORMALIZATION = re.compile(r"(%s|\$\d|\?)")

_BASIC_NORMALIZATION_RULES = {
    re.compile(r"/\*.*?\*/", re.DOTALL): "",
    re.compile(r"--.*$", re.MULTILINE): "",
    re.compile(r"\s+"): " ",
    re.compile(r"^\s+|[\s;]+$"): "",
    re.compile(r"\b\d+\b"): "?",
    re.compile(r"'[^']*'"): "?",
    re.compile(
        r"\b(IN|VALUES)\s*\( ?(?:%s|\$\d|\?)(?:, ?(?:%s|\$\d|\?))* ?\)", re.IGNORECASE
    ): r"\1 (?)",
    re.compile(r"\( "): "(",
    re.compile(r" \)"): ")",
    re.compile(r"\b ,"): ",",
    re.compile(r"\b,\b"): ", ",
    PLACEHOLDER_BACKWARD_FINGERPRINT_NORMALIZATION: "?",
}

_TABLE_NAME_NORMALIZATION_RULES = {
    re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    ): "00000000-0000-0000-0000-000000000000",
    re.compile(
        r"[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}",
        re.IGNORECASE,
    ): "00000000_0000_0000_0000_000000000000",
    re.compile(r"\b(ge_tmp_|ge_temp_|gx_temp_)[0-9a-f]{8}\b", re.IGNORECASE): r"\1abcdefgh",
    re.compile(r"\b(\w+)(19|20)\d{4}\b"): r"\1YYYYMM",
    re.compile(r"\b(\w+)(19|20)\d{6}\b"): r"\1YYYYMMDD",
    re.compile(r"\b(\w+)(19|20)\d{8}\b"): r"\1YYYYMMDDHH",
    re.compile(r"\b(\w+)(19|20)\d{10}\b"): r"\1YYYYMMDDHHMM",
}


def generalize_query_fast(
    expression: sqlglot.exp.ExpOrStr,
    dialect: DialectOrStr,
    change_table_names: bool = False,
) -> str:
    if isinstance(expression, sqlglot.exp.Expression):
        expression = expression.sql(dialect=get_dialect(dialect))
    query_text = expression

    REGEX_REPLACEMENTS = {
        **_BASIC_NORMALIZATION_RULES,
        **(_TABLE_NAME_NORMALIZATION_RULES if change_table_names else {}),
    }

    for pattern, replacement in REGEX_REPLACEMENTS.items():
        query_text = pattern.sub(replacement, query_text)
    return query_text


def generalize_query(expression: sqlglot.exp.ExpOrStr, dialect: DialectOrStr) -> str:
    dialect = get_dialect(dialect)
    expression = sqlglot.maybe_parse(expression, dialect=dialect)

    def _simplify_node_expressions(node: sqlglot.exp.Expression) -> None:
        is_last_literal = True
        for i, expression in reversed(list(enumerate(node.expressions))):
            if isinstance(expression, sqlglot.exp.Literal):
                if is_last_literal:
                    node.expressions[i] = sqlglot.exp.Placeholder()
                    is_last_literal = False
                else:
                    node.expressions.pop(i)
            elif isinstance(expression, sqlglot.exp.Tuple):
                _simplify_node_expressions(expression)

    def _strip_expression(
        node: sqlglot.exp.Expression,
    ) -> Optional[sqlglot.exp.Expression]:
        node.comments = None

        if isinstance(node, (sqlglot.exp.In, sqlglot.exp.Values)):
            _simplify_node_expressions(node)
        elif isinstance(node, sqlglot.exp.Literal):
            return sqlglot.exp.Placeholder()

        return node

    return expression.transform(_strip_expression, copy=True).sql(dialect=dialect)


def get_query_fingerprint_debug(
    expression: sqlglot.exp.ExpOrStr,
    platform: DialectOrStr,
    fast: bool = False,
    secondary_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    try:
        if not fast:
            dialect = get_dialect(platform)
            expression_sql = generalize_query(expression, dialect=dialect)
            expression_sql = PLACEHOLDER_BACKWARD_FINGERPRINT_NORMALIZATION.sub(
                "?", expression_sql
            )
        else:
            expression_sql = generalize_query_fast(expression, dialect=platform)
    except (ValueError, sqlglot.errors.SqlglotError) as e:
        if not isinstance(expression, str):
            raise
        logger.debug("Failed to generalize query for fingerprinting: %s", e)
        expression_sql = None

    text = expression_sql or _expression_to_string(expression, platform=platform)
    if secondary_id:
        text = text + " -- " + secondary_id
    fingerprint = generate_hash(text=text)
    return fingerprint, expression_sql


def get_query_fingerprint(
    expression: sqlglot.exp.ExpOrStr,
    platform: DialectOrStr,
    fast: bool = False,
    secondary_id: Optional[str] = None,
) -> str:
    return get_query_fingerprint_debug(
        expression=expression, platform=platform, fast=fast, secondary_id=secondary_id
    )[0]


@functools.lru_cache(maxsize=FORMAT_QUERY_CACHE_SIZE)
def try_format_query(
    expression: sqlglot.exp.ExpOrStr, platform: DialectOrStr, raises: bool = False
) -> str:
    try:
        dialect = get_dialect(platform)
        parsed_expression = parse_statement(expression, dialect=dialect)
        return parsed_expression.sql(dialect=dialect, pretty=True)
    except Exception as e:
        if raises:
            raise
        logger.debug("Failed to format query: %s", e)
        return _expression_to_string(expression, platform=platform)


def detach_ctes(
    sql: sqlglot.exp.ExpOrStr, platform: str, cte_mapping: Dict[str, str]
) -> sqlglot.exp.Expression:
    dialect = get_dialect(platform)
    statement = parse_statement(sql, dialect=dialect)

    if not cte_mapping:
        return statement

    def replace_cte_refs(node: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
        if (
            isinstance(node, sqlglot.exp.Identifier)
            and node.parent
            and not isinstance(node.parent.parent, sqlglot.exp.CTE)
            and node.name in cte_mapping
        ):
            full_new_name = cte_mapping[node.name]
            table_expr = sqlglot.maybe_parse(
                full_new_name, dialect=dialect, into=sqlglot.exp.Table
            )

            parent = node.parent

            if "catalog" in parent.arg_types and table_expr.catalog:
                parent.set("catalog", table_expr.catalog)
            if "db" in parent.arg_types and table_expr.db:
                parent.set("db", table_expr.db)

            new_node = sqlglot.exp.Identifier(this=table_expr.name)

            return new_node
        else:
            return node

    statement = statement.copy()
    statement = statement.transform(replace_cte_refs, copy=False)

    max_eliminate_calls = 5
    for _ in range(max_eliminate_calls):
        new_statement = sqlglot.optimizer.eliminate_ctes.eliminate_ctes(
            statement.copy()
        )
        if new_statement == statement:
            break
        statement = new_statement

    return statement


