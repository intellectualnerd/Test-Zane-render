import enum
from typing import Optional


PLATFORMS_WITH_CASE_SENSITIVE_TABLES = {
    "bigquery",
}

DIALECTS_WITH_CASE_INSENSITIVE_COLS = {
    "bigquery",
    "snowflake",
    "teradata",
    "mssql",
    "oracle",
}

DIALECTS_WITH_DEFAULT_UPPERCASE_COLS = {
    "snowflake",
    "oracle",
}

assert DIALECTS_WITH_DEFAULT_UPPERCASE_COLS.issubset(
    DIALECTS_WITH_CASE_INSENSITIVE_COLS
)


class QueryType(enum.Enum):
    UNKNOWN = "UNKNOWN"

    CREATE_DDL = "CREATE_DDL"
    CREATE_VIEW = "CREATE_VIEW"
    CREATE_TABLE_AS_SELECT = "CREATE_TABLE_AS_SELECT"
    CREATE_OTHER = "CREATE_OTHER"

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    MERGE = "MERGE"

    def is_create(self) -> bool:
        return self in {
            QueryType.CREATE_DDL,
            QueryType.CREATE_VIEW,
            QueryType.CREATE_TABLE_AS_SELECT,
            QueryType.CREATE_OTHER,
        }


QueryTypeProps = dict[str, object]


