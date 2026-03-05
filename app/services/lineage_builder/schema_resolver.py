from typing import Dict, List, Optional, Protocol, Set, Tuple

from ._models import _TableName

# A lightweight table schema: column -> type mapping.
SchemaInfo = Dict[str, str]


class SchemaResolverInterface(Protocol):
    @property
    def platform(self) -> str: ...

    def includes_temp_tables(self) -> bool: ...

    def resolve_table(self, table: _TableName) -> Tuple[str, Optional[SchemaInfo]]: ...

    def __hash__(self) -> int:
        return id(self)


class SchemaResolver(SchemaResolverInterface):
    def __init__(
        self,
        *,
        platform: str,
        platform_instance: Optional[str] = None,
        env: str = "PROD",
    ):
        self._platform = platform
        self.platform_instance = platform_instance
        self.env = env
        self._schema_cache: Dict[str, Optional[SchemaInfo]] = {}

    @property
    def platform(self) -> str:
        return self._platform

    def includes_temp_tables(self) -> bool:
        return False

    def add_raw_schema_info(self, urn: str, schema_info: SchemaInfo) -> None:
        self._schema_cache[urn] = schema_info

    def get_urn_for_table(self, table: _TableName) -> str:
        table_name = ".".join(filter(None, [table.database, table.db_schema, table.table]))
        return f"{self.platform}://{self.env}/{table_name}"

    def resolve_table(self, table: _TableName) -> Tuple[str, Optional[SchemaInfo]]:
        urn = self.get_urn_for_table(table)
        return urn, self._schema_cache.get(urn)

    # helpers for testability
    def has_urn(self, urn: str) -> bool:
        return urn in self._schema_cache and self._schema_cache[urn] is not None


