import functools
from typing import Any, Optional

import sqlglot
from pydantic import BaseModel


class _ParserBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


@functools.total_ordering
class _FrozenModel(_ParserBaseModel):
    class Config:
        frozen = True

    def __lt__(self, other: "_FrozenModel") -> bool:
        for field in self.__fields__:
            self_v = getattr(self, field)
            other_v = getattr(other, field)

            if self_v is None and other_v is not None:
                return False
            elif self_v is not None and other_v is None:
                return True
            elif self_v != other_v:
                return self_v < other_v

        return False


class _TableName(_FrozenModel):
    database: Optional[str] = None
    db_schema: Optional[str] = None
    table: str

    def as_sqlglot_table(self) -> sqlglot.exp.Table:
        return sqlglot.exp.Table(
            catalog=(
                sqlglot.exp.Identifier(this=self.database) if self.database else None
            ),
            db=sqlglot.exp.Identifier(this=self.db_schema) if self.db_schema else None,
            this=sqlglot.exp.Identifier(this=self.table),
        )

    def qualified(
        self,
        dialect: sqlglot.Dialect,
        default_db: Optional[str] = None,
        default_schema: Optional[str] = None,
    ) -> "_TableName":
        database = self.database or default_db
        db_schema = self.db_schema or default_schema

        return _TableName(
            database=database,
            db_schema=db_schema,
            table=self.table,
        )

    @classmethod
    def from_sqlglot_table(
        cls,
        table: sqlglot.exp.Table,
        default_db: Optional[str] = None,
        default_schema: Optional[str] = None,
    ) -> "_TableName":
        if isinstance(table.this, sqlglot.exp.Dot):
            parts = []
            exp = table.this
            while isinstance(exp, sqlglot.exp.Dot):
                parts.append(exp.this.name)
                exp = exp.expression
            parts.append(exp.name)
            table_name = ".".join(parts)
        else:
            table_name = table.this.name
        return cls(
            database=table.catalog or default_db,
            db_schema=table.db or default_schema,
            table=table_name,
        )


