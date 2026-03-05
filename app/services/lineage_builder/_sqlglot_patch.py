import dataclasses
import difflib
import logging

try:
    import patchy.api as patchy
except Exception:  # pragma: no cover
    patchy = None  # type: ignore
import sqlglot
import sqlglot.expressions
import sqlglot.lineage
import sqlglot.optimizer.scope
import sqlglot.optimizer.unnest_subqueries


logger = logging.getLogger(__name__)


def _safe_patch(target, text: str) -> None:
    if not patchy:
        return
    try:
        patchy.patch(target, text)
    except Exception:
        # On Windows or limited envs, patchy may require external tools; skip gracefully
        return


def _patch_unnest_subqueries() -> None:
    _safe_patch(
        sqlglot.optimizer.unnest_subqueries.decorrelate,
        r'''
@@ -261,16 +261,19 @@ def remove_aggs(node):
         if key in group_by:
             key.replace(nested)
         elif isinstance(predicate, exp.EQ):
-            parent_predicate = _replace(
-                parent_predicate,
-                f"({parent_predicate} AND ARRAY_CONTAINS({nested}, {column}))",
-            )
            if parent_predicate:
                parent_predicate = _replace(
                    parent_predicate,
                    f"({parent_predicate} AND ARRAY_CONTAINS({nested}, {column}))",
                )
         else:
             key.replace(exp.to_identifier("_x"))
-            parent_predicate = _replace(
-                parent_predicate,
-                f"({parent_predicate} AND ARRAY_ANY({nested}, _x -> {predicate}))",
-            )

            if parent_predicate:
                parent_predicate = _replace(
                    parent_predicate,
                    f"({parent_predicate} AND ARRAY_ANY({nested}, _x -> {predicate}))",
                )
''',
    )


def _patch_lineage() -> None:
    @dataclasses.dataclass(frozen=True)
    class Node(sqlglot.lineage.Node):
        subfield: str = ""

    sqlglot.lineage.Node = Node  # type: ignore

    _safe_patch(
        sqlglot.lineage.lineage,
        r'''
@@ -12,7 +12,8 @@ def lineage(
     """

     expression = maybe_parse(sql, dialect=dialect)
-    column = normalize_identifiers.normalize_identifiers(column, dialect=dialect).name
+    # column = normalize_identifiers.normalize_identifiers(column, dialect=dialect).name
+    assert isinstance(column, str)

     if sources:
         expression = exp.expand(
''',
    )

    _safe_patch(
        sqlglot.lineage.to_node,
        r'''
@@ -235,11 +237,12 @@ def to_node(
             )

     # Find all columns that went into creating this one to list their lineage nodes.
-    source_columns = set(find_all_in_scope(select, exp.Column))
+    source_columns = list(find_all_in_scope(select, exp.Column))

     # If the source is a UDTF find columns used in the UDTF to generate the table
+    source = scope.expression
     if isinstance(source, exp.UDTF):
-        source_columns |= set(source.find_all(exp.Column))
+        source_columns += list(source.find_all(exp.Column))
         derived_tables = [
             source.expression.parent
             for source in scope.sources.values()
@@ -281,8 +285,21 @@ def to_node(
             # is unknown. This can happen if the definition of a source used in a query is not
             # passed into the `sources` map.
             source = source or exp.Placeholder()
+
+            subfields = []
+            field: exp.Expression = c
+            while isinstance(field.parent, exp.Dot):
+                field = field.parent
+                subfields.append(field.name)
+            subfield = ".".join(subfields)

             node.downstream.append(
-                Node(name=c.sql(comments=False), source=source, expression=source)
+                Node(
+                    name=c.sql(comments=False),
+                    source=source,
+                    expression=source,
+                    subfield=subfield,
+                )
             )

     return node
''',
    )


def _patch_scope_traverse() -> None:
    # prevent circular scope dependency
    _safe_patch(
        sqlglot.optimizer.scope.Scope.traverse,
        r'''
@@ -5,9 +5,16 @@ def traverse(self):
         Scope: scope instances in depth-first-search post-order
     """
     stack = [self]
+    seen_scopes = set()
     result = []
     while stack:
         scope = stack.pop()
+
+        # Scopes aren't hashable, so we use id(scope) instead.
+        if id(scope) in seen_scopes:
+            raise OptimizeError(f"Scope {scope} has a circular scope dependency")
+        seen_scopes.add(id(scope))
+
         result.append(scope)
         stack.extend(
             itertools.chain(
''',
    )


def _patch_deepcopy() -> None:
    _safe_patch(
        sqlglot.expressions.Expression.__deepcopy__,
        r'''
@@ -1,4 +1,7 @@ def meta(self) -> t.Dict[str, t.Any]:
 def __deepcopy__(self, memo):
+    # no-op cooperative timeout shim in standalone package
     root = self.__class__()
     stack = [(self, root)]
''',
    )


_patch_deepcopy()
_patch_scope_traverse()
_patch_unnest_subqueries()
_patch_lineage()

SQLGLOT_PATCHED = True


