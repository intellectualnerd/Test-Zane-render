from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from typing import List, Dict, Any, Optional
import json
import psycopg2
import psycopg2.extras
from langchain.schema import Document

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "chroma_collection_setup")
LINEAGE_CSV_PATH = os.getenv("LINEAGE_CSV_PATH", "temp_lineage_data/lineage_output_deep.csv")
DATABASE_URL = os.getenv("DATABASE_URL")

# OpenAI embedding model for vector store
embedding = (
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)
# OpenAI LLM for existing functionality (impact analysis, etc.)
LLM = (
    ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)
# OpenAI LLM specifically for chatbot/agents
# Use gpt-4o-mini for general chat
CHAT_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
# Use OpenAI GPT-4o for code generation tasks (code suggestions only) - better at following structured instructions
# Valid models: gpt-4o, gpt-4-turbo, gpt-4o-mini (use gpt-4o-mini if gpt-4o requires verification)
CODE_SUGGESTION_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.chains import RetrievalQA


def get_org_vector_store(org_id: str) -> Chroma:
    """
    Returns a Chroma vector store bound to a specific org collection.
    """
    if embedding is None:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Cannot initialize vector store without embedding model."
        )
    collection_name = f"org_{org_id}"
    db = Chroma(
        collection_name=collection_name,
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embedding,
    )
    return db


def init_org_vector_store(org_id: str, csv_path: str = None) -> Chroma:
    """
    Initialize or update an org-specific collection.
    Optionally bootstrap with CSV data.
    """
    db = get_org_vector_store(org_id)

    if csv_path:  # bootstrap docs into the org collection
        loader = CSVLoader(csv_path)
        docs = loader.load()
        db.add_documents(docs)
        db.persist()
        print(f"Loaded {len(docs)} docs into collection for org {org_id}")

    return db


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(DATABASE_URL)


def _fetch_table_rows(
    table_name: str,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected_cols = ", ".join(columns) if columns else "*"
    query_parts: List[str] = [f"SELECT {selected_cols} FROM {table_name}"]
    if where_clause:
        query_parts.append(f"WHERE {where_clause}")
    if limit is not None and limit > 0:
        query_parts.append(f"LIMIT {limit}")
    query = " ".join(query_parts)

    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def _row_to_document(row: Dict[str, Any], metadata_fields: Optional[List[str]] = None) -> Document:
    metadata_fields = metadata_fields or []
    # Prefer a structured, readable page content; fallback to JSON
    if any(k in row for k in [
        "source_database", "source_schema", "source_table", "source_column",
        "target_database", "target_schema", "target_table", "target_column",
    ]):
        parts: List[str] = []
        for key in [
            "source_database", "source_schema", "source_table", "source_column",
            "target_database", "target_schema", "target_table", "target_column",
            "query_id", "query_type", "dbt_model_file_path", "dependency_score",
        ]:
            if key in row and row[key] is not None:
                parts.append(f"{key}: {row[key]}")
        page_content = "\n".join(parts) if parts else json.dumps(row, default=str)
    else:
        page_content = json.dumps(row, default=str)

    metadata: Dict[str, Any] = {}
    for mkey in metadata_fields:
        if mkey in row:
            val = row[mkey]
            # Chroma metadata must be primitives; coerce others to string
            if isinstance(val, (str, int, float, bool)) or val is None:
                metadata[mkey] = val
            else:
                try:
                    metadata[mkey] = val.isoformat()  # datetime-like
                except Exception:
                    metadata[mkey] = str(val)
    return Document(page_content=page_content, metadata=metadata)


def init_org_vector_store_from_table(
    org_id: str,
    table_name: str,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
    id_field: Optional[str] = None,
    metadata_fields: Optional[List[str]] = None,
) -> Chroma:
    """
    Initialize or update an org-specific collection using rows from a database table.
    If the collection exists, rows will be added (duplicates depend on Chroma ID handling).
    """
    db = get_org_vector_store(org_id)
    rows = _fetch_table_rows(table_name=table_name, columns=columns, where_clause=where_clause, limit=limit)
    if not rows:
        return db

    docs: List[Document] = [_row_to_document(r, metadata_fields=metadata_fields) for r in rows]
    ids: Optional[List[str]] = None
    if id_field:
        tmp_ids: List[str] = []
        for r in rows:
            val = r.get(id_field)
            if val is not None:
                tmp_ids.append(str(val))
            else:
                tmp_ids.append(None)  # type: ignore
        # Only set ids if all are present
        if all(x is not None for x in tmp_ids):
            ids = [str(x) for x in tmp_ids]  # type: ignore

    if ids:
        db.add_documents(docs, ids=ids)
    else:
        db.add_documents(docs)
    db.persist()
    return db


def upsert_org_vector_store_from_table(
    org_id: str,
    table_name: str,
    id_field: str,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
    metadata_fields: Optional[List[str]] = None,
) -> Chroma:
    """
    Upsert semantics using a stable ID field. Existing IDs will be replaced.
    """
    db = get_org_vector_store(org_id)
    rows = _fetch_table_rows(table_name=table_name, columns=columns, where_clause=where_clause, limit=limit)
    if not rows:
        return db

    docs: List[Document] = [_row_to_document(r, metadata_fields=metadata_fields) for r in rows]
    ids: List[str] = [str(r[id_field]) for r in rows if id_field in r and r[id_field] is not None]
    if not ids or len(ids) != len(docs):
        # Fallback to simple add if IDs are not fully present
        db.add_documents(docs)
    else:
        # Chroma supports upsert via add with the same IDs; duplicates get replaced in 0.4+
        db.add_documents(docs, ids=ids)
    db.persist()
    return db

def get_retriever(org_id: str, k: int = 8):
    db = get_org_vector_store(org_id)
    return db.as_retriever(search_kwargs={"k": k})


def get_qa_chain(org_id: str, k: int = 5, llm=None):
    """
    Get a QA chain for the given org_id.
    
    Args:
        org_id: Organization ID
        k: Number of documents to retrieve
        llm: Optional LLM to use. If None, defaults to LLM (gpt-4o-mini) for
             backward compatibility. For chatbot tools, pass CHAT_LLM (GPT-4o-mini).
    
    Returns:
        RetrievalQA chain
    """
    if llm is None:
        llm = LLM  # Default to gpt-4o-mini for backward compatibility
    if llm is None:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Cannot create QA chain without LLM."
        )
    
    retriever = get_retriever(org_id, k=k)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain
# # Initialize (bootstrap) a vector store for org_123 with CSV data
# DB = init_org_vector_store("123_org_1", LINEAGE_CSV_PATH)


# -----------------------------
# Column-level lineage embedding sync
# -----------------------------

def _format_timestamp_for_sql(ts: Any) -> Optional[str]:
    """
    Accepts a datetime or string and returns an ISO 8601 string suitable for SQL literal usage.
    Returns None if input is falsy.
    """
    if ts is None:
        return None
    try:
        # If object has isoformat (datetime), prefer that
        iso = ts.isoformat()  # type: ignore[attr-defined]
        return iso
    except Exception:
        # Assume string already
        return str(ts)


def _lineage_row_to_document(row: Dict[str, Any]) -> Document:
    """
    Specialized converter for column_level_lineage rows into a readable Document.
    Packs key lineage fields into page_content and keeps additional fields in metadata.
    """
    parts: List[str] = []
    for key in [
        "source_database",
        "source_schema",
        "source_table",
        "source_column",
        "target_database",
        "target_schema",
        "target_table",
        "target_column",
        "query_id",
        "query_type",
        "dbt_model_file_path",
        "dependency_score",
    ]:
        if key in row and row[key] is not None:
            parts.append(f"{key}: {row[key]}")
    page_content = "\n".join(parts) if parts else json.dumps(row, default=str)

    metadata: Dict[str, Any] = {}
    for mkey in [
        "org_id",
        "connection_id",
        "session_id",
        "created_at",
        "updated_at",
        "is_active",
        "dependency_score",
        "dbt_model_file_path",
    ]:
        if mkey in row and row[mkey] is not None:
            val = row[mkey]
            # Ensure metadata values are primitives (str/int/float/bool/None)
            if isinstance(val, (str, int, float, bool)):
                metadata[mkey] = val
            else:
                try:
                    metadata[mkey] = val.isoformat()  # datetime -> iso string
                except Exception:
                    metadata[mkey] = str(val)
    return Document(page_content=page_content, metadata=metadata)


def upsert_lineage_embeddings(
    org_id: str,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
) -> Chroma:
    """
    Upsert active lineage rows into the org's Chroma collection using stable `id`.
    The caller should pass a where_clause that filters by org_id and time window.
    Only rows with is_active = 1 are upserted.
    """
    base_filters = [f"org_id = '{org_id}'", "is_active = 1"]
    if where_clause and where_clause.strip():
        base_filters.append(f"({where_clause})")
    final_where = " AND ".join(base_filters)

    rows = _fetch_table_rows(
        table_name="column_level_lineage",
        columns=None,
        where_clause=final_where,
        limit=limit,
    )

    db = get_org_vector_store(org_id)
    if not rows:
        return db

    docs: List[Document] = [_lineage_row_to_document(r) for r in rows]
    ids: List[str] = [str(r["id"]) for r in rows if r.get("id") is not None]
    if ids and len(ids) == len(docs):
        db.add_documents(docs, ids=ids)
    else:
        db.add_documents(docs)
    db.persist()
    return db


def delete_inactive_lineage_embeddings(
    org_id: str,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
) -> Chroma:
    """
    Delete embeddings for lineage rows that are inactive (is_active = 0) matching filters.
    Deletion is based on the row `id` used as the Chroma document ID.
    """
    base_filters = [f"org_id = '{org_id}'", "is_active = 0"]
    if where_clause and where_clause.strip():
        base_filters.append(f"({where_clause})")
    final_where = " AND ".join(base_filters)

    rows = _fetch_table_rows(
        table_name="column_level_lineage",
        columns=["id"],
        where_clause=final_where,
        limit=limit,
    )

    db = get_org_vector_store(org_id)
    if not rows:
        return db

    ids: List[str] = [str(r["id"]) for r in rows if r.get("id") is not None]
    if ids:
        try:
            db.delete(ids=ids)
            db.persist()
        except Exception:
            # Best-effort delete; keep going to avoid breaking ingestion pipelines
            pass
    return db


def sync_lineage_embeddings_incremental(
    org_id: str,
    since: Optional[Any] = None,
    until: Optional[Any] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Perform an incremental sync for `column_level_lineage` into Chroma for a given org.
    - Upsert rows where is_active=1 and updated_at/created_at in [since, until]
    - Delete rows where is_active=0 and updated_at/created_at in [since, until]

    The `since`/`until` parameters can be datetime or ISO strings. If omitted, all rows match.
    """
    since_iso = _format_timestamp_for_sql(since)
    until_iso = _format_timestamp_for_sql(until)

    time_filters: List[str] = []
    if since_iso and until_iso:
        time_filters.append(
            f"((updated_at >= '{since_iso}' OR created_at >= '{since_iso}') AND (updated_at <= '{until_iso}' OR created_at <= '{until_iso}'))"
        )
    elif since_iso:
        time_filters.append(f"(updated_at >= '{since_iso}' OR created_at >= '{since_iso}')")
    elif until_iso:
        time_filters.append(f"(updated_at <= '{until_iso}' OR created_at <= '{until_iso}')")

    time_where = " AND ".join(time_filters) if time_filters else None

    # Upsert active rows in window
    upsert_lineage_embeddings(org_id=org_id, where_clause=time_where, limit=limit)

    # Delete inactive rows in window
    delete_inactive_lineage_embeddings(org_id=org_id, where_clause=time_where, limit=limit)


def upsert_lineage_embeddings_by_batch(
    org_id: str,
    batch_id: str,
    limit: Optional[int] = None,
) -> None:
    """
    Convenience method to sync rows for a specific batch_id. Performs upsert for active rows
    and deletion for inactive rows in the given batch.
    """
    where = f"batch_id = '{batch_id}'"
    upsert_lineage_embeddings(org_id=org_id, where_clause=where, limit=limit)
    delete_inactive_lineage_embeddings(org_id=org_id, where_clause=where, limit=limit)
