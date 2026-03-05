from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from datetime import datetime
import psycopg2
import psycopg2.extras
from psycopg2.extras import Json
import os
import json
import logging
import re

# Import embedding and LLM from vector_db to avoid duplication
from app.vector_db import embedding, LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "chroma_collection_setup")
LINEAGE_CSV_PATH = os.getenv("LINEAGE_CSV_PATH", "temp_lineage_data/lineage_output_deep.csv")
DATABASE_URL = os.getenv("DATABASE_URL")
 
# SQLAlchemy models for storing PR analysis in first-class tables
from sqlalchemy.orm import Session
from app.utils.models import GitHubPullRequestAnalysis, GitHubRepository, GitHubInstallation, DbtManifestNode


def get_model_metadata(db: Session, file_path: str, org_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve dbt model metadata from the dbt_manifest_node table by file path.
    
    Args:
        db: SQLAlchemy database session
        file_path: The original file path to search for
        org_id: Organization ID to filter by
        
    Returns:
        Dictionary containing the model metadata if found, None otherwise
    """
    try:
        # Validate input parameters
        if not file_path or not org_id:
            logger.warning(f"Invalid parameters: file_path={file_path}, org_id={org_id}")
            return None
        
        # Query the dbt_manifest_node table for the specific file path and org
        node = db.query(DbtManifestNode).filter(
            DbtManifestNode.org_id == org_id,
            DbtManifestNode.original_file_path == file_path
        ).first()
        
        if node:
            # Convert the SQLAlchemy model to a dictionary
            return {
                "id": str(node.id),
                "org_id": str(node.org_id),
                "connection_id": str(node.connection_id),
                "run_id": node.run_id,
                "unique_id": node.unique_id,
                "database": node.database,
                "schema": node.schema,
                "name": node.name,
                "package_name": node.package_name,
                "path": node.path,
                "original_file_path": node.original_file_path,
                "resource_type": node.resource_type,
                "raw_code": node.raw_code,
                "compiled_code": node.compiled_code,
                "downstream_models": node.downstream_models,
                "last_successful_run_at": node.last_successful_run_at.isoformat() if node.last_successful_run_at else None,
                "synced_at": node.synced_at.isoformat() if node.synced_at else None
            }
        return None
    except Exception as e:
        logger.error(f"Error retrieving model metadata for file_path {file_path}: {str(e)}")
        return None


# # We are assuming that we have created a vector sotre 
# def init_vector_store() -> Chroma:
#     if os.path.exists(VECTOR_STORE_DIR):
#         db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)
#         logger.info("Loaded existing Chroma vector store")
#     else:
#         loader = CSVLoader(LINEAGE_CSV_PATH)
#         docs = loader.load()
#         db = Chroma.from_documents(docs, embedding, persist_directory=VECTOR_STORE_DIR)
#         db.persist()
#         logger.info("Created and persisted new Chroma vector store")
#     return db




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


def get_retriever(org_id: str, k: int = 8):
    db = get_org_vector_store(org_id)
    return db.as_retriever(search_kwargs={"k": k})


def get_qa_chain(org_id: str, k: int = 5):
    if LLM is None:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Cannot create QA chain without LLM."
        )
    retriever = get_retriever(org_id, k=k)
    qa_chain = RetrievalQA.from_chain_type(
        llm=LLM,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


# Removed global, hardcoded retriever/qa_chain initialization.
# Use get_retriever(org_id) and get_qa_chain(org_id) dynamically per request.


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def fetch_queries(query_ids: List[str]) -> List[Dict]:
    if not query_ids:
        return []
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        placeholders = ",".join(["%s"] * len(query_ids))
        cursor.execute(
            f"""
            SELECT query_id, query_text FROM "snowflake_query_history"
            WHERE query_id IN ({placeholders})
        """,
            query_ids,
        )
        return cursor.fetchall()


def store_pr_analysis(
    db: Session,
    *,
    org_id: str,
    installation_id_str: str,
    repo_full_name: str,
    pr_number: int,
    pr_title: Optional[str],
    pr_description: Optional[str] = None,
    branch_name: Optional[str] = None,
    author_name: Optional[str] = None,
    pr_url: Optional[str] = None,
    total_impacted_queries: Optional[int] = None,
    analysis_data: Dict,
) -> str:
    """
    Persist PR analysis using SQLAlchemy model `GitHubPullRequestAnalysis` and link to
    related GitHub entities.

    Returns the created analysis UUID as string.
    """
    # Resolve installation row (by external installation id string)
    installation = (
        db.query(GitHubInstallation)
        .filter(GitHubInstallation.installation_id == installation_id_str)
        .first()
    )
    if not installation:
        raise ValueError("Installation not found for storing PR analysis")

    # Try to resolve repository row by full_name under this installation
    repository = (
        db.query(GitHubRepository)
        .filter(
            GitHubRepository.installation_id == installation.id,
            GitHubRepository.full_name == repo_full_name,
        )
        .first()
    )

    # If total_impacted_queries not provided, calculate from analysis_data
    if total_impacted_queries is None:
        all_query_ids = set()
        files = analysis_data.get("files", [])
        for file_data in files:
            affected_ids = file_data.get("affected_query_ids", [])
            all_query_ids.update(affected_ids)
        total_impacted_queries = len(all_query_ids)

    analysis = GitHubPullRequestAnalysis(
        org_id=installation.org_id,
        installation_id=installation.id,
        repository_id=repository.id if repository else None,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        pr_title=pr_title,
        pr_description=pr_description,
        branch_name=branch_name,
        author_name=author_name,
        pr_url=pr_url,
        total_impacted_queries=total_impacted_queries,
        analysis_data=analysis_data,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return str(analysis.id)


# -----------------------------
# Prompts for Schema changes
# -----------------------------
PLAN_PROMPT = """
You are a metadata-aware lineage analyst. You will receive a SQL/schema change and a set of context snippets from a lineage knowledge base (CSV rows embedded in a vector store). 

CRITICAL: You MUST be extremely selective and ONLY identify entities that have an EXPLICIT, DIRECT dependency on the specific column being changed in the SQL statement.

Your task:
1) Parse the SQL change to identify the EXACT column being modified (e.g., if "DROP COLUMN scenariocd", the column is "scenariocd").
2) From the snippets, find ONLY target columns that have an EXPLICIT source-to-target relationship with this specific column.
3) DO NOT include columns just because they're in the same table as a dependent column.
4) **AGGRESSIVELY** suggest next queries to expand downstream dependencies (multi-hop) to find ALL impacted assets at ALL depth levels (depth 1, 2, 3, 4, 5+).
5) Return a STRICT JSON as text with the following structure:
       "found_entities": ["database.schema.table.column", ...], 
       "next_queries": ["..."], 
       "notes": "..."

CRITICAL RULES:
- ONLY include columns that are explicitly shown in snippets to depend on the changed column (or on previously found columns)
- If the change is "DROP COLUMN scenariocd", ONLY include columns where snippets show: source_column=scenariocd → target_column
- DO NOT include other columns from the same target table unless they also explicitly depend on the changed column
- For example, if snippets show "scenariocd → allocationtype", ONLY include "allocationtype", NOT "accountfrom", "accountto", etc.
- **IMPORTANT FOR NEXT_QUERIES**: You MUST be aggressive in finding downstream dependencies. For EACH found entity, generate a query to find its downstream dependencies. Continue this pattern to find depth 3, 4, 5+ dependencies.
- **DO NOT stop early**: Even if you think you've found enough, continue generating queries for entities that might have further downstream dependencies.
- Only return an empty list for "next_queries" if you are absolutely certain there are NO more downstream dependencies at ANY depth level.
- Prefer exact entity strings from the snippets for next queries (e.g., "bronze.accounts.allocationrule.allocationtype").
- For each found entity, create a query like: "What are the downstream dependencies of database.schema.table.column?"

SQL/Schema Change:
{change}

Context Snippets (may be empty):
{snippets}

Respond only with JSON text, do not include any extra explanation.
"""

FINAL_REPORT_PROMPT = """
You are a metadata-aware assistant tasked with generating a **complete multi-hop downstream impact analysis report**.

CRITICAL: You MUST be extremely strict and ONLY report columns that have an EXPLICIT, DIRECT dependency on the specific column being changed. DO NOT include columns just because they're in the same table as a dependent column.

You are given:
1. The SQL/schema change (parse it to identify the EXACT column being modified)
2. ALL retrieved lineage context (multi-hop) as raw snippets

VALIDATION FIRST:
- Parse the SQL change to extract the exact column name being changed (e.g., if "DROP COLUMN scenariocd", the column is "scenariocd")
- Check snippets for EXPLICIT source_column → target_column relationships involving this specific column
- If snippets show a relationship like "source_column: scenariocd → target_column: allocationtype", ONLY include "allocationtype"
- DO NOT include other columns from the same target table (e.g., "accountfrom", "accountto") unless snippets explicitly show they also depend on the changed column

Your task:
- Build a dependency graph from the snippets, identifying the EXACT dependency chain
- Traverse the lineage graph recursively (level by level) using the snippets
- Include ONLY downstream columns that have EXPLICIT dependencies on the changed column (or on depth 1 columns, etc.)
- Group impacts by **depth level** (1-hop, 2-hop, 3-hop, etc.) with STRICT depth classification
- Collect impacted query IDs and metadata

CRITICAL RULES FOR DEPTH CLASSIFICATION (READ CAREFULLY):

**STEP 1: Identify the SOURCE column being changed**
- The SOURCE column is: {source_column}
- The SOURCE table is: {source_table}
- This is your ROOT/SOURCE: {source_table}.{source_column}

**STEP 2: Build Depth 1 (Direct Dependencies)**
- Look through ALL snippets for relationships where: source_table = {source_table} AND source_column = {source_column}
- ONLY include target columns where the snippet shows: {source_table}.{source_column} → target_table.target_column
- These are Depth 1 columns
- CRITICAL: If a snippet shows source_table = {source_table} and source_column = {source_column}, then the target is Depth 1
- DO NOT include columns that depend on other columns - only those that directly depend on {source_table}.{source_column}
- Example: If snippet shows "source_table: financial_activity_summary, source_column: accounting_period → target_table: gl_period_balances, target_column: accounting_period", then gl_period_balances.accounting_period is Depth 1

**STEP 3: Build Depth 2 (Indirect Dependencies)**
- First, identify all Depth 1 columns from Step 2
- Then look through ALL snippets for relationships where the SOURCE is a Depth 1 column (NOT the original source)
- ONLY include target columns where:
  - The snippet shows: depth1_table.depth1_column → target_table.target_column
  - AND the snippet does NOT show: {source_table}.{source_column} → target_table.target_column (if it does, it's Depth 1, not Depth 2)
- CRITICAL: A column is Depth 2 ONLY if:
  1. It depends on a Depth 1 column (as shown in snippets: depth1_table.depth1_column → this target)
  2. It does NOT directly depend on {source_table}.{source_column} (no snippet shows {source_table}.{source_column} → this target)
- Example: If snippets show:
  - "source_table: financial_activity_summary, source_column: accounting_period → target_table: gl_period_balances, target_column: accounting_period" (Depth 1)
  - "source_table: gl_period_balances, source_column: accounting_period → target_table: asset_period_financial_profile, target_column: accounting_period" (Depth 2)
  - But NO snippet shows: "source_table: financial_activity_summary, source_column: accounting_period → target_table: asset_period_financial_profile, target_column: accounting_period"
  Then asset_period_financial_profile.accounting_period is Depth 2, NOT Depth 1

**STEP 4: Build Depth 3+ (Continue the pattern)**
- Depth 3: Columns that depend on Depth 2 columns AND NOT on Depth 1 or source
- Continue this pattern for deeper levels

**VALIDATION CHECKLIST FOR EACH COLUMN:**
Before placing a column in a depth level, verify:
1. For Depth 1: Does a snippet show source_table={source_table} AND source_column={source_column} → this target? YES = Depth 1
2. For Depth 2: Does a snippet show a Depth 1 column → this target? YES, and NO snippet shows {source_table}.{source_column} → this target? = Depth 2
3. For Depth 3+: Does a snippet show a previous depth column → this target? YES, and no earlier depth dependency? = correct depth

**EXAMPLES:**

Example 1 - Correct Depth Classification:
- Source: {source_table}.{source_column} (e.g., retail.raw.financial_activity_summary.accounting_period)
- Snippet 1: "source_table: financial_activity_summary, source_column: accounting_period → target_table: gl_period_balances, target_column: accounting_period"
  → gl_period_balances.accounting_period is Depth 1 ✓ (directly depends on source)
- Snippet 2: "source_table: gl_period_balances, source_column: accounting_period → target_table: asset_period_financial_profile, target_column: accounting_period"
  → asset_period_financial_profile.accounting_period is Depth 2 ✓ (depends on Depth 1, NOT directly on source)
- Snippet 3: NO snippet shows "source_table: financial_activity_summary, source_column: accounting_period → target_table: asset_period_financial_profile"
  → Confirms it's Depth 2, not Depth 1 ✓

Example 2 - Wrong Classification (DO NOT DO THIS):
- Source: {source_table}.{source_column}
- Snippet 1: "source_table: financial_activity_summary, source_column: accounting_period → target_table: gl_period_balances, target_column: accounting_period"
- Snippet 2: "source_table: gl_period_balances, source_column: accounting_period → target_table: asset_period_financial_profile, target_column: accounting_period"
- WRONG: Classifying asset_period_financial_profile.accounting_period as Depth 1 ✗
- CORRECT: It's Depth 2 because it depends on gl_period_balances (Depth 1), not directly on financial_activity_summary

CRITICAL RULES FOR INCLUDING COLUMNS:
1. Depth 1: ONLY columns where snippets show: source_table={source_table} AND source_column={source_column} → target_table.target_column
2. Depth 2: ONLY columns where snippets show: depth1_table.depth1_column → target_table.target_column AND NO snippet shows source_table={source_table} AND source_column={source_column} → target_table.target_column
3. DO NOT include all columns from a table just because one column in that table has a dependency
4. For example, if "scenariocd → allocationtype" is found, ONLY list "allocationtype" in source_metadata, NOT other columns from allocationrule table

---

Your output must be valid JSON with these keys:
{{
  "impact_report": "<the full Markdown report, format attached below>",
  "affected_query_ids": ["q1", "q2", ...],
  "source_metadata": [
    {{
      "target_database": "...",
      "target_schema": "...",
      "target_table": "...",
      "target_column": "..."
    }}
  ]
}}

---

### 📑 Impact Report Markdown format

📌 **Change Summary**

Change summary description: _(Explain why this change may have an impact downstream based on source-to-target dependencies. Be specific about which columns are impacted and why.)_

| Field                  | Description |
|------------------------|-------------|
| Change Type            | e.g., Add/Drop/Alter Column |
| Affected Table         | <database.schema.table> |
| Affected Column(s)     | <column name(s)> |
| Requested Change       | <exact change or best-effort> |
| Reason for Change      | <reason if known, else N/A> |

---

### **Downstream Impact Analysis**

_List ONLY impacted downstream targets that have EXPLICIT dependencies on the changed column, grouped by depth._

#### Depth 1 (direct dependencies - ONLY columns that explicitly depend on the changed column):
1. **Target Database:** ...
   **Target Schema:** ...
   **Target Table:** ...
   **Target Column:** ...
   **Explanation:** _(Explain how this SPECIFIC column depends on the changed column, referencing explicit relationships in snippets)_

#### Depth 2 (ONLY columns that explicitly depend on depth 1 columns):
1. **Target Database:** ...
   **Target Schema:** ...
   **Target Table:** ...
   **Target Column:** ...
   **Explanation:** _(Explain how this SPECIFIC column depends on a specific depth 1 column)_

#### Depth 3 (ONLY columns that explicitly depend on depth 2 columns):
1. **Target Database:** ...
   **Target Schema:** ...
   **Target Table:** ...
   **Target Column:** ...
   **Explanation:** _(Explain how this SPECIFIC column depends on a specific depth 2 column)_

#### Depth 4 (ONLY columns that explicitly depend on depth 3 columns):
1. **Target Database:** ...
   **Target Schema:** ...
   **Target Table:** ...
   **Target Column:** ...
   **Explanation:** _(Explain how this SPECIFIC column depends on a specific depth 3 column)_

(Continue for depth 5, 6, 7, etc. as needed. DO NOT stop at depth 2 - continue analyzing ALL depth levels until you've covered ALL dependencies found in the snippets. The goal is to find 10-20 impacted assets across ALL depth levels, not just depth 1 and 2.)

---

**Explanation:**
- Describe clearly how the change propagates through ALL levels (up to the deepest retrieved).
- For EACH listed column, explain the EXPLICIT dependency chain from the changed column.
- Mention necessary updates (views, ETL, dashboards, schema enforcement, SELECT * risks, etc.).
- If a table has multiple columns but only one depends on the changed column, ONLY mention that one column.

---

Details:
- `affected_query_ids`: Collect all query IDs seen in snippets for ONLY the impacted columns (not all columns from impacted tables).
- `source_metadata`: Extract structured metadata for ONLY columns that have explicit dependencies at any depth.
- Ensure the JSON is strictly valid (no trailing commas, no comments, no markdown around it).
- Do not hallucinate. If a depth has no entries, omit it.
- Be SELECTIVE - only include columns with clear, explicit dependency relationships shown in snippets.

---

SQL/Schema Change:
{change}

ALL Retrieved Snippets:
{snippets}
"""


# -----------------------------
# Prompts for DBT model changes
# -----------------------------
dbt_PLAN_PROMPT = """
You are a metadata-aware lineage analyst. You will receive a json data, which explains change in dbt model logic and their impacted columns in target. 
sample/example json data :
{{'impact': [{{'change': "Modified the allocationruledesc column to append '_UPDATED' when allocationrulekey is 1, otherwise keep the original value.",
   'explanation': 'This change introduces a conditional update to the allocationruledesc column based on the allocationrulekey. This is important because it alters the data content of this column based on a specific condition, which might affect downstream processes or reports that rely on the original value or a specific pattern in this column.',
   'impacted_columns': ['PROD_TZ.EDW_STAGING_EDW.DIMSHAREDSERVICESALLOCATIONRULE.allocationruledesc']}},
  {{'change': "Modified the payrollbasis column to be 'NEW_PAYROLL_BASIS' when allocationrulekey is 1, otherwise keep the original value.",
   'explanation': 'This change introduces a conditional update to the payrollbasis column based on the allocationrulekey. This means payrollbasis will have a new value for allocationrulekey = 1 and this change might impact any process that depends on the original value of payrollbasis',
   'impacted_columns': ['PROD_TZ.EDW_STAGING_EDW.DIMSHAREDSERVICESALLOCATIONRULE.payrollbasis']}}]}}

Your task:
1) Go through impacted_columns fields in json, Infer the impacted entities (tables/columns).
2) Suggest next queries to expand downstream dependencies (multi-hop), if any.
3) Return a STRICT JSON as text with the following structure:
       "found_entities": ["schema.table.column", ...], 
       "next_queries": ["..."], 
       "notes": "..."

- If no further queries are useful, return an empty list for "next_queries".
- Prefer exact entity strings from the snippets for next queries (e.g., "edw_staging.allocationrule.departmentlist").

json data:
{safe_json_text}


Respond only with JSON text, do not include any extra explanation.
"""


dbt_FINAL_REPORT_PROMPT = """
You are a metadata-aware assistant tasked with generating a **complete multi-hop downstream impact analysis report**.


You are given:
1. A json data, which explains change in dbt model logic and their impacted columns in target.
2. Model metadata : which gives the information about which table is materilized and any downstream models are there.
2. ALL retrieved lineage context (multi-hop) as raw snippets based on impacted target columns received in json data.

--
json data:
{safe_json_text}

ALL Retrieved Snippets:
{snippets}

--

Your task:
- Traverse the lineage graph recursively (level by level) using the snippets.
- Include ALL downstream assets until no further dependencies remain (not just direct neighbors).
- Group impacts by **depth level** (1-hop, 2-hop, 3-hop, etc.).
- Collect impacted query IDs and metadata.
- For each change, explicitly trace: **code change → affected column → downstream propagation → explanation**.

---

Your output must be valid JSON with these keys:
{{
  "impact_report": "<the full Markdown report, format attached below>",
  "affected_query_ids": ["q1", "q2", ...],
  "source_metadata": [
    {{
      "target_database": "...",
      "target_schema": "...",
      "target_table": "...",
      "target_column": "..."
    }}
  ]
}}

---

### 📑 Impact Report Markdown format

📌 **Change Summary**

Change summary description: _(Explain why this transformation may impact downstream based on source-to-target dependencies.)_

| Field                  | Description |
|------------------------|-------------|
| Change Type            | e.g., Transformation Logic Update |
| Affected Table         | <database.schema.table> |
| Affected Column(s)     | <column name(s)> |
| Requested Change       | <exact code modification> |
| Reason for Change      | <reason if known, else N/A> |

---

### **Change-to-Impact Mapping**

For **each detected change**, show the chain of impact:

#### 🔄 Change 1
```sql
Old: 1 - ratio  
New: cast(0.95 * (ratio) as decimal(7,6))
```
Impacted Column: corporateservices (in STAGING.legalallocation)
Impact Chain for target table:

allocationtype.column_name → allocationrule_upsert.column_name → target.column_name

target to downstream chain: 
target.col_name → downstream_table.column_name

Impacted Downstream DBT Model : 

Explanation:


#### 🔄 Change 2
```sql
Old: 1 - ratio  
New: cast(0.95 * (ratio) as decimal(7,6))
```
Impacted Column: corporateservices (in STAGING.legalallocation)
Impact Chain for target table:

allocationtype.column_name → allocationrule_upsert.column_name → target.column_name

target to downstream chain: 
target.col_name → downstream_table.column_name

Explanation:

### **Downstream Impact Analysis (Grouped by Depth)** ###
Depth 1 (direct dependencies):

Target Table: STAGING.legalallocation
Impacted Columns: corporateservices, allocationtype, payrollbasis
explanation: 

Depth 2:

Target Table: STAGING.allocationrule_upsert
Impacted Columns: corporateservices, allocationtype, payrollbasis

Depth 3:

Target Table: STAGING.finalsource
Impacted Columns: corporateservices, allocationtype, payrollbasis
"""



def _docs_to_snippets(docs: List[Document]) -> str:
    return "\n".join(d.page_content.strip() for d in docs)


def _safe_json_parse(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(txt[start : end + 1])
            except Exception:
                pass
        return {}


def parse_schema_change(change_text: str) -> Dict[str, Any]:
    """
    Parse SQL schema change to extract table and column information.
    Handles formats like:
    - ALTER TABLE bronze.accounts.gl_summary DROP COLUMN scenariocd;
    - ALTER TABLE gl_summary DROP COLUMN scenariocd;
    - ALTER TABLE schema.table ADD COLUMN newcol;
    """
    result = {
        "change_type": None,
        "table_full_path": None,
        "table_name": None,
        "schema": None,
        "database": None,
        "column_name": None,
        "original_change": change_text
    }
    
    change_upper = change_text.upper().strip()
    
    # Extract change type
    if "DROP COLUMN" in change_upper:
        result["change_type"] = "DROP_COLUMN"
        # Extract column name
        match = re.search(r"DROP\s+COLUMN\s+(\w+)", change_upper, re.IGNORECASE)
        if match:
            result["column_name"] = match.group(1)
    elif "ADD COLUMN" in change_upper:
        result["change_type"] = "ADD_COLUMN"
        match = re.search(r"ADD\s+COLUMN\s+(\w+)", change_upper, re.IGNORECASE)
        if match:
            result["column_name"] = match.group(1)
    elif "ALTER COLUMN" in change_upper or "MODIFY COLUMN" in change_upper:
        result["change_type"] = "ALTER_COLUMN"
        match = re.search(r"(?:ALTER|MODIFY)\s+COLUMN\s+(\w+)", change_upper, re.IGNORECASE)
        if match:
            result["column_name"] = match.group(1)
    
    # Extract table information
    match = re.search(r"ALTER\s+TABLE\s+([^\s;]+)", change_upper, re.IGNORECASE)
    if match:
        table_path = match.group(1).strip()
        result["table_full_path"] = table_path
        
        # Parse database.schema.table or schema.table or just table
        parts = [p.strip() for p in table_path.split(".")]
        if len(parts) == 3:
            result["database"] = parts[0]
            result["schema"] = parts[1]
            result["table_name"] = parts[2]
        elif len(parts) == 2:
            result["schema"] = parts[0]
            result["table_name"] = parts[1]
        elif len(parts) == 1:
            result["table_name"] = parts[0]
    
    return result


def validate_impact_metadata(source_metadata: List[Dict[str, Any]], parsed_change: Dict[str, Any], snippets: str) -> List[Dict[str, Any]]:
    """
    Validate source_metadata and log warnings for potential false positives.
    The main filtering should be done by the LLM via improved prompts, but this provides
    additional validation and logging for debugging.
    """
    if not source_metadata or not parsed_change.get("column_name"):
        return source_metadata
    
    changed_column = parsed_change["column_name"].lower()
    snippet_lower = snippets.lower() if snippets else ""
    
    # Log information about what was found
    logger.info(
        f"Validating {len(source_metadata)} metadata entries for changed column: {changed_column}"
    )
    
    # For now, trust the LLM's filtering (via improved prompts) and just log
    # In the future, we could add more sophisticated validation if needed
    for entry in source_metadata:
        target_column = entry.get("target_column", "").lower()
        target_table = entry.get("target_table", "").lower()
        
        # Check if both columns are mentioned in snippets (heuristic check)
        both_mentioned = changed_column in snippet_lower and target_column in snippet_lower
        
        if not both_mentioned:
            logger.debug(
                f"Potential validation concern: {target_table}.{target_column} - "
                f"changed column '{changed_column}' or target column may not be explicitly "
                f"mentioned together in snippets (relying on LLM filtering)"
            )
    
    # Return all entries - the improved prompts should have done the filtering
    # This function is mainly for logging and future enhancement
    return source_metadata


class IterativeConfig(BaseModel):
    max_iters: int = 10  # Increased default to allow deeper traversal
    k_per_query: int = 8
    dedupe: bool = True
    max_assets: int = 20  # Target number of assets to find before stopping
    min_assets: int = 10  # Minimum number of assets before considering stopping


def schema_detection_rag(change_text: str, org_id: str, cfg: Optional[IterativeConfig] = None):
    cfg = cfg or IterativeConfig()
    
    # Parse the SQL change to extract column and table information
    parsed_change = parse_schema_change(change_text)
    
    # Create more precise initial query based on parsed information
    initial_queries = [change_text]
    if parsed_change.get("column_name") and parsed_change.get("table_name"):
        # Create a more specific query targeting the exact column
        table_path = parsed_change.get("table_full_path") or parsed_change.get("table_name")
        column_name = parsed_change["column_name"]
        specific_query = f"what are the downstream columns for {table_path}.{column_name}"
        initial_queries.append(specific_query)
        logger.info(f"Using parsed change info: table={table_path}, column={column_name}")
    
    used_queries: List[str] = []
    collected_docs: List[Document] = []
    seen_texts: Set[str] = set()
    seen_entities: Set[str] = set()  # Track unique entities found
    frontier: List[str] = initial_queries
    retriever = get_retriever(org_id, k=cfg.k_per_query)

    iteration = 0
    while iteration < cfg.max_iters:
        new_docs: List[Document] = []
        for q in frontier:
            if q in used_queries:
                continue
            used_queries.append(q)
            docs = retriever.get_relevant_documents(q)
            try:
                logger.info(
                    "lineage.retrieval - org=%s iteration=%d query_len=%d docs=%d",
                    org_id,
                    iteration + 1,
                    len(q or ""),
                    len(docs or []),
                )
                for idx, d in enumerate(docs[:3]):
                    preview = (d.page_content or "").strip().replace("\n", " ")[:300]
                    logger.info("lineage.retrieval.preview[%d]: %s", idx, preview)
            except Exception:
                logger.debug("lineage.retrieval - logging failed for query preview")
            for d in docs:
                if not cfg.dedupe or d.page_content not in seen_texts:
                    seen_texts.add(d.page_content)
                    new_docs.append(d)
                    collected_docs.append(d)
        
        if not new_docs:
            logger.info("No new documents found at iteration %d, stopping", iteration + 1)
            break
        
        # Count unique entities found so far (approximate by counting unique target columns)
        # This is a heuristic - we count unique target_table.target_column combinations
        # Extract entities from all collected docs (not just new ones) to get accurate count
        for doc in collected_docs:
            content = doc.page_content
            # Try to extract target entities from document content
            # Look for patterns like "target_table: X" and "target_column: Y"
            # Handle both single-line and multi-line patterns
            lines = content.split('\n')
            current_table = None
            current_column = None
            current_db = None
            current_schema = None
            
            for line in lines:
                line_lower = line.lower().strip()
                if 'target_database:' in line_lower:
                    db_match = re.search(r'target_database:\s*(\S+)', line, re.IGNORECASE)
                    if db_match:
                        current_db = db_match.group(1).strip()
                if 'target_schema:' in line_lower:
                    schema_match = re.search(r'target_schema:\s*(\S+)', line, re.IGNORECASE)
                    if schema_match:
                        current_schema = schema_match.group(1).strip()
                if 'target_table:' in line_lower:
                    table_match = re.search(r'target_table:\s*(\S+)', line, re.IGNORECASE)
                    if table_match:
                        current_table = table_match.group(1).strip()
                if 'target_column:' in line_lower:
                    col_match = re.search(r'target_column:\s*(\S+)', line, re.IGNORECASE)
                    if col_match:
                        current_column = col_match.group(1).strip()
                        # Build entity key
                        if current_table and current_column:
                            if current_db and current_schema:
                                entity_key = f"{current_db}.{current_schema}.{current_table}.{current_column}".lower()
                            elif current_schema:
                                entity_key = f"{current_schema}.{current_table}.{current_column}".lower()
                            else:
                                entity_key = f"{current_table}.{current_column}".lower()
                            seen_entities.add(entity_key)
                            # Reset for next entity
                            current_table = None
                            current_column = None
        
        current_asset_count = len(seen_entities)
        logger.info(
            "Iteration %d: Found %d unique assets so far (target: %d-%d)",
            iteration + 1,
            current_asset_count,
            cfg.min_assets,
            cfg.max_assets,
        )
        
        # Check if we've found enough assets
        if current_asset_count >= cfg.max_assets:
            logger.info(
                "Reached max_assets limit (%d), stopping retrieval", cfg.max_assets
            )
            break
        
        snippets_text = _docs_to_snippets(new_docs)
        plan = LLM.invoke(PLAN_PROMPT.format(change=change_text, snippets=snippets_text))
        plan_text = plan.content if hasattr(plan, "content") else str(plan)
        plan_json = _safe_json_parse(plan_text)
        found_entities = plan_json.get("found_entities", []) or []
        next_queries: List[str] = plan_json.get("next_queries", []) or []
        
        # Generate queries for found entities that haven't been queried yet
        for entity in found_entities:
            entity_lower = entity.lower().strip()
            if entity_lower and entity_lower not in seen_entities:
                # Generate a query for this entity's downstream dependencies
                query = f"What are the downstream dependencies of {entity}?"
                if query not in used_queries:
                    next_queries.append(query)
        
        # Also add any explicitly suggested queries
        for q in next_queries:
            if q and q not in used_queries:
                # Check if this query is about an entity we haven't explored
                entity_match = re.search(
                    r"downstream.*?of\s+([^\?]+)", q, re.IGNORECASE
                )
                if entity_match:
                    entity = entity_match.group(1).strip()
                    if entity.lower() not in seen_entities:
                        # This is a new entity to explore
                        pass
        
        frontier = [q for q in next_queries if q and q not in used_queries]
        
        if not frontier:
            # Only stop if we have minimum assets OR no more queries
            if current_asset_count >= cfg.min_assets:
                logger.info(
                    "No more queries and reached min_assets (%d), stopping",
                    cfg.min_assets,
                )
                break
            else:
                logger.info(
                    "No more queries but only found %d assets (min: %d), continuing...",
                    current_asset_count,
                    cfg.min_assets,
                )
                # Try to generate more queries from collected docs
                if collected_docs:
                    # Extract more entities from collected docs to query
                    for doc in collected_docs[-20:]:  # Check last 20 docs
                        content = doc.page_content
                        lines = content.split('\n')
                        current_table = None
                        current_column = None
                        current_db = None
                        current_schema = None
                        
                        for line in lines:
                            line_lower = line.lower().strip()
                            if 'target_database:' in line_lower:
                                db_match = re.search(r'target_database:\s*(\S+)', line, re.IGNORECASE)
                                if db_match:
                                    current_db = db_match.group(1).strip()
                            if 'target_schema:' in line_lower:
                                schema_match = re.search(r'target_schema:\s*(\S+)', line, re.IGNORECASE)
                                if schema_match:
                                    current_schema = schema_match.group(1).strip()
                            if 'target_table:' in line_lower:
                                table_match = re.search(r'target_table:\s*(\S+)', line, re.IGNORECASE)
                                if table_match:
                                    current_table = table_match.group(1).strip()
                            if 'target_column:' in line_lower:
                                col_match = re.search(r'target_column:\s*(\S+)', line, re.IGNORECASE)
                                if col_match:
                                    current_column = col_match.group(1).strip()
                                    # Build entity key and query
                                    if current_table and current_column:
                                        if current_db and current_schema:
                                            entity_key = f"{current_db}.{current_schema}.{current_table}.{current_column}".lower()
                                            entity_str = f"{current_db}.{current_schema}.{current_table}.{current_column}"
                                        elif current_schema:
                                            entity_key = f"{current_schema}.{current_table}.{current_column}".lower()
                                            entity_str = f"{current_schema}.{current_table}.{current_column}"
                                        else:
                                            entity_key = f"{current_table}.{current_column}".lower()
                                            entity_str = f"{current_table}.{current_column}"
                                        
                                        if entity_key not in seen_entities:
                                            query = f"What are the downstream dependencies of {entity_str}?"
                                            if query not in used_queries:
                                                frontier.append(query)
                                                logger.info(
                                                    "Generated additional query from collected docs: %s",
                                                    query[:100]
                                                )
                                                break
                                    # Reset for next entity
                                    current_table = None
                                    current_column = None
                        if frontier:
                            break
        
        iteration += 1

    all_snippets_text = _docs_to_snippets(collected_docs) if collected_docs else "(no snippets retrieved)"
    
    # Include parsed change info in the final prompt for better context
    source_table = parsed_change.get('table_full_path', 'UNKNOWN')
    source_column = parsed_change.get('column_name', 'UNKNOWN')
    change_context = (
        f"{change_text}\n\nParsed Information:\n"
        f"- Change Type: {parsed_change.get('change_type', 'UNKNOWN')}\n"
        f"- Source Table: {source_table}\n"
        f"- Source Column: {source_column}"
    )
    
    # Format prompt with explicit source information for depth classification
    formatted_prompt = FINAL_REPORT_PROMPT.format(
        change=change_context,
        snippets=all_snippets_text,
        source_table=source_table,
        source_column=source_column
    )
    
    final = LLM.invoke(formatted_prompt)
    final_text = final.content if hasattr(final, "content") else str(final)
    final_json_response = _safe_json_parse(final_text)
    
    # Validate and filter source_metadata to remove false positives
    if "source_metadata" in final_json_response:
        final_json_response["source_metadata"] = validate_impact_metadata(
            final_json_response["source_metadata"],
            parsed_change,
            all_snippets_text
        )
    
    return final_json_response


def dbt_model_detection_rag(code_changes: str, file_path: str, org_id: str, db: Session, cfg: Optional[IterativeConfig] = None):
    def get_dbt_impact(code_changes_inner: str, file_path_inner: str):
        model_metadata = get_model_metadata(db, file_path_inner, org_id)
        query = f"""You are DBT model specialised anlayser. You are provided with these things:
        1) Change in dbt model, basically diff text like this (below is just example of how we get the diff text)
        Example :
        File: models/EDW/DIMSHAREDSERVICESALLOCATIONRULE.sql (modified) [+10/-2]
        @@ -8,7 +8,11 @@ with source_allocationrule as (
        ...
        2) Lineage information (source columns and its target columns with DBT file path information, query id informaiton) is provided through vector store embedded
        3) compiled sql code through dbt manifest.json file, which gives you broader context about complete model.
        4) Dbt Model metadata like which database, schema and table it materilize.
        Your tasks :
        1. so based on above three points which is additional context to you, you need to analyse which are columns that will be impacted in target table mapping change that we are reciving.
        2. once analysed you need to give output in striclty in stuctured json format like this. no other text, only json.
        {{
        "impact": [
            {{
            "change": "describe the specific change in logic from the diff",
            "explanation": "explain what this change does and why it matters",
            "impacted_columns": [
                "database.schema.table.column2"
            ]
            }}
        ]
        }}
        --------------------------- information you recieved -------------------
        dbt model change {code_changes_inner}
        dbt model metadata {model_metadata}
        """
        qa_chain_local = get_qa_chain(org_id, k=cfg.k_per_query) if cfg else get_qa_chain(org_id)
        result = qa_chain_local.invoke({"query": query})
        safe_json = _safe_json_parse(result["result"]) if isinstance(result, dict) and "result" in result else {}
        impacted_columns = [col for impact in safe_json.get("impact", []) for col in impact.get("impacted_columns", [])]
        return {"safe_json": safe_json, "impacted_columns": impacted_columns}

    intermediate_resultset = get_dbt_impact(code_changes, file_path)
    impact_json_data = intermediate_resultset["safe_json"]
    impact_columns = intermediate_resultset["impacted_columns"]

    cfg = cfg or IterativeConfig()
    used_queries: List[str] = []
    collected_docs: List[Document] = []
    seen_texts: Set[str] = set()
    frontier: List[str] = impact_columns
    retriever = get_retriever(org_id, k=cfg.k_per_query)

    for _ in range(cfg.max_iters):
        new_docs: List[Document] = []
        for q in frontier:
            used_queries.append(q)
            docs = retriever.get_relevant_documents(q)
            try:
                logger.info(
                    "dbt.retrieval - org=%s query_len=%d docs=%d",
                    org_id,
                    len(q or ""),
                    len(docs or []),
                )
                for idx, d in enumerate(docs[:3]):
                    preview = (d.page_content or "").strip().replace("\n", " ")[:300]
                    logger.info("dbt.retrieval.preview[%d]: %s", idx, preview)
            except Exception:
                logger.debug("dbt.retrieval - logging failed for query preview")
            for d in docs:
                if not cfg.dedupe or d.page_content not in seen_texts:
                    seen_texts.add(d.page_content)
                    new_docs.append(d)
                    collected_docs.append(d)
        if not new_docs:
            break
        snippets_text = _docs_to_snippets(new_docs)
        plan = LLM.invoke(dbt_PLAN_PROMPT.format(safe_json_text=impact_json_data, snippets=snippets_text))
        plan_text = plan.content if hasattr(plan, "content") else str(plan)
        plan_json = _safe_json_parse(plan_text)
        next_queries: List[str] = plan_json.get("next_queries", []) or []
        frontier = [q for q in next_queries if q and q not in used_queries]
        if not frontier:
            break

    all_snippets_text = _docs_to_snippets(collected_docs) if collected_docs else "(no snippets retrieved)"
    final = LLM.invoke(dbt_FINAL_REPORT_PROMPT.format(safe_json_text=impact_json_data, snippets=all_snippets_text))
    final_text = final.content if hasattr(final, "content") else str(final)
    final_json_response = _safe_json_parse(final_text)
    return final_json_response


