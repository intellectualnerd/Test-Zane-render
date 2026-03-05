from typing import List, Dict, Any, Optional, Tuple
from langchain.agents import Tool
from app.tools.pr_repo import fetch_pr_analyses_for_org, fetch_pr_files_with_contents
from app.vector_db import CHAT_LLM, CODE_SUGGESTION_LLM
from app.services.impact_analysis import fetch_queries, schema_detection_rag, IterativeConfig
import logging
import json
import re

logger = logging.getLogger(__name__)


def analyze_impact_and_suggest_code(
    org_id: str,
    repo_full_name: str,
    pr_number: int,
) -> Dict[str, Any]:
    """
    Analyze PR impact report and suggest code changes to mitigate impacts.
    
    Steps:
    1. Fetch PR analysis (impact report, affected queries, etc.)
    2. Fetch PR file contents
    3. Use LLM to analyze impact types and generate code suggestions
    """
    logger.info(f"Analyzing impact and suggesting code for PR #{pr_number} in {repo_full_name}")
    
    # Step 1: Fetch PR analysis
    analyses = fetch_pr_analyses_for_org(
        org_id=org_id,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        limit=1,
    )
    
    if not analyses:
        return {"error": f"No PR analysis found for PR #{pr_number} in {repo_full_name}"}
    
    analysis = analyses[0]
    analysis_data = analysis.get("analysis_data", {})
    files_data = analysis_data.get("files", []) if isinstance(analysis_data, dict) else []
    
    if not files_data:
        return {"error": "No file analysis data found in PR analysis"}
    
    # Step 2: Fetch PR file contents
    file_contents = fetch_pr_files_with_contents(
        org_id=org_id,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        which_ref="head",
    )
    
    if "error" in file_contents:
        logger.warning(f"Could not fetch file contents: {file_contents.get('error')}")
        # Continue anyway - we can still suggest based on impact report
    
    # Step 3: Build context for LLM
    suggestions_by_file: List[Dict[str, Any]] = []
    
    for file_analysis in files_data:
        filename = None
        impact_analysis = file_analysis.get("impact_analysis", "")
        affected_query_ids = file_analysis.get("affected_query_ids", [])
        source_metadata = file_analysis.get("source_metadata", [])
        sql_change = file_analysis.get("sql_change", "")
        
        # Try to extract filename from sql_change or find matching file
        if sql_change:
            for line in sql_change.splitlines():
                if "File:" in line:
                    parts = line.split("File:")
                    if len(parts) > 1:
                        filename = parts[1].strip().split()[0] if parts[1].strip() else None
                        break
        
        # Find matching file content
        file_content_obj = None
        if "files" in file_contents and isinstance(file_contents, dict):
            for f in file_contents.get("files", []):
                if f.get("filename") == filename or (not filename and f.get("status") == "added"):
                    file_content_obj = f
                    break
        
        # Fetch actual query texts for affected queries
        affected_queries = []
        if affected_query_ids:
            try:
                query_records = fetch_queries(affected_query_ids)
                affected_queries = [
                    {"query_id": q.get("query_id"), "query_text": q.get("query_text", "")}
                    for q in query_records
                ]
                logger.info(f"Fetched {len(affected_queries)} actual queries for code suggestions")
            except Exception as e:
                logger.warning(f"Could not fetch query texts: {e}")
        
        # Detect change type for post-processing
        detected_change_type, detected_change_details = _detect_change_type(sql_change, impact_analysis)
        
        # Generate suggestions using LLM
        suggestion_prompt = _build_suggestion_prompt(
            filename=filename or "unknown",
            sql_change=sql_change,
            impact_analysis=impact_analysis,
            affected_query_ids=affected_query_ids,
            affected_queries=affected_queries,
            source_metadata=source_metadata,
            file_content=file_content_obj.get("content") if file_content_obj else None,
            change_type=detected_change_type,
            change_details=detected_change_details,
        )
        
        try:
            # Use CODE_SUGGESTION_LLM (more capable model) for code generation, fallback to CHAT_LLM
            llm_to_use = CODE_SUGGESTION_LLM if CODE_SUGGESTION_LLM else CHAT_LLM
            model_name = "CODE_SUGGESTION_LLM" if CODE_SUGGESTION_LLM else "CHAT_LLM"
            
            # Extract actual model name from LLM object
            actual_model = "unknown"
            if llm_to_use:
                # Try to get model name from different LLM types
                if hasattr(llm_to_use, 'model_name'):
                    actual_model = llm_to_use.model_name
                elif hasattr(llm_to_use, 'model'):
                    actual_model = llm_to_use.model
                elif hasattr(llm_to_use, '_model_name'):
                    actual_model = llm_to_use._model_name
                # Also log the LLM class name for debugging
                llm_class = type(llm_to_use).__name__
                logger.info(f"Using {model_name} ({llm_class}) with model: {actual_model} for code suggestions for {filename}")
            else:
                logger.warning(f"No LLM available for code suggestions for {filename}")
            
            try:
                llm_response = llm_to_use.invoke(suggestion_prompt)
                suggestion_text = getattr(llm_response, "content", str(llm_response))
            except Exception as e:
                # If CODE_SUGGESTION_LLM fails (e.g., model not available), fallback to CHAT_LLM
                error_msg = str(e)
                logger.warning(f"LLM invocation failed for {filename}: {error_msg}")
                # Check if this is a model error and we should fallback
                should_fallback = (
                    "model_not_found" in error_msg.lower() or 
                    "404" in error_msg or 
                    "not available" in error_msg.lower() or
                    "verified" in error_msg.lower()
                )
                
                if should_fallback and CODE_SUGGESTION_LLM and CHAT_LLM and llm_to_use == CODE_SUGGESTION_LLM:
                    logger.warning(f"CODE_SUGGESTION_LLM failed, falling back to CHAT_LLM for {filename}")
                    try:
                        llm_to_use = CHAT_LLM
                        model_name = "CHAT_LLM (fallback)"
                        llm_response = llm_to_use.invoke(suggestion_prompt)
                        suggestion_text = getattr(llm_response, "content", str(llm_response))
                        logger.info(f"Successfully used CHAT_LLM fallback for code suggestions for {filename}")
                    except Exception as fallback_error:
                        logger.error(f"CHAT_LLM fallback also failed for {filename}: {fallback_error}")
                        raise
                else:
                    # Re-raise if we can't fallback
                    raise
            
            logger.info(f"LLM response length: {len(suggestion_text)} characters")
            logger.debug(f"LLM response preview: {suggestion_text[:500]}")
            
            # Try to parse structured JSON from response
            suggestion_json = _parse_suggestion_json(suggestion_text)
            
            if not suggestion_json:
                logger.warning("Failed to parse JSON from LLM response, using raw text")
                logger.debug(f"Raw response: {suggestion_text}")
            elif detected_change_type == "column_removal":
                # Post-process to ensure ALTER TABLE statements are included
                removed_column = detected_change_details.get("column_name", "")
                if removed_column:
                    suggestion_json = _ensure_alter_table_statements(
                        suggestion_json,
                        source_metadata,
                        removed_column,
                    )
            
            suggestions_by_file.append({
                "filename": filename or "unknown",
                "sql_change": sql_change,
                "impact_analysis": impact_analysis,
                "affected_query_ids": affected_query_ids,
                "source_metadata": source_metadata,
                "file_content": file_content_obj.get("content") if file_content_obj else None,
                "suggestions": suggestion_json if suggestion_json else {"raw": suggestion_text},
            })
        except Exception as e:
            logger.error(f"Failed to generate suggestions for {filename}: {e}", exc_info=True)
            suggestions_by_file.append({
                "filename": filename or "unknown",
                "error": f"Failed to generate suggestions: {str(e)}",
            })
    
    return {
        "pr_number": pr_number,
        "repo_full_name": repo_full_name,
        "suggestions_by_file": suggestions_by_file,
    }


def analyze_described_changes_and_suggest_code(
    org_id: str,
    change_description: str,
) -> Dict[str, Any]:
    """
    Analyze user-described changes and suggest code changes to mitigate impacts.
    
    Steps:
    1. Use schema_detection_rag to find affected queries based on the description
    2. Use LLM to extract change type and details from the description
    3. Use LLM to generate code suggestions based on affected queries
    """
    logger.info(f"Analyzing described changes for org {org_id}: {change_description[:100]}...")
    
    # Step 1: Find affected queries using schema_detection_rag
    try:
        impact_result = schema_detection_rag(
            change_text=change_description,
            org_id=org_id,
            cfg=IterativeConfig(max_iters=10, max_assets=20, min_assets=10)
        )
        
        impact_report = impact_result.get("impact_report", "")
        affected_query_ids = impact_result.get("affected_query_ids", [])
        source_metadata = impact_result.get("source_metadata", [])
        
        logger.info(f"Found {len(affected_query_ids)} affected queries from described changes")
    except Exception as e:
        logger.error(f"Error in schema_detection_rag: {e}", exc_info=True)
        return {"error": f"Failed to analyze changes: {str(e)}"}
    
    if not affected_query_ids:
        return {
            "change_description": change_description,
            "suggestions_by_file": [{
                "filename": "unknown",
                "error": "No affected queries found for the described changes. The changes may not impact any existing queries.",
            }],
        }
    
    # Step 2: Fetch actual query texts
    affected_queries = []
    try:
        query_records = fetch_queries(affected_query_ids)
        affected_queries = [
            {"query_id": q.get("query_id"), "query_text": q.get("query_text", "")}
            for q in query_records
        ]
        logger.info(f"Fetched {len(affected_queries)} actual queries for code suggestions")
    except Exception as e:
        logger.warning(f"Could not fetch query texts: {e}")
    
    # Step 3: Use LLM to extract change type and details from description
    change_extraction_prompt = f"""You are analyzing a user's description of database changes they want to make.

User's description: "{change_description}"

Extract the following information and return ONLY valid JSON:
{{
    "change_type": "column_removal|column_rename|column_addition|column_type_change|table_rename|table_removal|table_addition|transformation_logic_change|generic_schema_change",
    "change_details": {{
        "column_name": "name if column change",
        "old_column": "old name if rename",
        "new_column": "new name if rename",
        "table_name": "table name if table change",
        "old_table": "old name if table rename",
        "new_table": "new name if table rename",
        "old_type": "old data type if type change",
        "new_type": "new data type if type change"
    }},
    "sql_change": "Generate a representative SQL statement for this change (e.g., 'ALTER TABLE X DROP COLUMN Y')",
    "filename": "Infer a likely filename or use 'unknown'"
}}

Return ONLY the JSON, no other text."""
    
    try:
        llm_to_use = CODE_SUGGESTION_LLM if CODE_SUGGESTION_LLM else CHAT_LLM
        if not llm_to_use:
            return {"error": "No LLM available for change extraction"}
        
        try:
            extraction_response = llm_to_use.invoke(change_extraction_prompt)
            extraction_text = getattr(extraction_response, "content", str(extraction_response))
        except Exception as e:
            # If CODE_SUGGESTION_LLM fails (e.g., model not available), fallback to CHAT_LLM
            error_msg = str(e)
            logger.warning(f"LLM invocation failed: {error_msg}")
            # Check if this is a model error and we should fallback
            should_fallback = (
                "model_not_found" in error_msg.lower() or 
                "404" in error_msg or 
                "not available" in error_msg.lower() or
                "verified" in error_msg.lower()
            )
            
            if should_fallback and CODE_SUGGESTION_LLM and CHAT_LLM and llm_to_use == CODE_SUGGESTION_LLM:
                logger.warning(f"CODE_SUGGESTION_LLM failed, falling back to CHAT_LLM")
                try:
                    llm_to_use = CHAT_LLM
                    extraction_response = llm_to_use.invoke(change_extraction_prompt)
                    extraction_text = getattr(extraction_response, "content", str(extraction_response))
                    logger.info("Successfully used CHAT_LLM fallback for change extraction")
                except Exception as fallback_error:
                    logger.error(f"CHAT_LLM fallback also failed: {fallback_error}")
                    raise
            else:
                # Re-raise if we can't fallback
                raise
        
        # Parse JSON from response
        change_info = _parse_suggestion_json(extraction_text)
        if not change_info:
            # Fallback: try to detect change type using existing function
            change_type, change_details = _detect_change_type(change_description, change_description)
            change_info = {
                "change_type": change_type,
                "change_details": change_details,
                "sql_change": change_description,
                "filename": "unknown"
            }
        
        change_type = change_info.get("change_type", "generic_schema_change")
        change_details = change_info.get("change_details", {})
        sql_change = change_info.get("sql_change", change_description)
        filename = change_info.get("filename", "unknown")
        
        logger.info(f"Extracted change type: {change_type} from description")
    except Exception as e:
        logger.error(f"Error extracting change info: {e}", exc_info=True)
        # Fallback to generic
        change_type, change_details = _detect_change_type(change_description, change_description)
        change_type = change_type or "generic_schema_change"
        change_details = change_details or {}
        sql_change = change_description
        filename = "unknown"
    
    # Step 4: Generate suggestions using LLM
    suggestion_prompt = _build_suggestion_prompt(
        filename=filename,
        sql_change=sql_change,
        impact_analysis=impact_report,
        affected_query_ids=affected_query_ids,
        affected_queries=affected_queries,
        source_metadata=source_metadata,
        file_content=None,  # No file content for described changes
        change_type=change_type,
        change_details=change_details,
    )
    
    suggestions_by_file = []
    try:
        llm_to_use = CODE_SUGGESTION_LLM if CODE_SUGGESTION_LLM else CHAT_LLM
        model_name = "CODE_SUGGESTION_LLM" if CODE_SUGGESTION_LLM else "CHAT_LLM"
        
        if llm_to_use:
            actual_model = "unknown"
            if hasattr(llm_to_use, 'model_name'):
                actual_model = llm_to_use.model_name
            elif hasattr(llm_to_use, 'model'):
                actual_model = llm_to_use.model
            elif hasattr(llm_to_use, '_model_name'):
                actual_model = llm_to_use._model_name
            llm_class = type(llm_to_use).__name__
            logger.info(f"Using {model_name} ({llm_class}) with model: {actual_model} for code suggestions from described changes")
        else:
            logger.warning(f"No LLM available for code suggestions")
            return {"error": "No LLM available for code suggestions"}
        
        try:
            llm_response = llm_to_use.invoke(suggestion_prompt)
            suggestion_text = getattr(llm_response, "content", str(llm_response))
        except Exception as e:
            # If CODE_SUGGESTION_LLM fails (e.g., model not available), fallback to CHAT_LLM
            error_msg = str(e)
            logger.warning(f"LLM invocation failed: {error_msg}")
            # Check if this is a model error and we should fallback
            should_fallback = (
                "model_not_found" in error_msg.lower() or 
                "404" in error_msg or 
                "not available" in error_msg.lower() or
                "verified" in error_msg.lower()
            )
            
            if should_fallback and CODE_SUGGESTION_LLM and CHAT_LLM and llm_to_use == CODE_SUGGESTION_LLM:
                logger.warning(f"CODE_SUGGESTION_LLM failed, falling back to CHAT_LLM")
                try:
                    llm_to_use = CHAT_LLM
                    model_name = "CHAT_LLM (fallback)"
                    llm_response = llm_to_use.invoke(suggestion_prompt)
                    suggestion_text = getattr(llm_response, "content", str(llm_response))
                    logger.info("Successfully used CHAT_LLM fallback for code suggestions")
                except Exception as fallback_error:
                    logger.error(f"CHAT_LLM fallback also failed: {fallback_error}")
                    raise
            else:
                # Re-raise if we can't fallback
                raise
        
        logger.info(f"LLM response length: {len(suggestion_text)} characters")
        logger.debug(f"LLM response preview: {suggestion_text[:500]}")
        
        # Try to parse structured JSON from response
        suggestion_json = _parse_suggestion_json(suggestion_text)
        
        if not suggestion_json:
            logger.warning("Failed to parse JSON from LLM response, using raw text")
            logger.debug(f"Raw response: {suggestion_text}")
        elif change_type == "column_removal":
            # Post-process to ensure ALTER TABLE statements are included
            removed_column = change_details.get("column_name", "")
            if removed_column:
                suggestion_json = _ensure_alter_table_statements(
                    suggestion_json,
                    source_metadata,
                    removed_column,
                )
        
        suggestions_by_file.append({
            "filename": filename,
            "sql_change": sql_change,
            "impact_analysis": impact_report,
            "affected_query_ids": affected_query_ids,
            "source_metadata": source_metadata,
            "file_content": None,
            "suggestions": suggestion_json if suggestion_json else {"raw": suggestion_text},
        })
    except Exception as e:
        logger.error(f"Failed to generate suggestions: {e}", exc_info=True)
        suggestions_by_file.append({
            "filename": filename,
            "error": f"Failed to generate suggestions: {str(e)}",
        })
    
    return {
        "change_description": change_description,
        "suggestions_by_file": suggestions_by_file,
    }


def _detect_change_type(
    sql_change: str,
    impact_analysis: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect the specific type of schema change from SQL change and impact analysis.
    Returns (change_type, change_details)
    """
    sql_lower = (sql_change or "").lower()
    impact_lower = (impact_analysis or "").lower()
    combined = f"{sql_lower} {impact_lower}"
    
    change_details = {}
    
    # Check for column removal
    if any(pattern in combined for pattern in ["drop column", "remove column", "delete column", "column removal"]):
        # Extract column name
        col_match = re.search(r"drop\s+column\s+([a-z0-9_]+)", sql_lower, re.IGNORECASE)
        if col_match:
            change_details["column_name"] = col_match.group(1)
        return "column_removal", change_details
    
    # Check for column rename
    if any(pattern in combined for pattern in ["rename column", "alter column.*rename", "column rename", "rename to"]):
        # Extract old and new column names
        rename_match = re.search(r"rename\s+(?:column\s+)?([a-z0-9_]+)\s+to\s+([a-z0-9_]+)", sql_lower, re.IGNORECASE)
        if rename_match:
            change_details["old_column"] = rename_match.group(1)
            change_details["new_column"] = rename_match.group(2)
        return "column_rename", change_details
    
    # Check for column addition
    if any(pattern in combined for pattern in ["add column", "alter table.*add", "column addition", "new column"]):
        col_match = re.search(r"add\s+(?:column\s+)?([a-z0-9_]+)", sql_lower, re.IGNORECASE)
        if col_match:
            change_details["column_name"] = col_match.group(1)
        return "column_addition", change_details
    
    # Check for data type change
    if any(pattern in combined for pattern in ["alter column.*type", "change type", "modify column", "data type"]):
        type_match = re.search(r"alter\s+column\s+([a-z0-9_]+).*type\s+([a-z0-9_()]+)", sql_lower, re.IGNORECASE)
        if type_match:
            change_details["column_name"] = type_match.group(1)
            change_details["new_type"] = type_match.group(2)
        return "column_type_change", change_details
    
    # Check for nullability change
    if any(pattern in combined for pattern in ["not null", "null", "set null", "drop not null"]):
        return "column_nullability_change", change_details
    
    # Check for table rename
    if any(pattern in combined for pattern in ["rename table", "alter table.*rename", "table rename"]):
        rename_match = re.search(r"rename\s+(?:table\s+)?([a-z0-9_.]+)\s+to\s+([a-z0-9_.]+)", sql_lower, re.IGNORECASE)
        if rename_match:
            change_details["old_table"] = rename_match.group(1)
            change_details["new_table"] = rename_match.group(2)
        return "table_rename", change_details
    
    # Check for table removal
    if any(pattern in combined for pattern in ["drop table", "remove table", "delete table", "table removal"]):
        table_match = re.search(r"drop\s+table\s+([a-z0-9_.]+)", sql_lower, re.IGNORECASE)
        if table_match:
            change_details["table_name"] = table_match.group(1)
        return "table_removal", change_details
    
    # Check for table addition
    if any(pattern in combined for pattern in ["create table", "new table", "table addition"]):
        return "table_addition", change_details
    
    # Check for DBT transformation logic change
    if any(pattern in combined for pattern in ["transformation", "dbt model", "model logic", "logic update"]):
        return "transformation_logic_change", change_details
    
    # Default to generic schema change
    return "generic_schema_change", change_details


def _extract_affected_tables(source_metadata: List[Dict[str, Any]]) -> List[str]:
    """Extract list of affected tables from source metadata."""
    affected_tables = set()
    for meta in source_metadata:
        if meta.get("target_table"):
            db = meta.get("target_database", "")
            schema = meta.get("target_schema", "")
            table = meta.get("target_table", "")
            full_name = f"{db}.{schema}.{table}".strip('.')
            if full_name:
                affected_tables.add(full_name)
    return list(affected_tables)[:10]


def _build_specialized_prompt(
    change_type: str,
    change_details: Dict[str, Any],
    filename: str,
    sql_change: str,
    impact_analysis: str,
    affected_query_ids: List[str],
    affected_queries: List[Dict[str, Any]],
    source_metadata: List[Dict[str, Any]],
    file_content: Optional[str],
) -> str:
    """
    Build a specialized prompt for generating precise SQL code changes.
    """
    
    # Format affected queries with full context
    queries_context = ""
    if affected_queries:
        queries_context = "\n## AFFECTED QUERIES TO MODIFY\n"
        for idx, q in enumerate(affected_queries[:5], 1):
            query_id = q.get("query_id", "unknown")
            query_text = q.get("query_text", "")
            if query_text:
                queries_context += f"\n### Query {idx} (ID: {query_id})\n```sql\n{query_text[:2000]}\n```\n"
    
    # Extract affected objects for context
    affected_tables = _extract_affected_tables(source_metadata)
    tables_list = ', '.join(list(affected_tables)[:15]) if affected_tables else "See source metadata"
    
    # Base context shared across all change types
    base_context = f"""# SCHEMA CHANGE ANALYSIS CONTEXT

## File Information
- **File:** {filename}
- **Change Type:** {change_type}

## SQL Change Made
```sql
{sql_change[:2000] if sql_change else "N/A"}
```

## Impact Analysis
{impact_analysis[:3000] if impact_analysis else "N/A"}

## Affected Query IDs
{', '.join(affected_query_ids[:10]) if affected_query_ids else "None identified"}

{queries_context}

## Source Metadata (Impacted Objects)
```json
{json.dumps(source_metadata[:20], indent=2) if source_metadata else "[]"}
```

## Affected Downstream Objects
{tables_list}

## Current File Content (Reference)
```sql
{file_content[:2000] if file_content else "File not available"}
```

---
"""

    # Change-specific prompts
    if change_type == "column_removal":
        column_name = change_details.get("column_name", "the_column")
        
        return f"""{base_context}

# TASK: COLUMN REMOVAL CODE CHANGES

## ⚠️ CRITICAL REQUIREMENT ⚠️

**FOR INSERT STATEMENTS:** When you remove a target column from an INSERT statement's column list, you MUST add an `ALTER TABLE DROP COLUMN` statement for that target column in the `additional_statements` field. This is MANDATORY, not optional!

Example: Removing `DEPARTMENTLIST` from `INSERT INTO allocationrule (...)` requires: `ALTER TABLE allocationrule DROP COLUMN departmentlist;`

## Change Details
- **Removed Column:** `{column_name}`
- **Action Required:** Remove all references to `{column_name}` from downstream queries AND drop target columns that were populated by `{column_name}`

---

## INSTRUCTIONS

You are a SQL expert tasked with generating **EXACT CODE CHANGES** to fix queries broken by the removal of column `{column_name}`.

### CRITICAL RULES

1. **USE THE ACTUAL QUERIES PROVIDED ABOVE** - The "AFFECTED QUERIES TO MODIFY" section contains the real SQL that needs fixing
2. **GENERATE COMPLETE SQL STATEMENTS** - Not descriptions, not pseudo-code, but actual executable SQL
3. **PROVIDE BEFORE/AFTER CODE** - Every change must show the exact old code and exact new code
4. **ONE CODE CHANGE PER QUERY** - Each affected query gets one code_changes entry

### STEP-BY-STEP APPROACH

For each query in "AFFECTED QUERIES TO MODIFY":

1. **Identify all occurrences** of `{column_name}` in the query
2. **Determine the fix** based on where the column appears:
   - **SELECT clause:** Remove column from selection list
   - **WHERE clause:** Remove the condition or entire WHERE if it's the only condition
   - **JOIN ON clause:** Remove join or find alternative join key
   - **GROUP BY:** Remove from grouping columns
   - **ORDER BY:** Remove from sorting columns
   - **INSERT column list:** Remove column name and corresponding value
   - **UPDATE SET:** Remove the assignment
   - **HAVING clause:** Remove the condition

3. **Generate the fix:**
   - Copy the ENTIRE original query as `old_code`
   - Create modified version with `{column_name}` completely removed as `new_code`
   - Ensure the new query is syntactically valid SQL

4. **CRITICAL: Identify target columns in INSERT statements and ALWAYS add DROP COLUMN statements:**
   - When removing a column from an INSERT statement's column list, identify the TARGET COLUMN that was being populated
   - For example: If removing `UPPER(NVL(DEPTCD, 'N/A')) AS DEPARTMENTLIST` from SELECT, the target column is `DEPARTMENTLIST` in the INSERT column list
   - The target column is the column name in the INSERT INTO table (...) clause that corresponds to the removed source column
   - **Use the Source Metadata section above** to identify target_table and target_column relationships
   - **MANDATORY:** For EVERY target column removed from an INSERT statement, you MUST add an `ALTER TABLE DROP COLUMN` statement
   - Format: `ALTER TABLE <target_database>.<target_schema>.<target_table> DROP COLUMN <target_column>;` or `ALTER TABLE <target_table> DROP COLUMN <target_column>;`
   - **THIS IS NOT OPTIONAL:** The `additional_statements` field MUST contain the ALTER TABLE DROP COLUMN statement for each removed target column
   - Example: If you remove `DEPARTMENTLIST` from `INSERT INTO allocationrule (...)`, you MUST include `ALTER TABLE transform_zone.edw.allocationrule DROP COLUMN departmentlist;` in additional_statements

5. **Add the ALTER TABLE statement** for the source table if this is the source table being modified

### EXAMPLES

#### Example 1: SELECT with WHERE
```json
{{
  "old_code": "SELECT product_id, {column_name}, price FROM products WHERE {column_name} = 'active' AND price > 100",
  "new_code": "SELECT product_id, price FROM products WHERE price > 100",
  "additional_statements": "ALTER TABLE products DROP COLUMN {column_name};"
}}
```

#### Example 2: INSERT statement with target column removal (REQUIRED PATTERN)
```json
{{
  "old_code": "INSERT INTO allocationrule (allocationrulekey, departmentlist, fundlist) SELECT rulekey, UPPER(NVL({column_name}, 'N/A')) AS departmentlist, bookname FROM gl_summary",
  "new_code": "INSERT INTO allocationrule (allocationrulekey, fundlist) SELECT rulekey, bookname FROM gl_summary",
  "additional_statements": "ALTER TABLE transform_zone.edw.allocationrule DROP COLUMN departmentlist;"
}}
```
**CRITICAL RULE:** When you remove a column from an INSERT statement's column list (like `departmentlist`), you MUST add an `ALTER TABLE DROP COLUMN` statement for that target column. The `additional_statements` field MUST contain the DROP COLUMN statement - it is NOT optional!

#### Example 3: JOIN condition
```json
{{
  "old_code": "SELECT o.order_id, c.customer_name FROM orders o JOIN customers c ON o.{column_name} = c.{column_name}",
  "new_code": "SELECT o.order_id, c.customer_name FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
  "additional_statements": "-- Used alternative join key: customer_id"
}}
```

#### Example 4: Aggregate with GROUP BY
```json
{{
  "old_code": "SELECT {column_name}, COUNT(*) as cnt FROM sales GROUP BY {column_name}",
  "new_code": "SELECT COUNT(*) as cnt FROM sales",
  "additional_statements": "-- Removed grouping since grouping column no longer exists"
}}
```

#### Example 5: Real-world example - deptcd removal (EXACT PATTERN TO FOLLOW)
```json
{{
  "old_code": "INSERT INTO TRANSFORM_ZONE.EDW.ALLOCATIONRULE (ALLOCATIONRULEKEY, DEPARTMENTLIST, FUNDLIST) SELECT rulekey, UPPER(NVL(DEPTCD, 'N/A')) AS DEPARTMENTLIST, bookname FROM GL_SUMMARY",
  "new_code": "INSERT INTO TRANSFORM_ZONE.EDW.ALLOCATIONRULE (ALLOCATIONRULEKEY, FUNDLIST) SELECT rulekey, bookname FROM GL_SUMMARY",
  "additional_statements": "ALTER TABLE TRANSFORM_ZONE.EDW.ALLOCATIONRULE DROP COLUMN DEPARTMENTLIST;"
}}
```
**CRITICAL:** When `deptcd` (source column) is removed and it was populating `DEPARTMENTLIST` (target column) in the INSERT:
1. Remove `DEPARTMENTLIST` from the INSERT column list ✅
2. Remove `UPPER(NVL(DEPTCD, 'N/A')) AS DEPARTMENTLIST` from SELECT ✅
3. **MUST ADD:** `ALTER TABLE TRANSFORM_ZONE.EDW.ALLOCATIONRULE DROP COLUMN DEPARTMENTLIST;` in additional_statements ✅

**If you skip step 3, your response is INCOMPLETE!**

---

## REQUIRED OUTPUT FORMAT

Return ONLY valid JSON with this exact structure:

```json
{{
  "change_type": "column_removal",
  "removed_column": "{column_name}",
  "suggestions": [
    {{
      "impact_level": "depth_1",
      "affected_component": "full.table.name or file.path",
      "description": "Brief description of what's being fixed",
      "code_changes": [
        {{
          "file": "path/to/file.sql",
          "old_code": "COMPLETE original SQL statement",
          "new_code": "COMPLETE modified SQL statement with column removed",
          "explanation": "Specific explanation of what was changed and why",
          "line_number": 10,
          "additional_statements": "ALTER TABLE <target_database>.<target_schema>.<target_table> DROP COLUMN <target_column>;"
        }}
      ],
      "priority": "high|medium|low"
    }}
  ],
  "summary": "Overall summary of changes made"
}}
```

### OUTPUT REQUIREMENTS

✅ **DO:**
- Provide COMPLETE SQL statements (full SELECT, INSERT, UPDATE, etc.)
- Show exact before/after code for each affected query
- Include line numbers if known
- Explain what specifically was removed and why
- Mark priority based on impact (query will fail = high, cosmetic = low)
- **CRITICAL: For INSERT statements, identify target columns being removed and add ALTER TABLE DROP COLUMN statements**
  - When removing a column from INSERT column list, identify the corresponding target column name
  - Add `ALTER TABLE <target_table> DROP COLUMN <target_column>;` to additional_statements
  - Example: If removing `DEPARTMENTLIST` from INSERT INTO allocationrule, add `ALTER TABLE allocationrule DROP COLUMN departmentlist;`
- Include ALTER TABLE statements for source table modifications

❌ **DON'T:**
- Return generic descriptions like "Update the query" or "Remove references"
- Provide partial code snippets or pseudo-code
- Say "review and update" without showing the actual code
- Include explanatory text outside the JSON structure
- Return empty code_changes arrays

---

## VALIDATION CHECKLIST

Before returning your response, verify:

- [ ] Every affected query has a corresponding code_changes entry
- [ ] Each old_code is a complete, executable SQL statement
- [ ] Each new_code is a complete, executable SQL statement with column removed
- [ ] The new_code is syntactically valid SQL
- [ ] All occurrences of `{column_name}` are removed from new_code
- [ ] **MANDATORY CHECK: For EVERY INSERT statement where you removed a target column from the INSERT column list, the `additional_statements` field MUST contain `ALTER TABLE <target_table> DROP COLUMN <target_column>;`**
  - Example: If you removed `DEPARTMENTLIST` from `INSERT INTO allocationrule (...)`, then `additional_statements` MUST contain `ALTER TABLE allocationrule DROP COLUMN departmentlist;`
  - If this is missing, your response is INCOMPLETE and WRONG
- [ ] The JSON is valid and follows the exact schema above
- [ ] No text exists outside the JSON structure

**FINAL CHECK:** Count how many target columns you removed from INSERT statements. Count how many ALTER TABLE DROP COLUMN statements you have. These numbers MUST match!

Generate the code changes now."""

    elif change_type == "column_rename":
        old_column = change_details.get("old_column", "old_column")
        new_column = change_details.get("new_column", "new_column")
        
        return f"""{base_context}

# TASK: COLUMN RENAME CODE CHANGES

## Change Details
- **Old Column Name:** `{old_column}`
- **New Column Name:** `{new_column}`
- **Action Required:** Replace all occurrences of `{old_column}` with `{new_column}`

---

## INSTRUCTIONS

You are a SQL expert tasked with generating **EXACT CODE CHANGES** to update queries after column `{old_column}` was renamed to `{new_column}`.

### CRITICAL RULES

1. **USE THE ACTUAL QUERIES PROVIDED ABOVE** - The "AFFECTED QUERIES TO MODIFY" section contains the real SQL
2. **FIND AND REPLACE ALL OCCURRENCES** - Every instance of `{old_column}` must become `{new_column}`
3. **PRESERVE ALL OTHER LOGIC** - Only the column name changes; everything else stays identical
4. **PROVIDE COMPLETE STATEMENTS** - Full before/after SQL, not just the changed parts

### STEP-BY-STEP APPROACH

For each query in "AFFECTED QUERIES TO MODIFY":

1. **Find all occurrences** of `{old_column}` in:
   - SELECT list
   - WHERE conditions
   - JOIN conditions
   - GROUP BY clauses
   - ORDER BY clauses
   - HAVING clauses
   - INSERT column lists
   - UPDATE SET clauses
   - Column aliases (if column name is used as alias)

2. **Generate the fix:**
   - Copy the ENTIRE original query as `old_code`
   - Create modified version replacing ALL `{old_column}` → `{new_column}` as `new_code`
   - Ensure logic remains identical except for the name

3. **Add ALTER TABLE for source** if applicable

### EXAMPLES

#### Example 1: SELECT with multiple references
```json
{{
  "old_code": "SELECT {old_column}, price FROM products WHERE {old_column} IS NOT NULL ORDER BY {old_column}",
  "new_code": "SELECT {new_column}, price FROM products WHERE {new_column} IS NOT NULL ORDER BY {new_column}",
  "additional_statements": "ALTER TABLE products RENAME COLUMN {old_column} TO {new_column};"
}}
```

#### Example 2: JOIN with table aliases
```json
{{
  "old_code": "SELECT p.{old_column}, o.order_id FROM products p JOIN orders o ON p.{old_column} = o.{old_column}",
  "new_code": "SELECT p.{new_column}, o.order_id FROM products p JOIN orders o ON p.{new_column} = o.{new_column}",
  "additional_statements": "-- Column renamed in both JOIN condition and SELECT"
}}
```

#### Example 3: Aggregate function
```json
{{
  "old_code": "SELECT category, AVG({old_column}) as avg_val FROM sales GROUP BY category HAVING AVG({old_column}) > 100",
  "new_code": "SELECT category, AVG({new_column}) as avg_val FROM sales GROUP BY category HAVING AVG({new_column}) > 100",
  "additional_statements": "-- Column renamed in both AVG function calls"
}}
```

---

## REQUIRED OUTPUT FORMAT

```json
{{
  "change_type": "column_rename",
  "old_column": "{old_column}",
  "new_column": "{new_column}",
  "suggestions": [
    {{
      "impact_level": "depth_1",
      "affected_component": "full.table.name or file.path",
      "description": "Rename column {old_column} to {new_column} in [specific location]",
      "code_changes": [
        {{
          "file": "path/to/file.sql",
          "old_code": "COMPLETE original SQL with {old_column}",
          "new_code": "COMPLETE modified SQL with {new_column}",
          "explanation": "Renamed all occurrences of {old_column} to {new_column}",
          "line_number": 10,
          "additional_statements": "ALTER TABLE statement or notes"
        }}
      ],
      "priority": "high|medium|low"
    }}
  ],
  "summary": "Renamed column {old_column} to {new_column} in X queries across Y objects"
}}
```

### OUTPUT REQUIREMENTS

✅ **DO:**
- Replace ALL occurrences of the old column name
- Keep all other SQL logic exactly the same
- Provide complete before/after statements
- Include ALTER TABLE for source table modifications

❌ **DON'T:**
- Miss any occurrences of the old column name
- Change any other part of the query
- Provide partial code or descriptions

---

## VALIDATION CHECKLIST

- [ ] Every affected query has a code_changes entry
- [ ] All occurrences of `{old_column}` are replaced with `{new_column}` in new_code
- [ ] All other SQL logic remains unchanged
- [ ] The new_code is syntactically valid
- [ ] JSON is valid and complete

Generate the code changes now."""

    elif change_type == "column_addition":
        new_column = change_details.get("column_name", "new_column")
        data_type = change_details.get("data_type", "VARCHAR")
        
        return f"""{base_context}

# TASK: COLUMN ADDITION CODE CHANGES

## Change Details
- **New Column Added:** `{new_column}` ({data_type})
- **Action Required:** Handle the new column in downstream queries

---

## INSTRUCTIONS

You are a SQL expert tasked with updating queries to handle the newly added column `{new_column}`.

### CRITICAL RULES

1. **IDENTIFY SELECT * STATEMENTS** - These automatically include the new column
2. **UPDATE INSERT STATEMENTS** - May need to include the new column
3. **REVIEW EXPLICIT COLUMN LISTS** - Decide if new column should be included
4. **MAINTAIN BACKWARD COMPATIBILITY** - Don't break existing logic

### STEP-BY-STEP APPROACH

For each query in "AFFECTED QUERIES TO MODIFY":

1. **Check for SELECT ***
   - If found, replace with explicit column list
   - Decide: include new column or exclude it?
   - Default: exclude unless query needs it

2. **Check INSERT statements**
   - If inserting into the modified table, add new column
   - Provide appropriate default value or NULL

3. **Check views/CTEs**
   - Update to explicitly list columns

### EXAMPLES

#### Example 1: Replace SELECT *
```json
{{
  "old_code": "SELECT * FROM products WHERE category = 'electronics'",
  "new_code": "SELECT product_id, name, price, category FROM products WHERE category = 'electronics'",
  "additional_statements": "-- Excluded new column {new_column} to maintain existing query output"
}}
```

#### Example 2: INSERT with new column
```json
{{
  "old_code": "INSERT INTO products (product_id, name, price) VALUES (1, 'Widget', 29.99)",
  "new_code": "INSERT INTO products (product_id, name, price, {new_column}) VALUES (1, 'Widget', 29.99, DEFAULT)",
  "additional_statements": "ALTER TABLE products ADD COLUMN {new_column} {data_type} DEFAULT NULL;"
}}
```

#### Example 3: View definition
```json
{{
  "old_code": "CREATE VIEW product_summary AS SELECT * FROM products",
  "new_code": "CREATE OR REPLACE VIEW product_summary AS SELECT product_id, name, price, category FROM products",
  "additional_statements": "-- View updated with explicit columns, excluding {new_column}"
}}
```

---

## REQUIRED OUTPUT FORMAT

```json
{{
  "change_type": "column_addition",
  "new_column": "{new_column}",
  "data_type": "{data_type}",
  "suggestions": [
    {{
      "impact_level": "depth_1",
      "affected_component": "component.name",
      "description": "Update query to handle new column {new_column}",
      "code_changes": [
        {{
          "file": "path/to/file.sql",
          "old_code": "COMPLETE original SQL",
          "new_code": "COMPLETE modified SQL",
          "explanation": "Why this change was made",
          "line_number": 10,
          "additional_statements": "Supporting SQL if needed"
        }}
      ],
      "priority": "medium|low"
    }}
  ],
  "summary": "Updated X queries to handle new column {new_column}"
}}
```

Generate the code changes now."""

    elif change_type == "column_type_change":
        column_name = change_details.get("column_name", "column")
        old_type = change_details.get("old_type", "old_type")
        new_type = change_details.get("new_type", "new_type")
        
        return f"""{base_context}

# TASK: COLUMN DATA TYPE CHANGE

## Change Details
- **Column:** `{column_name}`
- **Old Type:** {old_type}
- **New Type:** {new_type}
- **Action Required:** Add type casting for compatibility

---

## INSTRUCTIONS

You are a SQL expert tasked with adding type conversions after `{column_name}` changed from {old_type} to {new_type}.

### CRITICAL RULES

1. **ADD EXPLICIT TYPE CASTING** where type mismatches occur
2. **IDENTIFY INCOMPATIBLE OPERATIONS** - comparisons, joins, arithmetic
3. **PRESERVE QUERY LOGIC** - only add casting, don't change logic
4. **USE APPROPRIATE CAST FUNCTIONS** - CAST(), CONVERT(), ::type syntax

### COMMON TYPE CHANGE SCENARIOS

**String → Number:**
- WHERE clauses: `WHERE CAST({column_name} AS {new_type}) > 100`
- Arithmetic: `SELECT CAST({column_name} AS {new_type}) * 1.1`

**Number → String:**
- Concatenation: `SELECT CAST({column_name} AS VARCHAR) || ' units'`
- Comparisons: `WHERE CAST({column_name} AS VARCHAR) LIKE '5%'`

**Date type changes:**
- Format: `SELECT TO_CHAR({column_name}, 'YYYY-MM-DD')`
- Parse: `WHERE CAST({column_name} AS DATE) > '2024-01-01'`

### EXAMPLES

#### Example 1: Comparison with type mismatch
```json
{{
  "old_code": "SELECT * FROM orders WHERE {column_name} > 1000",
  "new_code": "SELECT * FROM orders WHERE CAST({column_name} AS {new_type}) > 1000",
  "additional_statements": "ALTER TABLE orders ALTER COLUMN {column_name} TYPE {new_type};"
}}
```

#### Example 2: JOIN with type mismatch
```json
{{
  "old_code": "SELECT * FROM orders o JOIN customers c ON o.{column_name} = c.{column_name}",
  "new_code": "SELECT * FROM orders o JOIN customers c ON CAST(o.{column_name} AS {new_type}) = CAST(c.{column_name} AS {new_type})",
  "additional_statements": "-- Both sides cast to ensure type compatibility"
}}
```

#### Example 3: Arithmetic operation
```json
{{
  "old_code": "SELECT {column_name} * 1.5 as calculated FROM products",
  "new_code": "SELECT CAST({column_name} AS DECIMAL(10,2)) * 1.5 as calculated FROM products",
  "additional_statements": "-- Cast to numeric type for arithmetic"
}}
```

---

## REQUIRED OUTPUT FORMAT

```json
{{
  "change_type": "column_type_change",
  "column": "{column_name}",
  "old_type": "{old_type}",
  "new_type": "{new_type}",
  "suggestions": [
    {{
      "impact_level": "depth_1",
      "affected_component": "component.name",
      "description": "Add type casting for {column_name} ({old_type} → {new_type})",
      "code_changes": [
        {{
          "file": "path/to/file.sql",
          "old_code": "COMPLETE original SQL",
          "new_code": "COMPLETE SQL with CAST() added",
          "explanation": "Added type casting to maintain compatibility",
          "line_number": 10,
          "additional_statements": "ALTER TABLE statement"
        }}
      ],
      "priority": "high"
    }}
  ],
  "summary": "Added type casting for {column_name} in X queries"
}}
```

Generate the code changes now."""

    elif change_type == "table_rename":
        old_table = change_details.get("old_table", "old_table")
        new_table = change_details.get("new_table", "new_table")
        
        return f"""{base_context}

# TASK: TABLE RENAME CODE CHANGES

## Change Details
- **Old Table:** `{old_table}`
- **New Table:** `{new_table}`
- **Action Required:** Replace all references to old table name

---

## INSTRUCTIONS

Simple find-and-replace operation: change all occurrences of `{old_table}` to `{new_table}`.

### EXAMPLES

```json
{{
  "old_code": "SELECT * FROM {old_table} WHERE status = 'active'",
  "new_code": "SELECT * FROM {new_table} WHERE status = 'active'",
  "additional_statements": "ALTER TABLE {old_table} RENAME TO {new_table};"
}}
```

```json
{{
  "old_code": "INSERT INTO {old_table} (col1) VALUES ('val1')",
  "new_code": "INSERT INTO {new_table} (col1) VALUES ('val1')",
  "additional_statements": "-- Table renamed"
}}
```

Generate the code changes now with the standard JSON format."""

    elif change_type == "table_removal":
        table_name = change_details.get("table_name", "table")
        
        return f"""{base_context}

# TASK: TABLE REMOVAL CODE CHANGES

## Change Details
- **Removed Table:** `{table_name}`
- **Action Required:** Remove queries or find replacement data sources

---

## INSTRUCTIONS

For each query using `{table_name}`:

1. **If query ONLY uses removed table:** Comment out or delete
2. **If query joins multiple tables:** Remove the join to {table_name}
3. **If data is available elsewhere:** Replace with alternative source

### EXAMPLES

#### Example 1: Query depends solely on removed table
```json
{{
  "old_code": "SELECT col1, col2 FROM {table_name} WHERE status = 'active'",
  "new_code": "-- REMOVED: Table {table_name} no longer exists
-- SELECT col1, col2 FROM {table_name} WHERE status = 'active'",
  "additional_statements": "DROP TABLE {table_name};"
}}
```

#### Example 2: Remove JOIN
```json
{{
  "old_code": "SELECT o.order_id, t.detail FROM orders o LEFT JOIN {table_name} t ON o.id = t.order_id",
  "new_code": "SELECT o.order_id FROM orders o",
  "additional_statements": "-- Removed JOIN to {table_name} as table no longer exists"
}}
```

Generate the code changes now with the standard JSON format."""

    else:  # generic_schema_change
        return f"""{base_context}

# TASK: GENERIC SCHEMA CHANGE

## Instructions

Analyze the impact analysis report and affected queries to determine what changed.
Then generate appropriate code fixes following these principles:

1. **Use the actual affected queries** provided above
2. **Generate complete SQL statements** for before/after
3. **Follow the JSON format** shown in previous examples
4. **Provide specific explanations** of what changed

Look for:
- Column additions/removals/renames
- Table renames
- Type changes
- Constraint changes

Generate appropriate code changes in the standard JSON format."""

    return prompt


def _build_suggestion_prompt(
    filename: str,
    sql_change: str,
    impact_analysis: str,
    affected_query_ids: List[str],
    affected_queries: List[Dict[str, Any]],
    source_metadata: List[Dict[str, Any]],
    file_content: Optional[str],
    change_type: Optional[str] = None,
    change_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a specialized prompt for the LLM based on the detected change type.
    If change_type and change_details are provided, use them; otherwise detect from sql_change and impact_analysis.
    """
    # Use provided change_type/change_details or detect from sql_change and impact_analysis
    if change_type is None or change_details is None:
        detected_type, detected_details = _detect_change_type(sql_change, impact_analysis)
        change_type = change_type or detected_type
        change_details = change_details or detected_details
    
    logger.info(f"Using change type: {change_type} for file {filename}")
    
    # Build specialized prompt
    return _build_specialized_prompt(
        change_type=change_type,
        change_details=change_details,
        filename=filename,
        sql_change=sql_change,
        impact_analysis=impact_analysis,
        affected_query_ids=affected_query_ids,
        affected_queries=affected_queries,
        source_metadata=source_metadata,
        file_content=file_content,
    )


def _parse_suggestion_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract JSON from LLM response.
    """
    try:
        # Try direct JSON parse
        return json.loads(text)
    except Exception:
        pass
    
    try:
        # Try to find JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    
    return None


def _ensure_alter_table_statements(
    suggestion_json: Dict[str, Any],
    source_metadata: List[Dict[str, Any]],
    removed_column: str,
) -> Dict[str, Any]:
    """
    Post-process suggestion JSON to ensure ALTER TABLE DROP COLUMN statements are included
    for target columns removed from INSERT statements.
    """
    if not suggestion_json or "suggestions" not in suggestion_json:
        return suggestion_json
    
    # Build a map of target columns that should have DROP COLUMN statements
    # Key: (target_database, target_schema, target_table, target_column)
    # Value: full table name
    target_columns_map = {}
    for meta in source_metadata:
        source_col = meta.get("source_column", "").lower()
        if source_col == removed_column.lower():
            target_db = meta.get("target_database", "")
            target_schema = meta.get("target_schema", "")
            target_table = meta.get("target_table", "")
            target_column = meta.get("target_column", "")
            if target_table and target_column:
                # Build full table name
                full_table = f"{target_db}.{target_schema}.{target_table}".strip('.')
                key = (target_db, target_schema, target_table, target_column)
                target_columns_map[key] = {
                    "full_table": full_table,
                    "target_column": target_column,
                }
    
    # Process each suggestion
    for suggestion in suggestion_json.get("suggestions", []):
        code_changes = suggestion.get("code_changes", [])
        for code_change in code_changes:
            old_code = code_change.get("old_code", "").upper()
            new_code = code_change.get("new_code", "").upper()
            additional_statements = code_change.get("additional_statements", "")
            
            # Check if this is an INSERT statement
            if "INSERT INTO" in old_code and "INSERT INTO" in new_code:
                # Find columns removed from INSERT column list
                # Extract INSERT INTO table name and column list from old_code
                import re
                
                # Find INSERT INTO ... (columns) pattern
                insert_match_old = re.search(r"INSERT\s+INTO\s+([^\s(]+)\s*\(([^)]+)\)", old_code, re.IGNORECASE)
                insert_match_new = re.search(r"INSERT\s+INTO\s+([^\s(]+)\s*\(([^)]+)\)", new_code, re.IGNORECASE)
                
                if insert_match_old and insert_match_new:
                    old_table = insert_match_old.group(1).strip()
                    old_columns = [c.strip().upper() for c in insert_match_old.group(2).split(",")]
                    new_columns = [c.strip().upper() for c in insert_match_new.group(2).split(",")]
                    
                    # Find removed columns
                    removed_target_columns = set(old_columns) - set(new_columns)
                    
                    # For each removed target column, check if we have ALTER TABLE statement
                    for removed_target_col in removed_target_columns:
                        # Check if this column is in our target_columns_map
                        found_match = False
                        for key, info in target_columns_map.items():
                            if info["target_column"].upper() == removed_target_col:
                                # Check if ALTER TABLE statement exists
                                table_name_variations = [
                                    info["full_table"],
                                    f"{key[0]}.{key[1]}.{key[2]}".strip('.'),
                                    key[2],  # Just table name
                                ]
                                
                                alter_pattern = rf"ALTER\s+TABLE\s+.*DROP\s+COLUMN\s+{re.escape(removed_target_col)}"
                                if not re.search(alter_pattern, additional_statements, re.IGNORECASE):
                                    # Add ALTER TABLE statement
                                    alter_stmt = f"ALTER TABLE {info['full_table']} DROP COLUMN {removed_target_col};"
                                    if additional_statements:
                                        additional_statements += "\n" + alter_stmt
                                    else:
                                        additional_statements = alter_stmt
                                    code_change["additional_statements"] = additional_statements
                                    logger.info(f"Added missing ALTER TABLE statement: {alter_stmt}")
                                found_match = True
                                break
                        
                        if not found_match:
                            # Still add a generic ALTER TABLE statement if missing
                            alter_pattern = rf"ALTER\s+TABLE\s+.*DROP\s+COLUMN\s+{re.escape(removed_target_col)}"
                            if not re.search(alter_pattern, additional_statements, re.IGNORECASE):
                                # Try to extract table name from INSERT INTO
                                table_name = old_table
                                alter_stmt = f"ALTER TABLE {table_name} DROP COLUMN {removed_target_col};"
                                if additional_statements:
                                    additional_statements += "\n" + alter_stmt
                                else:
                                    additional_statements = alter_stmt
                                code_change["additional_statements"] = additional_statements
                                logger.info(f"Added missing ALTER TABLE statement (generic): {alter_stmt}")
    
    return suggestion_json


def build_org_code_suggestion_tool(org_id: str) -> Tool:
    """
    Build a LangChain Tool for code suggestions based on PR analysis.
    """
    def _fn(question: str) -> str:
        # Parse repo and PR number from question
        import re as _re
        repo_full_name = None
        pr_number = None
        
        # Repo pattern: owner/repo
        m = _re.search(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", question or "")
        if m:
            repo_full_name = m.group(1)
        
        # PR number
        n = _re.search(r"\bpr\s*#?(\d+)\b|\b#(\d+)\b|\bpr\s+(\d+)\b", (question or "").lower())
        if n:
            pr_number = int(next(g for g in n.groups() if g))
        
        # If we have both repo and PR number, use PR analysis path
        if repo_full_name and pr_number:
            logger.info(f"Detected PR analysis request: PR #{pr_number} in {repo_full_name}")
            result = analyze_impact_and_suggest_code(
                org_id=org_id,
                repo_full_name=repo_full_name,
                pr_number=pr_number,
            )
        else:
            # Otherwise, treat as user-described changes
            logger.info(f"Detected user-described changes: {question[:100]}...")
            result = analyze_described_changes_and_suggest_code(
                org_id=org_id,
                change_description=question,
            )
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Format response with emphasis on actual SQL code
        if repo_full_name and pr_number:
            parts = [f"## Code Suggestions for PR #{pr_number} in {repo_full_name}\n"]
        else:
            parts = [f"## Code Suggestions for Described Changes\n"]
            parts.append(f"**Change Description:** {result.get('change_description', question)}\n")
        parts.append("=" * 80)
        
        has_code_changes = False
        
        for file_suggestion in result.get("suggestions_by_file", []):
            filename = file_suggestion.get("filename", "unknown")
            parts.append(f"\n### File: {filename}\n")
            
            if "error" in file_suggestion:
                parts.append(f"❌ Error: {file_suggestion['error']}\n")
                continue
            
            suggestions = file_suggestion.get("suggestions", {})
            if isinstance(suggestions, dict) and "suggestions" in suggestions:
                for idx, sug in enumerate(suggestions["suggestions"], 1):
                    parts.append(f"\n#### Suggestion {idx}: {sug.get('description', 'Code change needed')}")
                    parts.append(f"- **Impact Level:** {sug.get('impact_level', 'unknown')}")
                    parts.append(f"- **Affected Component:** {sug.get('affected_component', 'unknown')}")
                    parts.append(f"- **Priority:** {sug.get('priority', 'medium')}\n")
                    
                    code_changes = sug.get("code_changes", [])
                    if code_changes:
                        has_code_changes = True
                        parts.append(f"**Required Code Changes ({len(code_changes)}):**\n")
                        for cc_idx, cc in enumerate(code_changes, 1):
                            parts.append(f"\n**Change {cc_idx}:**")
                            if cc.get('file'):
                                parts.append(f"- File: `{cc.get('file')}`")
                            if cc.get('line_number'):
                                parts.append(f"- Line: {cc.get('line_number')}")
                            if cc.get('explanation'):
                                parts.append(f"- Reason: {cc.get('explanation')}")
                            
                            old_code = cc.get('old_code', '').strip()
                            new_code = cc.get('new_code', '').strip()
                            
                            if old_code:
                                parts.append(f"\n**BEFORE (Current Code):**")
                                parts.append(f"```sql\n{old_code}\n```\n")
                            
                            if new_code:
                                parts.append(f"**AFTER (Required Change):**")
                                parts.append(f"```sql\n{new_code}\n```\n")
                            else:
                                parts.append(f"⚠️ **WARNING:** No new code provided for this change.\n")
                    else:
                        parts.append("⚠️ **WARNING:** No code changes provided. This suggestion needs actual SQL code.\n")
            else:
                # Try to extract any code from raw response
                raw_text = str(suggestions)
                if "old_code" in raw_text or "new_code" in raw_text or "```sql" in raw_text:
                    parts.append(f"**Raw Response (may contain code):**\n{raw_text[:2000]}\n")
                else:
                    parts.append(f"⚠️ **WARNING:** No structured code suggestions found.\n")
                    parts.append(f"Raw response: {raw_text[:1000]}\n")
        
        if not has_code_changes:
            parts.append("\n" + "=" * 80)
            parts.append("⚠️ **CRITICAL:** No actual SQL code changes were provided!")
            parts.append("The suggestions above are generic descriptions. Please ensure the LLM")
            parts.append("returns actual SQL code in the 'code_changes' array with 'old_code' and 'new_code' fields.")
            parts.append("=" * 80)
        
        # Add structured data for frontend (truncated)
        parts.append("\n\n---\n*Structured data available in response*\n")
        
        return "\n".join(parts)
    
    return Tool(
        name="code_suggestion",
        func=_fn,
        description=(
            "Analyze PR impact reports and suggest code changes to mitigate impacts. "
            "IMPORTANT: This tool automatically fetches PR analysis internally - you do NOT need to call pr_repo_analysis first. "
            "Input format: repo name and PR number (e.g., 'PR 95 in Intellytics-Solutions/github-app-poc' or 'suggest code changes for PR 63 in owner/repo'). "
            "Alternatively, you can provide a description of changes (e.g., 'I am dropping column X from table Y'). "
            "Returns specific SQL code suggestions with old_code and new_code to fix impacted queries and downstream dependencies."
        ),
    )


__all__ = [
    "analyze_impact_and_suggest_code",
    "analyze_described_changes_and_suggest_code",
    "build_org_code_suggestion_tool",
]

