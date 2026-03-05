from typing import Dict, Any, List, Optional, Set, Tuple
from langchain.agents import Tool
from langchain.schema import Document
from app.vector_db import get_qa_chain, CHAT_LLM, get_retriever
from app.services.impact_analysis import schema_detection_rag, fetch_queries, IterativeConfig, _docs_to_snippets, _safe_json_parse
import json
import logging

logger = logging.getLogger(__name__)


# Prompts for recursive lineage extraction
LINEAGE_PLAN_PROMPT = """
You are a lineage analyst. You will receive a lineage question and a set of context snippets from a lineage knowledge base.

Your task:
1) Extract ONLY TARGET entities (downstream columns) that have a DIRECT dependency relationship with the source entity mentioned in the question.
   - Look for explicit references showing source_column → target_column relationships
   - DO NOT include columns just because they're in the same table as a dependent column
   - ONLY include columns that are explicitly shown to depend on the source entity in the snippets
2) For each target entity found, generate a query to find ITS downstream dependencies.
3) Return a STRICT JSON as text with the following structure:
       "found_entities": ["database.schema.table.column", ...], 
       "next_queries": ["What are the downstream dependencies of database.schema.table.column?", ...], 
       "notes": "..."

CRITICAL INSTRUCTIONS:
- Extract ONLY TARGET entities that have a DIRECT dependency on the source entity from the question
- DO NOT include all columns from a table - only include columns that are explicitly shown to depend on the source
- For example, if the source is "edw.gl_summary.deptcd" and you see "allocationrule.departmentlist" depends on it, ONLY include "departmentlist", NOT all other columns from allocationrule
- For each target entity in "found_entities", create a query in "next_queries" like: "What are the downstream dependencies of [target_entity]?"
- Use EXACT entity names from snippets (e.g., "transform_zone.edw.allocationrule.departmentlist")
- If no target entities with direct dependencies are found, return empty lists
- Be SELECTIVE - only include entities with clear dependency relationships

Lineage Question:
{question}

Context Snippets (may be empty):
{snippets}

Respond only with JSON text, do not include any extra explanation.
"""

LINEAGE_FINAL_PROMPT = """
You are a lineage analyst tasked with generating a **complete multi-hop downstream lineage report**.

CRITICAL: You MUST ONLY report lineage relationships that are EXPLICITLY shown in the snippets. DO NOT assume, infer, or hallucinate relationships that are not clearly stated.

You are given:
1. The lineage question (identifies the source entity)
2. ALL retrieved lineage context (multi-hop) as raw snippets from multiple iterations

VALIDATION FIRST:
- Check if snippets contain ANY information about the source entity from the question
- If NO relevant information is found, you MUST respond with JSON containing: {{"lineage_report": "NO DATA FOUND: No lineage information available for {source_entity} in the knowledge base.", "source_entity": "{source_entity}", "downstream_entities": []}}
- If snippets are found but don't show downstream dependencies, respond with: {{"lineage_report": "NO DATA FOUND: No downstream dependencies found for {source_entity} in the available lineage data.", "source_entity": "{source_entity}", "downstream_entities": []}}

Your task (ONLY if validation passes):
- Analyze ALL snippets to build a complete downstream lineage graph
- Identify which entities are at DEPTH 1 (directly depend on source), DEPTH 2 (depend on depth 1), DEPTH 3 (depend on depth 2), etc.
- Trace the complete chain: source → depth 1 → depth 2 → depth 3 → depth 4 → depth 5
- Include ALL downstream entities found in the snippets, grouped by their depth level

HOW TO IDENTIFY DEPTH LEVELS:
- Depth 1: Entities that directly reference the source entity in snippets
- Depth 2: Entities that reference depth 1 entities (but NOT the source)
- Depth 3: Entities that reference depth 2 entities (but NOT depth 1 or source)
- Continue this pattern for depth 4, 5, etc.

---

Your output must be valid JSON with these keys:
{{
  "lineage_report": "<the full lineage report, format attached below>",
  "source_entity": "<database.schema.table.column>",
  "downstream_entities": [
    {{
      "depth": 1,
      "database": "...",
      "schema": "...",
      "table": "...",
      "column": "..."
    }}
  ]
}}

---

### 📑 Lineage Report Format (Markdown)

**Source Entity:** {source_entity}

**Complete Downstream Lineage:**

#### Depth 1 (direct dependencies - ONLY columns that directly depend on the source):
1. `database.schema.table.column`
   - Explanation: How this SPECIFIC column directly depends on the source entity {source_entity}
   - IMPORTANT: Only list this column if there is an explicit dependency relationship shown in snippets

#### Depth 2 (ONLY columns that depend on depth 1 columns):
1. `database.schema.table.column`
   - Explanation: How this SPECIFIC column depends on a specific depth 1 column (specify which depth 1 column)
   - IMPORTANT: Only list this column if there is an explicit dependency relationship with a depth 1 column

#### Depth 3 (ONLY columns that depend on depth 2 columns):
1. `database.schema.table.column`
   - Explanation: How this SPECIFIC column depends on a specific depth 2 column (specify which depth 2 column)
   - IMPORTANT: Only list this column if there is an explicit dependency relationship with a depth 2 column

(Continue for depth 4, 5 if found)

**REMEMBER:**
- DO NOT list all columns from a table just because one column in that table has a dependency
- ONLY list columns that have an explicit, direct dependency relationship shown in the snippets
- For example, if "allocationrule.departmentlist" depends on source, ONLY list "departmentlist", NOT "accountfrom", "accountto", etc. unless they also have explicit dependencies

---

**CRITICAL INSTRUCTIONS:**
- Analyze ALL snippets carefully to identify downstream entities that have DIRECT dependency relationships
- ONLY include columns that are explicitly shown to depend on the source entity (for depth 1) or on depth 1 entities (for depth 2), etc.
- DO NOT include all columns from a table just because one column in that table depends on the source
- Group entities by their actual depth level (how many hops from source)
- If an entity appears in multiple snippets, determine its correct depth based on the dependency chain
- Be SELECTIVE - only include entities with clear, explicit dependency relationships shown in the snippets
- The "lineage_report" field MUST contain the FULL detailed lineage report in Markdown format
- DO NOT just say "lineage has been extracted" - you MUST include the actual lineage details
- The lineage_report should show ONLY columns that are directly impacted by the source entity at each depth
- For each entity listed, provide a clear explanation of HOW it depends on the source (or previous depth entity)
- Ensure the JSON is strictly valid (no trailing commas, no comments, no markdown around it)
- Do not hallucinate entities not found in snippets
- If a depth has no entries, omit that depth section
- The lineage_report field is what will be shown to the user - make it accurate and focused on actual dependencies

---

Lineage Question:
{question}

Source Entity: {source_entity}

ALL Retrieved Snippets (from multiple iterations):
{snippets}
"""


LINEAGE_INTENT_CLASSIFIER_PROMPT = """
You are a lineage query classifier. Analyze the following question and determine the user's intent.

Question: {question}

Classify the intent as one of:
1. "upstream" - User wants to know what FEEDS INTO or is the SOURCE OF a column/table
   Examples: "which columns feed into X", "what is the source of X", "where does X come from", "what columns are used to populate X"
   
2. "downstream" - User wants to know what DEPENDS ON or is IMPACTED BY a column/table
   Examples: "trace lineage for X", "what depends on X", "downstream dependencies of X", "what is impacted by X"
   
3. "general" - General lineage question that doesn't clearly indicate direction

Respond with ONLY one word: "upstream", "downstream", or "general"
"""


def classify_lineage_intent(question: str) -> str:
    """
    Use LLM to classify whether the question is asking for upstream, downstream, or general lineage.
    Returns: "upstream", "downstream", or "general"
    """
    try:
        prompt = LINEAGE_INTENT_CLASSIFIER_PROMPT.format(question=question)
        response = CHAT_LLM.invoke(prompt)
        intent = response.content if hasattr(response, "content") else str(response)
        intent = intent.strip().lower()
        
        # Validate response
        if intent in ["upstream", "downstream", "general"]:
            logger.info("Lineage intent classified as: %s for question: %s", intent, question[:100])
            return intent
        else:
            logger.warning("Unexpected intent classification: %s, defaulting to general", intent)
            return "general"
    except Exception as e:
        logger.error("Error classifying lineage intent: %s, defaulting to general", str(e))
        return "general"


def get_recursive_lineage_for_org(org_id: str, question: str, max_depth: int = 5, k_per_query: int = 8) -> str:
    """
    Recursively extract complete downstream lineage up to max_depth levels.
    Similar to schema_detection_rag but focused on lineage tracing.
    """
    # Double-check intent using LLM (in case agent reformulated upstream query as downstream)
    intent = classify_lineage_intent(question)
    if intent == "upstream":
        logger.warning("Detected upstream query in recursive function, redirecting to upstream handler")
        return get_upstream_lineage_for_org(org_id=org_id, question=question, k=k_per_query)
    
    used_queries: List[str] = []
    collected_docs: List[Document] = []
    seen_texts: Set[str] = set()
    seen_entities: Set[str] = set()  # Track entities we've already queried
    frontier: List[str] = [question]
    retriever = get_retriever(org_id, k=k_per_query)

    # Iterative retrieval
    for iteration in range(max_depth):
        new_docs: List[Document] = []
        for q in frontier:
            if q in used_queries:
                continue
            used_queries.append(q)
            docs = retriever.get_relevant_documents(q)
            logger.info(
                "lineage.recursive.retrieval - org=%s iteration=%d query='%s' docs=%d",
                org_id,
                iteration + 1,
                q[:80],
                len(docs or []),
            )
            for d in docs:
                if d.page_content not in seen_texts:
                    seen_texts.add(d.page_content)
                    new_docs.append(d)
                    collected_docs.append(d)
        
        if not new_docs:
            logger.info("No new documents found at iteration %d, stopping", iteration + 1)
            break
        
        snippets_text = _docs_to_snippets(new_docs)
        plan = CHAT_LLM.invoke(LINEAGE_PLAN_PROMPT.format(question=question, snippets=snippets_text))
        plan_text = plan.content if hasattr(plan, "content") else str(plan)
        plan_json = _safe_json_parse(plan_text)
        
        found_entities = plan_json.get("found_entities", []) or []
        next_queries_raw = plan_json.get("next_queries", []) or []
        
        # Build queries for entities we haven't seen yet
        next_queries: List[str] = []
        for entity in found_entities:
            entity_lower = entity.lower().strip()
            if entity_lower and entity_lower not in seen_entities:
                seen_entities.add(entity_lower)
                # Generate a query for this entity's downstream dependencies
                query = f"What are the downstream dependencies of {entity}?"
                if query not in used_queries:
                    next_queries.append(query)
        
        # Also add any explicitly suggested queries that aren't duplicates
        for q in next_queries_raw:
            q_clean = q.strip()
            if q_clean and q_clean not in used_queries:
                next_queries.append(q_clean)
        
        logger.info(
            "lineage.recursive.planning - iteration=%d found_entities=%d next_queries=%d",
            iteration + 1,
            len(found_entities),
            len(next_queries),
        )
        
        frontier = next_queries
        
        if not frontier:
            logger.info("No more queries to follow at iteration %d, stopping", iteration + 1)
            break

    # Generate final lineage report
    all_snippets_text = _docs_to_snippets(collected_docs) if collected_docs else "(no snippets retrieved)"
    
    # Extract source entity from question (improved heuristics)
    source_entity = question
    question_lower = question.lower()
    
    # Try to extract entity from common patterns
    if "lineage for" in question_lower:
        parts = question_lower.split("lineage for")
        if len(parts) > 1:
            entity_part = parts[1]
            # Handle "X column in Y" pattern
            if " in " in entity_part:
                # Extract "X column" part
                source_entity = entity_part.split(" in ")[0].strip()
                # Remove "column" if present
                source_entity = source_entity.replace(" column", "").replace("column ", "").strip()
                # Get the table part
                table_part = entity_part.split(" in ")[1].split(" to")[0].strip()
                # Combine: table.column
                source_entity = f"{table_part}.{source_entity}"
            else:
                source_entity = entity_part.split("to")[0].strip()
    elif "trace" in question_lower and "lineage" in question_lower:
        # Pattern: "Trace lineage for X"
        parts = question_lower.split("lineage")
        if len(parts) > 1:
            remaining = parts[1]
            if "for" in remaining:
                source_entity = remaining.split("for")[1].split("to")[0].strip()
    elif "downstream" in question_lower:
        # Pattern: "What are downstream dependencies of X"
        if "of" in question_lower:
            source_entity = question_lower.split("of")[1].strip().rstrip("?")
    
    # Clean up the source entity
    source_entity = source_entity.strip().rstrip("?.,")
    
    # If we have a column name but need to construct full path, try to infer
    # For pattern like "deptcd column in edw.gl_summary", we want "edw.gl_summary.deptcd"
    if "." not in source_entity and " in " in question_lower:
        # Try to extract table from "in X" pattern
        if " in " in question_lower:
            table_part = question_lower.split(" in ")[1].split(" to")[0].split(" column")[0].strip()
            if table_part:
                source_entity = f"{table_part}.{source_entity}"
    
    logger.info(
        "lineage.recursive.final - org=%s collected_docs=%d source_entity='%s'",
        org_id,
        len(collected_docs),
        source_entity[:100],
    )
    
    final = CHAT_LLM.invoke(LINEAGE_FINAL_PROMPT.format(question=question, snippets=all_snippets_text, source_entity=source_entity))
    final_text = final.content if hasattr(final, "content") else str(final)
    
    logger.info("lineage.recursive.final.response - response_length=%d", len(final_text))
    
    # Try to parse JSON
    final_json = _safe_json_parse(final_text)
    
    # Extract lineage_report from JSON
    lineage_report = final_json.get("lineage_report", "")
    
    # If JSON parsing failed or lineage_report is empty, try to extract from markdown/text
    if not lineage_report or lineage_report == "No complete lineage found.":
        logger.warning("lineage.recursive.final - JSON parsing may have failed, trying fallback extraction")
        
        # Try to extract markdown content between markers
        if "**Complete Downstream Lineage:**" in final_text or "#### Depth" in final_text:
            # Extract the markdown section
            if "**Complete Downstream Lineage:**" in final_text:
                parts = final_text.split("**Complete Downstream Lineage:**")
                if len(parts) > 1:
                    lineage_report = parts[1].strip()
                    # Remove any trailing JSON closing braces
                    if lineage_report.endswith("}"):
                        lineage_report = lineage_report.rsplit("}", 1)[0].strip()
        elif "lineage_report" in final_text.lower():
            # Try to extract from JSON-like structure even if not valid JSON
            import re
            match = re.search(r'"lineage_report"\s*:\s*"([^"]+)"', final_text, re.DOTALL)
            if not match:
                match = re.search(r'"lineage_report"\s*:\s*"([^"]*(?:\\.|[^"\\])*)"', final_text, re.DOTALL)
            if match:
                lineage_report = match.group(1).replace('\\n', '\n').replace('\\"', '"')
        
        # If still empty, use the full response (might be markdown already)
        if not lineage_report or len(lineage_report.strip()) < 50:
            logger.warning("lineage.recursive.final - Using full response as fallback")
            lineage_report = final_text.strip()
            # Remove JSON wrapper if present
            if lineage_report.startswith("{") and '"lineage_report"' in lineage_report:
                # Try to extract just the report content
                try:
                    import json
                    temp_json = json.loads(final_text)
                    lineage_report = temp_json.get("lineage_report", final_text)
                except:
                    pass
    
    # Check if the report is just a confirmation message (too short or generic)
    lineage_report_lower = lineage_report.lower() if lineage_report else ""
    is_generic_confirmation = any(phrase in lineage_report_lower for phrase in [
        "successfully extracted",
        "has been extracted",
        "extraction complete",
        "lineage extracted",
    ]) and len(lineage_report.strip()) < 200
    
    # Final fallback: if we have collected docs but no meaningful report, create a detailed summary
    if (not lineage_report or len(lineage_report.strip()) < 200 or is_generic_confirmation) and collected_docs:
        logger.warning(
            "lineage.recursive.final - Report too short or generic (%d chars), creating detailed summary from collected docs",
            len(lineage_report.strip()) if lineage_report else 0
        )
        
        # Use LLM to generate a summary from the collected snippets
        summary_prompt = f"""You are a lineage analyst. Generate a complete downstream lineage report from the following snippets.

Source Entity: {source_entity}

Snippets:
{all_snippets_text[:4000]}

Generate a detailed lineage report showing:
1. Source entity
2. Depth 1 (direct dependencies)
3. Depth 2 (dependencies of depth 1)
4. Continue for all depths found

Format as Markdown with clear depth levels. Include entity names and brief explanations."""
        
        try:
            summary_response = CHAT_LLM.invoke(summary_prompt)
            summary_text = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
            if summary_text and len(summary_text.strip()) > 100:
                lineage_report = summary_text.strip()
                logger.info("lineage.recursive.final - Generated summary from LLM, length=%d", len(lineage_report))
            else:
                raise ValueError("LLM summary too short")
        except Exception as e:
            logger.warning("lineage.recursive.final - LLM summary failed: %s, using snippet summary", str(e))
            # Fallback to snippet-based summary
            lineage_report = f"**Source Entity:** {source_entity}\n\n**Complete Downstream Lineage:**\n\n"
            lineage_report += f"Found {len(collected_docs)} lineage documents across multiple depth levels.\n\n"
            lineage_report += "**Lineage Information:**\n\n"
            lineage_report += all_snippets_text[:3000]  # First 3000 chars of snippets
    
    if not lineage_report or len(lineage_report.strip()) < 50:
        # Check if we actually have relevant documents
        if len(collected_docs) == 0:
            lineage_report = f"**Source Entity:** {source_entity}\n\n**Status:** NO DATA FOUND: No lineage information available for '{source_entity}' in the knowledge base. Please verify the entity name and ensure lineage data has been ingested."
        else:
            # We have docs but couldn't generate report - might be irrelevant docs
            lineage_report = f"**Source Entity:** {source_entity}\n\n**Status:** NO DATA FOUND: The retrieved documents do not contain specific lineage information for '{source_entity}'. Please verify the entity name."
    
    logger.info("lineage.recursive.final.result - report_length=%d, is_generic=%s", len(lineage_report), is_generic_confirmation)
    return lineage_report


UPSTREAM_LINEAGE_PROMPT = """
You are a lineage analyst. You will receive a question asking which columns feed into target column(s), and context snippets from a lineage knowledge base.

CRITICAL: You MUST ONLY answer based on what is EXPLICITLY shown in the snippets. DO NOT assume, infer, or hallucinate relationships. DO NOT mention targets that are not in the question.

IMPORTANT: The snippets are formatted as separate fields. Each snippet represents ONE lineage relationship:
- target_database: <database>
- target_schema: <schema>
- target_table: <table>
- target_column: <column>
- source_database: <database>
- source_schema: <schema>
- source_table: <table>
- source_column: <column>

This means: source_table.source_column → target_table.target_column

TARGET COLUMNS FROM QUESTION (ONLY answer for these):
{target_info}

YOUR TASK:
1. Look at the question and identify EXACTLY which target column(s) are mentioned
2. For EACH target column mentioned in the question, find snippets where target_table AND target_column match that target
3. Extract ONLY the source_table and source_column from those matching snippets
4. Format each source as: `source_database.source_schema.source_table.source_column`
5. If question mentions ONE target, answer for ONE target only. If question mentions TWO targets, answer for TWO targets only.

CRITICAL RULES - FOLLOW STRICTLY:

**RULE 1: ANSWER ONLY FOR TARGETS IN THE QUESTION**
- Count how many targets are mentioned in the question
- Answer ONLY for those targets - no more, no less
- If question asks about "allocationrule.departmentlist" (ONE target), answer for ONE target only
- DO NOT mention "second target" or "other targets" if they're not in the question

**RULE 2: SAME TABLE = NOT A SOURCE**
- If target is "allocationrule.accountto", DO NOT include "allocationrule.accountlist" or ANY column from "allocationrule" table
- Columns from the same table as the target are SIBLINGS, not SOURCES
- A source MUST be from a DIFFERENT table than the target
- Example: Target "allocationrule.accountto" → Valid source: "gl_summary.accountcd" ✓, Invalid: "allocationrule.accountlist" ✗

**RULE 3: EXACT TARGET MATCHING**
- Only use snippets where target_table AND target_column BOTH match the question's target
- If question asks about "allocationrule.accountto", only use snippets with:
  - target_table: allocationrule
  - target_column: accountto
- DO NOT use snippets with different target_column (e.g., target_column: accountlist) even if same table

**RULE 4: ONE SOURCE PER SNIPPET**
- Each snippet shows ONE source → ONE target relationship
- If snippet shows "source_table: gl_summary, source_column: accountcd, target_table: allocationrule, target_column: accountto"
  → This means ONLY "gl_summary.accountcd" feeds into "allocationrule.accountto"
  → DO NOT include other columns from gl_summary or allocationrule

**RULE 5: NO DATA = SAY SO (ONLY FOR TARGETS IN QUESTION)**
- If no snippets match a target mentioned in the question → "NO DATA FOUND: No lineage information available for [target]"
- If snippets exist but show no sources → "NO DATA FOUND: No upstream sources found for [target]"
- DO NOT mention targets that are NOT in the question

EXAMPLE - SINGLE TARGET:

Question: "Which columns feed into transform_zone.edw.allocationrule.departmentlist?"
Snippets:
- target_database: transform_zone, target_schema: edw, target_table: allocationrule, target_column: departmentlist, source_database: transform_zone, source_schema: edw, source_table: gl_summary, source_column: deptcd

Correct Answer:
The columns that feed into `transform_zone.edw.allocationrule.departmentlist` are:
1. `transform_zone.edw.gl_summary.deptcd`

Incorrect Answer (DO NOT DO THIS):
The columns that feed into `transform_zone.edw.allocationrule.departmentlist` are:
1. `transform_zone.edw.gl_summary.deptcd`
For the second target `edw.allocationrule`, there is no lineage information available.

✗ WRONG: Question only mentions ONE target, so do NOT mention a "second target"

EXAMPLE - MULTIPLE TARGETS:

Question: "Which columns feed into allocationrule.accountto and allocationrule.marketlist?"
Snippets:
- target_table: allocationrule, target_column: accountto, source_table: gl_summary, source_column: accountcd
- target_table: allocationrule, target_column: marketlist, source_table: gl_summary, source_column: segment3

Correct Answer:
The columns that feed into `allocationrule.accountto` are:
1. `transform_zone.edw.gl_summary.accountcd`

The columns that feed into `allocationrule.marketlist` are:
1. `transform_zone.edw.gl_summary.segment3`

Question:
{question}

Context Snippets (ONLY snippets matching the target columns are included):
{snippets}

Now provide your answer. Count how many targets are in the question and answer for EXACTLY that many targets - no more, no less.
"""


def _extract_targets_from_question(question: str) -> List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """
    Extract ALL target database, schema, table, and column from question.
    Handles multiple targets (e.g., "feed into X and Y").
    Returns list of (database, schema, table, column) tuples.
    More precise - only extracts full qualified names, avoids partial matches.
    """
    import re
    targets = []
    
    # More precise patterns - look for full qualified column names
    # Pattern 1: database.schema.table.column (4 parts)
    pattern_4 = r"(\w+)\.(\w+)\.(\w+)\.(\w+)"
    matches_4 = re.finditer(pattern_4, question, re.IGNORECASE)
    for match in matches_4:
        groups = match.groups()
        targets.append((groups[0], groups[1], groups[2], groups[3]))
    
    # Pattern 2: schema.table.column (3 parts) - only if not already captured by 4-part pattern
    # But be careful - this might match partials of 4-part patterns
    # So we'll only use 3-part if it's clearly separate (has word boundaries)
    pattern_3 = r"(?<!\w)(\w+)\.(\w+)\.(\w+)(?!\.\w)"  # 3 parts, not part of 4-part
    matches_3 = re.finditer(pattern_3, question, re.IGNORECASE)
    for match in matches_3:
        groups = match.groups()
        # Check if this is already part of a 4-part match
        start, end = match.span()
        is_part_of_4part = False
        for target in targets:
            # If we already have a 4-part target, skip 3-part matches that overlap
            if target[0] is not None:  # 4-part target exists
                is_part_of_4part = True
                break
        if not is_part_of_4part:
            targets.append((None, groups[0], groups[1], groups[2]))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_targets = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            unique_targets.append(target)
    
    # If we have 4-part targets, prefer those and remove any 3-part that might be duplicates
    if any(t[0] is not None for t in unique_targets):
        # We have at least one 4-part target, filter out 3-part that are likely duplicates
        filtered = []
        for target in unique_targets:
            if target[0] is not None:
                filtered.append(target)
            else:
                # Check if this 3-part is a duplicate of a 4-part
                is_duplicate = False
                for t4 in unique_targets:
                    if t4[0] is not None and t4[1:] == target[1:]:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered.append(target)
        unique_targets = filtered
    
    return unique_targets if unique_targets else [(None, None, None, None)]


def _validate_source_columns(answer: str, docs: List[Document], targets: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]) -> str:
    """
    Post-process LLM answer to validate that returned columns are actually sources.
    Handles multiple targets - validates each section separately.
    Filters out columns from the same table as any target (they're not sources, they're siblings).
    Cross-references with actual documents to ensure accuracy.
    """
    import re
    
    # Extract all column references from answer (format: `database.schema.table.column`)
    column_pattern = r"`([^`]+)`"
    mentioned_columns = re.findall(column_pattern, answer)
    
    if not mentioned_columns:
        return answer  # No columns to validate
    
    # Build a set of valid source columns from documents for EACH target
    # A valid source must have target_table=target_table AND target_column=target_column
    valid_sources: Set[str] = set()
    target_tables: Set[str] = set()  # All target tables to filter against
    
    for target_db, target_schema, target_table, target_column in targets:
        if target_table:
            target_tables.add(target_table.lower())
        
        target_table_lower = target_table.lower() if target_table else None
        target_column_lower = target_column.lower() if target_column else None
        
        for doc in docs:
            content = doc.page_content.lower()
            # Extract target and source from document
            target_table_match = re.search(r"target_table:\s*(\w+)", content)
            target_column_match = re.search(r"target_column:\s*(\w+)", content)
            source_table_match = re.search(r"source_table:\s*(\w+)", content)
            source_column_match = re.search(r"source_column:\s*(\w+)", content)
            source_db_match = re.search(r"source_database:\s*(\w+)", content)
            source_schema_match = re.search(r"source_schema:\s*(\w+)", content)
            
            # Check if this document is about our target
            doc_target_table = target_table_match.group(1) if target_table_match else None
            doc_target_column = target_column_match.group(1) if target_column_match else None
            
            if target_table_lower and target_column_lower:
                # Must match both table and column
                if doc_target_table == target_table_lower and doc_target_column == target_column_lower:
                    # This is a valid source document
                    if source_table_match and source_column_match:
                        source_table = source_table_match.group(1)
                        source_column = source_column_match.group(1)
                        source_db = source_db_match.group(1) if source_db_match else None
                        source_schema = source_schema_match.group(1) if source_schema_match else None
                        
                        # Build full column path
                        parts = []
                        if source_db:
                            parts.append(source_db)
                        if source_schema:
                            parts.append(source_schema)
                        parts.append(source_table)
                        parts.append(source_column)
                        valid_source = ".".join(parts)
                        valid_sources.add(valid_source.lower())
            elif target_table_lower:
                # Only table match required
                if doc_target_table == target_table_lower:
                    if source_table_match and source_column_match:
                        source_table = source_table_match.group(1)
                        source_column = source_column_match.group(1)
                        source_db = source_db_match.group(1) if source_db_match else None
                        source_schema = source_schema_match.group(1) if source_schema_match else None
                        
                        parts = []
                        if source_db:
                            parts.append(source_db)
                        if source_schema:
                            parts.append(source_schema)
                        parts.append(source_table)
                        parts.append(source_column)
                        valid_source = ".".join(parts)
                        valid_sources.add(valid_source.lower())
    
    # Filter mentioned columns to only include valid sources
    # CRITICAL: Filter out columns from the same table as ANY target (they're not sources)
    filtered_columns = []
    invalid_columns = []
    
    for col in mentioned_columns:
        col_lower = col.lower()
        col_parts = col_lower.split('.')
        
        # Extract table from column path (usually second-to-last)
        col_table = None
        if len(col_parts) >= 2:
            col_table = col_parts[-2]  # e.g., "transform_zone.edw.allocationrule.departmentlist" -> "allocationrule"
        elif len(col_parts) == 1:
            continue  # Can't validate single part
        
        # CRITICAL: Filter out columns from the same table as ANY target
        # These are NOT sources, they're just other columns in the same table
        if col_table and col_table in target_tables:
            invalid_columns.append(col)
            logger.debug(f"Filtering out {col} - same table as target ({col_table})")
            continue
        
        # Check if this column is in our valid sources set
        if col_lower in valid_sources:
            filtered_columns.append(col)
        else:
            # Check if it's a partial match
            col_simple = ".".join(col_parts[-2:]) if len(col_parts) >= 2 else col_lower
            if any(vs.endswith(col_simple) or vs == col_simple for vs in valid_sources):
                filtered_columns.append(col)
            else:
                invalid_columns.append(col)
                logger.debug(f"Filtering out {col} - not in valid sources")
    
    # Rebuild answer with only valid columns, preserving section structure
    if not filtered_columns:
        return "NO DATA FOUND: No valid upstream sources found for the requested columns in the available lineage data. The retrieved documents do not show any columns that feed into the targets."
    
    # Rebuild the answer section by section
    lines = answer.split('\n')
    new_lines = []
    section_valid_count = {}  # Track count per section
    current_section = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Detect section headers (lines mentioning a target column)
        for target_db, target_schema, target_table, target_column in targets:
            if target_table and target_column:
                target_pattern = rf"{target_table}\.{target_column}|{target_column}"
                if re.search(target_pattern, line_stripped, re.IGNORECASE):
                    current_section = f"{target_table}.{target_column}"
                    if current_section not in section_valid_count:
                        section_valid_count[current_section] = 0
                    new_lines.append(line)  # Keep section header
                    continue
        
        # Check if this line contains a column reference
        col_match = re.search(column_pattern, line)
        if col_match:
            col = col_match.group(1)
            col_lower = col.lower()
            
            # Check if this column is valid
            if any(fc.lower() == col_lower for fc in filtered_columns):
                if current_section:
                    section_valid_count[current_section] += 1
                    count = section_valid_count[current_section]
                else:
                    # Fallback: count all valid columns
                    count = len([c for c in filtered_columns if c.lower() == col_lower])
                
                # Update numbering
                new_line = re.sub(r'^\d+\.', f'{count}.', line)
                new_lines.append(new_line)
            # Skip invalid columns silently
        else:
            # Keep non-column lines (headers, explanations, etc.)
            # But remove lines that mention invalid columns
            if not any(col.lower() in line_stripped.lower() for col in invalid_columns):
                new_lines.append(line)
    
    if invalid_columns:
        logger.warning(f"Filtered out invalid source columns (same table as target or not in documents): {invalid_columns}")
    
    return '\n'.join(new_lines)


def _filter_docs_by_targets(docs: List[Document], targets: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]) -> List[Document]:
    """
    Filter documents to only include those where target_table and target_column match the question's targets.
    This ensures the LLM only sees relevant snippets, preventing it from including wrong columns.
    """
    import re
    filtered_docs = []
    
    # Build set of (table, column) pairs from targets
    target_pairs = set()
    for target_db, target_schema, target_table, target_column in targets:
        if target_table and target_column:
            target_pairs.add((target_table.lower(), target_column.lower()))
    
    if not target_pairs:
        # If we can't extract targets, return all docs (fallback)
        return docs
    
    for doc in docs:
        content = doc.page_content.lower()
        # Extract target from document
        target_table_match = re.search(r"target_table:\s*(\w+)", content)
        target_column_match = re.search(r"target_column:\s*(\w+)", content)
        
        if target_table_match and target_column_match:
            doc_target_table = target_table_match.group(1)
            doc_target_column = target_column_match.group(1)
            
            # Only include if this document's target matches one of our question targets
            if (doc_target_table, doc_target_column) in target_pairs:
                filtered_docs.append(doc)
    
    return filtered_docs if filtered_docs else docs  # Fallback to all docs if filtering removes everything


def get_upstream_lineage_for_org(org_id: str, question: str, k: int = 8) -> str:
    """
    Extract upstream lineage (what feeds into a target column).
    Uses filtered retrieval + improved prompt to ensure accurate LLM responses without post-processing.
    """
    # Extract targets first to enable filtering
    targets = _extract_targets_from_question(question)
    
    # Retrieve documents
    retriever = get_retriever(org_id, k=k * 2)  # Retrieve more, then filter
    docs = retriever.get_relevant_documents(question)
    
    # CRITICAL: Filter documents to only include those matching our targets
    # This prevents the LLM from seeing irrelevant information
    filtered_docs = _filter_docs_by_targets(docs, targets)
    
    # Validate that we have documents
    if not filtered_docs or len(filtered_docs) == 0:
        return f"NO DATA FOUND: No lineage information available in the knowledge base for the requested column. Please verify the column name and ensure lineage data has been ingested."
    
    snippets_text = _docs_to_snippets(filtered_docs)
    
    # Build enhanced prompt with target information - make it very clear how many targets
    target_count = len([t for t in targets if t[2] and t[3]])  # Count targets with table and column
    target_info = ""
    if target_count == 0:
        target_info = "See question for target details."
    elif target_count == 1:
        # Single target - make it very clear
        for target_db, target_schema, target_table, target_column in targets:
            if target_table and target_column:
                if target_db and target_schema:
                    target_info = f"ONE target: {target_db}.{target_schema}.{target_table}.{target_column}"
                else:
                    target_info = f"ONE target: {target_table}.{target_column}"
                break
    else:
        # Multiple targets
        target_info = f"{target_count} targets:\n"
        for idx, (target_db, target_schema, target_table, target_column) in enumerate(targets, 1):
            if target_table and target_column:
                if target_db and target_schema:
                    target_info += f"{idx}. {target_db}.{target_schema}.{target_table}.{target_column}\n"
                else:
                    target_info += f"{idx}. {target_table}.{target_column}\n"
    
    enhanced_prompt = UPSTREAM_LINEAGE_PROMPT.format(
        question=question,
        snippets=snippets_text,
        target_info=target_info
    )
    
    # Use CHAT_LLM with enhanced prompt
    response = CHAT_LLM.invoke(enhanced_prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    
    # Add source previews only if we have actual data
    previews: List[str] = []
    for doc in filtered_docs[:3]:  # Limit to first 3 for brevity
        preview = (doc.page_content or "").strip()[:200]
        if preview:
            previews.append(preview)
    
    sources_block = ("\n\nSource Context:\n- " + "\n- ".join(previews)) if previews else ""
    return (answer or "No upstream dependencies found.") + sources_block


def get_lineage_answer_for_org(org_id: str, question: str, k: int = 8, original_question: Optional[str] = None) -> str:
    """
    Query the org's lineage vector store and return an answer with brief source previews.
    
    Uses LLM-based intent classification to determine if question is asking for:
    - Upstream lineage (what feeds into X): uses enhanced QA chain
    - Downstream lineage (what depends on X): uses recursive extraction
    - General lineage: uses simple QA chain
    
    Args:
        original_question: The original user question (if available) to help detect upstream queries
    """
    # Use original question for classification if available (helps when agent reformulates)
    classification_question = original_question or question
    
    # Classify intent using LLM
    intent = classify_lineage_intent(classification_question)
    
    question_lower = question.lower()
    
    if intent == "upstream":
        logger.info("Using upstream lineage extraction for question: %s (original: %s)", question[:100], original_question[:100] if original_question else "N/A")
        return get_upstream_lineage_for_org(org_id=org_id, question=original_question or question, k=k)
    elif intent == "downstream":
        # Check if user wants complete/downstream lineage with recursive extraction
        needs_recursive = any(keyword in question_lower for keyword in [
            "complete", "downstream", "full", "all", "entire", "trace", "follow"
        ])
        
        if needs_recursive:
            logger.info("Using recursive downstream lineage extraction for question: %s", question[:100])
            return get_recursive_lineage_for_org(org_id=org_id, question=question, max_depth=5, k_per_query=k)
        else:
            # Simple downstream query
            logger.info("Using simple downstream lineage extraction for question: %s", question[:100])
            qa = get_qa_chain(org_id=org_id, k=k, llm=CHAT_LLM)  # Use CHAT_LLM for chatbot tools
            result: Dict[str, Any] = qa.invoke({"query": question})
            answer = result.get("result", "") or result.get("answer", "")
            source_docs = result.get("source_documents", []) or []
            
            # Validate that we have source documents
            if not source_docs or len(source_docs) == 0:
                return "NO DATA FOUND: No lineage information available in the knowledge base for this query. Please verify the column/table names and ensure lineage data has been ingested."
            
            # Check if answer is generic or doesn't contain actual lineage information
            answer_lower = answer.lower() if answer else ""
            if not answer or len(answer.strip()) < 20:
                return "NO DATA FOUND: No specific lineage information found for this query in the knowledge base."
            
            previews: List[str] = []
            for doc in source_docs[:3]:  # Limit to first 3
                preview = (getattr(doc, "page_content", "") or "").splitlines()[:5]
                if preview:
                    previews.append("\n".join(preview))
            sources_block = ("\n\nSources:\n- " + "\n- ".join(previews)) if previews else ""
            return (answer or "No answer found.") + sources_block
    
    # Simple QA chain for basic lineage questions
    qa = get_qa_chain(org_id=org_id, k=k, llm=CHAT_LLM)  # Use CHAT_LLM for chatbot tools
    result: Dict[str, Any] = qa.invoke({"query": question})
    answer = result.get("result", "") or result.get("answer", "")
    source_docs = result.get("source_documents", []) or []

    # Validate that we have source documents
    if not source_docs or len(source_docs) == 0:
        return "NO DATA FOUND: No lineage information available in the knowledge base for this query. Please verify the column/table names and ensure lineage data has been ingested."
    
    # Check if answer is generic or doesn't contain actual lineage information
    answer_lower = answer.lower() if answer else ""
    if not answer or len(answer.strip()) < 20:
        return "NO DATA FOUND: No specific lineage information found for this query in the knowledge base."
    
    # Check if answer contains actual column/table references
    import re
    has_column_refs = bool(re.search(r'\w+\.\w+', answer))  # Check for table.column pattern
    if not has_column_refs and ("no" in answer_lower or "not found" in answer_lower or "unable" in answer_lower):
        return "NO DATA FOUND: No lineage information available for this query in the knowledge base."

    previews: List[str] = []
    for doc in source_docs[:3]:  # Limit to first 3
        preview = (getattr(doc, "page_content", "") or "").splitlines()[:5]
        if preview:
            previews.append("\n".join(preview))

    sources_block = ("\n\nSources:\n- " + "\n- ".join(previews)) if previews else ""
    return (answer or "No answer found.") + sources_block


def build_org_lineage_tool(org_id: str, k: int = 8) -> Tool:
    """
    Build a LangChain Tool bound to a specific org_id for lineage Q&A.
    """
    def _fn(question: str) -> str:
        # Store original question for upstream detection
        return get_lineage_answer_for_org(org_id=org_id, question=question, k=k, original_question=question)

    return Tool(
        name="extract_lineage",
        func=_fn,
        description=(
            "Answer lineage questions using the organization's vector store. "
            "IMPORTANT: Distinguish between upstream and downstream queries:\n"
            "- UPSTREAM: Questions asking 'which columns feed into X', 'what feeds X', 'source of X' - shows what feeds INTO a column\n"
            "- DOWNSTREAM: Questions asking 'trace lineage for X', 'downstream dependencies of X' - shows what depends ON a column\n"
            "For downstream questions with 'complete', 'full', 'all', or 'trace', automatically extracts multi-hop lineage up to 5 depths. "
            "For upstream questions, shows only the specific source columns that directly feed into the target. "
            "Input is a natural-language question (e.g., 'Which columns feed into departmentlist in dimsharedservicesallocationrule' or 'Trace lineage for edw.gl_summary.deptcd to downstream columns')."
        ),
    )


def get_query_history_search_for_org(
    org_id: str,
    question: str,
    max_iters: Optional[int] = 5,
) -> str:
    """
    Analyze schema/SQL changes and return impacted queries using the org's vector store.
    
    Takes a natural language question about a schema change (e.g., "I am going to remove x column")
    and returns a formatted response with impacted queries.
    """
    # Use schema_detection_rag to find impacted queries
    cfg = IterativeConfig(max_iters=max_iters) if max_iters else None
    result = schema_detection_rag(change_text=question, org_id=org_id, cfg=cfg)
    
    impact_report = result.get("impact_report", "")
    affected_query_ids = result.get("affected_query_ids", [])
    
    # Fetch actual query texts for the affected query IDs
    regression_queries = []
    if affected_query_ids:
        regression_queries = fetch_queries(affected_query_ids)
    
    # Format the response - make it very clear and structured for the agent
    response_parts = []
    
    if impact_report:
        # Truncate impact report if too long to prevent agent confusion
        report_text = impact_report[:1000] + "..." if len(impact_report) > 1000 else impact_report
        response_parts.append("=== IMPACT ANALYSIS ===")
        response_parts.append(report_text)
        response_parts.append("")  # Empty line for readability
    
    if regression_queries:
        response_parts.append(f"=== IMPACTED QUERIES ({len(affected_query_ids)} total) ===")
        for idx, query_info in enumerate(regression_queries[:10], 1):  # Limit to first 10 for brevity
            query_id = query_info.get("query_id", "Unknown")
            query_text = query_info.get("query_text", "")
            # Clean up query text - remove extra whitespace and newlines
            query_text = " ".join(query_text.split())[:200] + "..." if len(query_text) > 200 else " ".join(query_text.split())
            response_parts.append(f"{idx}. Query ID: {query_id}")
            response_parts.append(f"   SQL Preview: {query_text}")
            response_parts.append("")  # Empty line between queries
        
        if len(regression_queries) > 10:
            response_parts.append(f"... and {len(regression_queries) - 10} more queries")
    elif affected_query_ids:
        # If we have IDs but couldn't fetch queries, just report the count
        response_parts.append(f"=== IMPACTED QUERY IDs ({len(affected_query_ids)} total) ===")
        response_parts.append(", ".join(affected_query_ids[:10]))
        if len(affected_query_ids) > 10:
            response_parts.append(f"... and {len(affected_query_ids) - 10} more")
    else:
        response_parts.append("=== NO IMPACTED QUERIES FOUND ===")
        response_parts.append("No impacted queries found in query history.")
    
    # Return a clean, structured response
    result = "\n".join(response_parts) if response_parts else "No impact analysis results found."
    
    # Ensure response is not too long (max 3000 chars to allow for more queries)
    if len(result) > 3000:
        result = result[:2900] + "\n... (response truncated due to length)"
    
    return result


def build_org_query_history_tool(org_id: str, max_iters: Optional[int] = 5) -> Tool:
    """
    Build a LangChain Tool bound to a specific org_id for query history search.
    Analyzes schema changes and finds impacted queries.
    """
    def _fn(question: str) -> str:
        return get_query_history_search_for_org(org_id=org_id, question=question, max_iters=max_iters)
    
    return Tool(
        name="query_history_search",
        func=_fn,
        description=(
            "Analyze schema/SQL changes and find impacted queries. Input is a natural-language description of a change "
            "(e.g., 'I am going to remove column X from table Y'). Returns a formatted list with Query IDs and SQL previews. "
            "IMPORTANT: The agent MUST include the complete tool output in the final answer, not a summary."
        ),
    )


__all__ = [
    "get_lineage_answer_for_org",
    "build_org_lineage_tool",
    "get_query_history_search_for_org",
    "build_org_query_history_tool",
    "CHAT_LLM",
]


