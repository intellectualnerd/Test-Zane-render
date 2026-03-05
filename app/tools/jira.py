from typing import Dict, Any, Optional, List
from langchain.agents import Tool
import uuid
import psycopg2
import psycopg2.extras
import requests
from requests.auth import HTTPBasicAuth
import logging
import re

from app.vector_db import get_db_connection
from app.vector_db import CHAT_LLM
from app.tools.pr_repo import fetch_pr_analyses_for_org
from app.api.jira import get_jira_projects, get_jira_issue_types

logger = logging.getLogger(__name__)


def _fetch_rows(query: str, params: tuple) -> List[Dict[str, Any]]:
    """Helper to fetch rows from database"""
    with get_db_connection() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        norm_params = []
        for p in params:
            if isinstance(p, uuid.UUID):
                norm_params.append(str(p))
            else:
                norm_params.append(p)
        cur.execute(query, tuple(norm_params))
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def get_active_jira_connection(org_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the active Jira connection for an organization.
    Returns connection details or None if not found.
    """
    q = """
        SELECT id, connection_name, server_url, username, api_token
        FROM jira_connections
        WHERE org_id = %s::uuid AND is_active = TRUE
        LIMIT 1
    """
    rows = _fetch_rows(q, (str(org_id),))
    return rows[0] if rows else None


def create_jira_issue(
    server_url: str,
    username: str,
    api_token: str,
    project_key: str,
    summary: str,
    description: str,
    issue_type: str,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a Jira issue using the Jira API.
    Returns dict with key, id, and url.
    """
    try:
        server_url = server_url.rstrip('/')
        auth = HTTPBasicAuth(username, api_token)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Build issue data
        issue_data = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type}
            }
        }
        
        # Add optional fields
        if priority:
            issue_data["fields"]["priority"] = {"name": priority}
        
        if assignee:
            # Use accountId for Jira Cloud (modern format)
            issue_data["fields"]["assignee"] = {"accountId": assignee}
        
        response = requests.post(
            f"{server_url}/rest/api/2/issue",
            auth=auth,
            headers=headers,
            json=issue_data,
            timeout=15
        )
        
        if response.status_code == 201:
            issue_response = response.json()
            logger.info("Created Jira issue: %s", issue_response.get("key"))
            return {
                "key": issue_response["key"],
                "id": issue_response["id"],
                "url": f"{server_url}/browse/{issue_response['key']}"
            }
        else:
            error_msg = f"Failed to create issue: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {"error": error_msg}
    except Exception as e:
        error_msg = f"Failed to create Jira issue: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def create_jira_ticket_for_org(
    org_id: str,
    project_key: str,
    summary: str,
    description: str,
    issue_type: str,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
    pr_url: Optional[str] = None,
    analysis_report_url: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a Jira ticket for an organization.
    Uses the org's active Jira connection.
    Requires project_key and issue_type to be provided.
    """
    logger.info(f"Creating Jira ticket for org_id={org_id}, project={project_key}, summary={summary[:50]}...")
    
    # Get active Jira connection
    connection = get_active_jira_connection(org_id)
    if not connection:
        logger.warning(f"No active Jira connection found for org_id={org_id}")
        return {"error": "No active Jira connection found for this organization. Please configure a Jira connection first using the /jira/save-connection endpoint."}
    
    # Create the Jira issue
    issue_result = create_jira_issue(
        server_url=connection["server_url"],
        username=connection["username"],
        api_token=connection["api_token"],
        project_key=project_key,
        summary=summary,
        description=description,
        issue_type=issue_type,
        priority=priority,
        assignee=assignee,
    )
    
    if "error" in issue_result:
        return issue_result
    
    # Save ticket to database
    try:
        # Generate UUID for the ticket id
        ticket_id = str(uuid.uuid4())
        
        q = """
            INSERT INTO jira_tickets (
                id, connection_id, ticket_key, ticket_url, summary, description,
                issue_type, status, priority, assignee, pr_url, analysis_report_url, created_by
            )
            VALUES (
                %s::uuid, %s::uuid, %s, %s, %s, %s, %s, 'Open', %s, %s, %s, %s, %s::uuid
            )
            RETURNING id, ticket_key, ticket_url, summary, description, issue_type, status, priority, assignee, pr_url, analysis_report_url, created_at
        """
        # Use provided user_id or placeholder (user_id should be passed from chat endpoint)
        created_by = str(user_id) if user_id else str(uuid.uuid4())  # Placeholder if not provided
        
        params = (
            ticket_id,
            str(connection["id"]),
            issue_result["key"],
            issue_result["url"],
            summary,
            description,
            issue_type,
            priority or "Medium",
            assignee,
            pr_url,
            analysis_report_url,
            created_by,
        )
        
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            # Normalize params
            norm_params = []
            for p in params:
                if isinstance(p, uuid.UUID):
                    norm_params.append(str(p))
                else:
                    norm_params.append(p)
            cur.execute(q, tuple(norm_params))
            row = cur.fetchone()
            conn.commit()
            
            if row:
                ticket = dict(row)
                # Convert datetime objects to ISO strings for JSON serialization
                ticket_serializable = {}
                for k, v in ticket.items():
                    if hasattr(v, 'isoformat'):  # datetime objects
                        ticket_serializable[k] = v.isoformat()
                    elif isinstance(v, uuid.UUID):
                        ticket_serializable[k] = str(v)
                    else:
                        ticket_serializable[k] = v
                
                logger.info(f"Saved Jira ticket to database: {ticket['ticket_key']}")
                return {
                    "success": True,
                    "ticket": ticket_serializable,
                    "jira_issue": issue_result,
                }
            else:
                logger.warning("Failed to save ticket to database")
                return {
                    "success": True,
                    "ticket": None,
                    "jira_issue": issue_result,
                    "warning": "Ticket created in Jira but not saved to database",
                }
    except Exception as e:
        logger.error(f"Failed to save ticket to database: {e}", exc_info=True)
        return {
            "success": True,
            "ticket": None,
            "jira_issue": issue_result,
            "warning": f"Ticket created in Jira but database save failed: {str(e)}",
        }


def build_org_jira_tool(org_id: str, user_id: Optional[str] = None) -> Tool:
    """
    Build a LangChain Tool for creating Jira tickets.
    The tool will parse natural language requests and create tickets.
    """
    def _fn(question: str) -> str:
        """
        Parse the question to extract ticket creation details and create a Jira ticket.
        Uses LLM to extract structured information from natural language.
        Asks for project and issue type selection if not provided.
        """
        import json
        
        
        # Get active Jira connection first
        connection = get_active_jira_connection(org_id)
        if not connection:
            return "JIRA CONNECTION NOT CONFIGURED: No active Jira connection found for this organization. Please set up a Jira connection first using the /jira/save-connection endpoint. DO NOT try other tools - this is a configuration issue that must be resolved before creating tickets."
        
        connection_id = str(connection["id"])
        
        # First, try to parse Action Input format (keyword arguments)
        # The agent might pass: summary: "...", description: "...", project_key: "...", issue_type: "..."
        action_input_project_key = None
        action_input_issue_type = None
        
        # Parse Action Input format: "key: value, key: value" or "key=\"value\", key=\"value\""
        # Pattern 1: project_key: "QUEGAI" or project_key: QUEGAI
        project_match = re.search(r"(?:project_key|project)[\s:=]+[\"']?([A-Z][A-Z0-9]{1,10})[\"']?", question, re.IGNORECASE)
        if project_match:
            action_input_project_key = project_match.group(1).upper()
            logger.info(f"Extracted project_key from Action Input: {action_input_project_key}")
        
        # Pattern 2: issue_type: "Task" or issue_type: Task or type: Task
        issue_match = re.search(r"(?:issue_type|issue type|type)[\s:=]+[\"']?(Task|Bug|Story|Epic|Subtask|Issue|Feature)[\"']?", question, re.IGNORECASE)
        if issue_match:
            action_input_issue_type = issue_match.group(1)
            logger.info(f"Extracted issue_type from Action Input: {action_input_issue_type}")
        
        # Also check if the question itself is just "QUEGAI and Task" or similar simple format
        # This handles cases where the agent passes the user's response directly
        if not action_input_project_key or not action_input_issue_type:
            # Pattern: "QUEGAI and Task" or "QUEGAI, Task" or "QUEGAI Task"
            simple_response_pattern = re.search(r"([A-Z][A-Z0-9]{1,10})\s+(?:and|,)\s*(Task|Bug|Story|Epic|Subtask)", question, re.IGNORECASE)
            if simple_response_pattern:
                if not action_input_project_key:
                    action_input_project_key = simple_response_pattern.group(1).upper()
                if not action_input_issue_type:
                    action_input_issue_type = simple_response_pattern.group(2)
                logger.info(f"Extracted from simple response: project_key={action_input_project_key}, issue_type={action_input_issue_type}")
            
            # Pattern: "QUEGAI Task" (space separated)
            if not simple_response_pattern:
                simple_space_pattern = re.search(r"([A-Z][A-Z0-9]{1,10})\s+(Task|Bug|Story|Epic|Subtask)\b", question, re.IGNORECASE)
                if simple_space_pattern:
                    if not action_input_project_key:
                        action_input_project_key = simple_space_pattern.group(1).upper()
                    if not action_input_issue_type:
                        action_input_issue_type = simple_space_pattern.group(2)
                    logger.info(f"Extracted from space-separated: project_key={action_input_project_key}, issue_type={action_input_issue_type}")
        
        # Use LLM to extract ticket information from the question
        extract_prompt = f"""Extract Jira ticket creation details from this request:

"{question}"

Extract the following information:
- project_key: Project key (e.g., "QUEGAI", "PROJ") - if mentioned. Look for patterns like "QUEGAI", "Project: QUEGAI", "project_key: QUEGAI", or project names.
- summary: A clear, concise summary of the issue
- description: Detailed description of the issue
- issue_type: Type of issue (Task, Bug, Story, Epic, Subtask, etc.) - if mentioned. Look for patterns like "Task", "Issue type: Task", "issue_type: Task", or just "Task" after a project key.
- priority: Priority level (Highest, High, Medium, Low, Lowest) - default to "Medium" if not specified
- assignee: Email address or account ID of assignee (if mentioned)
- pr_url: PR URL if mentioned
- analysis_report_url: Analysis report URL if mentioned

IMPORTANT: If the user provides a simple response like "QUEGAI and Task" or "QUEGAI, Task", extract:
- project_key: "QUEGAI"
- issue_type: "Task"

IMPORTANT: If the input is in Action Input format (like "summary: ..., project_key: QUEGAI, issue_type: Task"), extract project_key and issue_type from those fields.

Respond with ONLY valid JSON in this format:
{{
  "project_key": null,
  "summary": "...",
  "description": "...",
  "issue_type": null,
  "priority": "Medium",
  "assignee": null,
  "pr_url": null,
  "analysis_report_url": null
}}
"""
        
        try:
            llm_response = CHAT_LLM.invoke(extract_prompt)
            response_text = getattr(llm_response, "content", str(llm_response))
            
            # Try to parse JSON from response
            ticket_data = None
            try:
                ticket_data = json.loads(response_text)
            except Exception:
                # Try to find JSON block
                start = response_text.find("{")
                end = response_text.rfind("}")
                if start != -1 and end != -1:
                    ticket_data = json.loads(response_text[start:end + 1])
            
            # Fallback: Try regex parsing if LLM extraction failed or didn't find project/issue type
            if not ticket_data:
                ticket_data = {}
            
            # Priority: 1) Action Input parsed values, 2) LLM extracted values, 3) Regex fallback
            # Use Action Input parsed values FIRST (they're most reliable when present)
            if action_input_project_key:
                ticket_data["project_key"] = action_input_project_key
                logger.info(f"Using Action Input project_key: {action_input_project_key}")
            elif ticket_data.get("project_key"):
                logger.info(f"Using LLM extracted project_key: {ticket_data.get('project_key')}")
            
            if action_input_issue_type:
                ticket_data["issue_type"] = action_input_issue_type
                logger.info(f"Using Action Input issue_type: {action_input_issue_type}")
            elif ticket_data.get("issue_type"):
                logger.info(f"Using LLM extracted issue_type: {ticket_data.get('issue_type')}")
            
            # Fallback regex parsing for common patterns like "QUEGAI and Task", "QUEGAI, Task", etc.
            # This is important for handling simple user responses - run ALWAYS to catch any format
            # Pattern 1: "QUEGAI and Task" or "QUEGAI, Task" or "QUEGAI Task"
            # Match uppercase project key followed by "and", comma, or space, then issue type
            if not ticket_data.get("project_key") or not ticket_data.get("issue_type"):
                simple_pattern = re.search(r"([A-Z][A-Z0-9]{1,10})\s+(?:and|,)\s*(Task|Bug|Story|Epic|Subtask|Issue|Feature)", question, re.IGNORECASE)
                if simple_pattern:
                    if not ticket_data.get("project_key"):
                        ticket_data["project_key"] = simple_pattern.group(1).upper()
                    if not ticket_data.get("issue_type"):
                        ticket_data["issue_type"] = simple_pattern.group(2)
                    logger.info(f"Regex extracted (pattern 1): project_key={ticket_data.get('project_key')}, issue_type={ticket_data.get('issue_type')}")
                
                # Pattern 1b: "QUEGAI Task" (space separated, no "and" or comma)
                if (not ticket_data.get("project_key") or not ticket_data.get("issue_type")) and not simple_pattern:
                    simple_pattern2 = re.search(r"([A-Z][A-Z0-9]{1,10})\s+(Task|Bug|Story|Epic|Subtask)\b", question, re.IGNORECASE)
                    if simple_pattern2:
                        if not ticket_data.get("project_key"):
                            ticket_data["project_key"] = simple_pattern2.group(1).upper()
                        if not ticket_data.get("issue_type"):
                            ticket_data["issue_type"] = simple_pattern2.group(2)
                        logger.info(f"Regex extracted (pattern 2): project_key={ticket_data.get('project_key')}, issue_type={ticket_data.get('issue_type')}")
                
                # Pattern 2: "Project: QUEGAI, Issue type: Task" or "project_key: QUEGAI"
                if not ticket_data.get("project_key"):
                    project_pattern = re.search(r"(?:project|project key|project_key)[\s:=]+[\"']?([A-Z][A-Z0-9]{1,10})[\"']?", question, re.IGNORECASE)
                    if project_pattern:
                        ticket_data["project_key"] = project_pattern.group(1).upper()
                        logger.info(f"Regex extracted project_key (pattern 3): {ticket_data['project_key']}")
                
                if not ticket_data.get("issue_type"):
                    issue_pattern = re.search(r"(?:issue type|issue_type|type)[\s:=]+[\"']?(Task|Bug|Story|Epic|Subtask|Issue|Feature)[\"']?", question, re.IGNORECASE)
                    if issue_pattern:
                        ticket_data["issue_type"] = issue_pattern.group(1)
                        logger.info(f"Regex extracted issue_type (pattern 3): {ticket_data['issue_type']}")
                
                # Pattern 4: Look for standalone project key and issue type anywhere in the question
                if not ticket_data.get("project_key"):
                    # Look for uppercase project keys (typically 2-10 chars) that aren't common words
                    project_key_matches = re.findall(r"\b([A-Z]{2,10})\b", question)
                    for potential_key in project_key_matches:
                        if potential_key not in ["PR", "URL", "ID", "AND", "THE", "FOR", "ARE", "TASK", "BUG", "STORY", "EPIC", "NONE", "NULL"]:
                            ticket_data["project_key"] = potential_key
                            logger.info(f"Regex extracted project_key (pattern 4): {potential_key}")
                            break
                
                if not ticket_data.get("issue_type"):
                    # Look for issue type words
                    issue_type_match = re.search(r"\b(Task|Bug|Story|Epic|Subtask|Issue|Feature)\b", question, re.IGNORECASE)
                    if issue_type_match:
                        ticket_data["issue_type"] = issue_type_match.group(1)
                        logger.info(f"Regex extracted issue_type (pattern 4): {ticket_data['issue_type']}")
            
            if not ticket_data:
                return f"Error: Could not parse ticket information from request. Please provide: summary, description, and optionally project_key, issue_type, priority, assignee, pr_url, analysis_report_url."
            
            # Validate required fields (but allow None if not in current question - might be in conversation context)
            summary = ticket_data.get("summary")
            description = ticket_data.get("description")
            
            # Only validate if they were provided in the current question
            if summary is not None:
                summary = summary.strip() if summary else ""
            if description is not None:
                description = description.strip() if description else ""
            
            # Quickly check if PR is mentioned and fetch PR analysis (for later use)
            repo_full_name = None
            pr_number = None
            pr_analysis = None
            
            # Try to parse repo and PR from the raw question
            repo_match = re.search(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", question or "")
            if repo_match:
                repo_full_name = repo_match.group(1)
            pr_match = re.search(r"\bpr\s*#?(\d+)\b|\b#(\d+)\b|\bpr\s+(\d+)\b", (question or "").lower())
            if pr_match:
                pr_number = int(next(g for g in pr_match.groups() if g))
            
            # Try to infer repo/pr from pr_url if provided
            pr_url = ticket_data.get("pr_url")
            if pr_url:
                url_match = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", pr_url)
                if url_match:
                    repo_full_name = repo_full_name or url_match.group(1)
                    pr_number = pr_number or int(url_match.group(2))
            
            # If PR number is mentioned but no repo, try to find it from database
            if pr_number is not None and not repo_full_name:
                try:
                    analyses = fetch_pr_analyses_for_org(
                        org_id=org_id,
                        pr_number=pr_number,
                        limit=1,
                    )
                    if analyses:
                        repo_full_name = analyses[0].get("repo_full_name")
                except Exception:
                    pass  # Continue without repo if fetch fails
            
            # If we have both repo and PR number, quickly fetch PR analysis
            if repo_full_name and pr_number is not None:
                try:
                    analyses = fetch_pr_analyses_for_org(
                        org_id=org_id,
                        repo_full_name=repo_full_name,
                        pr_number=int(pr_number),
                        limit=1,
                    )
                    if analyses:
                        pr_analysis = analyses[0]
                except Exception:
                    pass  # Continue without PR analysis if fetch fails
            
            # IMPORTANT: Always ask for project and issue type - never use inferred values
            # Extract from ticket_data for validation, but always prompt user
            extracted_project_key = ticket_data.get("project_key")
            extracted_issue_type = ticket_data.get("issue_type")
            
            # Clean and validate extracted values
            if extracted_project_key:
                extracted_project_key = extracted_project_key.strip()
            if extracted_issue_type:
                extracted_issue_type = extracted_issue_type.strip()
            
            project_key = None
            issue_type = None
            
            # ALWAYS ask for both project and issue type unless both are explicitly provided
            # Check if values look like placeholders or are missing
            is_placeholder = (
                extracted_project_key and (
                    "YOUR_PROJECT" in extracted_project_key.upper() or 
                    "PROJECT_KEY" in extracted_project_key.upper() or
                    len(extracted_project_key.strip()) == 0
                )
            )
            
            logger.info(f"Extracted project_key: {extracted_project_key}, issue_type: {extracted_issue_type}, is_placeholder: {is_placeholder}")
            logger.info(f"Action Input parsed - project_key: {action_input_project_key}, issue_type: {action_input_issue_type}")
            
            # Use Action Input values if they were parsed and LLM didn't extract them
            if action_input_project_key and not extracted_project_key:
                extracted_project_key = action_input_project_key
                logger.info(f"Using Action Input project_key: {extracted_project_key}")
            if action_input_issue_type and not extracted_issue_type:
                extracted_issue_type = action_input_issue_type
                logger.info(f"Using Action Input issue_type: {extracted_issue_type}")
            
            # Always ask if either is missing or looks like a placeholder
            if not extracted_project_key or not extracted_issue_type or is_placeholder:
                # Fetch available projects and issue types for the first project (as preview)
                try:
                    projects = get_jira_projects(
                        server_url=connection["server_url"],
                        username=connection["username"],
                        api_token=connection["api_token"]
                    )
                    if not projects:
                        return "No projects found in your Jira account. Please ensure you have access to at least one project."
                    
                    projects_list = "\n".join([f"- {p['key']}: {p['name']}" for p in projects])
                    
                    # Fetch issue types for the first project as a preview
                    issue_types_preview = ""
                    try:
                        first_project_key = projects[0]['key']
                        issue_types = get_jira_issue_types(
                            server_url=connection["server_url"],
                            username=connection["username"],
                            api_token=connection["api_token"],
                            project_key=first_project_key
                        )
                        if issue_types:
                            issue_types_preview = "\n\n" + "\n".join([f"- {it['name']}: {it.get('description', 'No description')}" for it in issue_types])
                            issue_types_preview = f"\n\nExample issue types (for project {first_project_key}):{issue_types_preview}"
                    except Exception:
                        # If we can't fetch issue types, continue without preview
                        pass
                    
                    return (
                        "To create a Jira ticket, I need to know which project and issue type to use.\n\n"
                        f"Available projects:\n{projects_list}{issue_types_preview}\n\n"
                        "Please specify:\n"
                        "1. The project key (e.g., 'QUEGAI') or project name\n"
                        "2. The issue type (e.g., 'Task', 'Bug', 'Story', 'Epic')\n\n"
                        "You can provide both in your response, for example: 'QUEGAI and Task' or 'Project: QUEGAI, Issue type: Task'\n\n"
                        "IMPORTANT: When calling this tool again after the user provides project and issue type, you MUST include them in the Action Input like: 'project_key: QUEGAI, issue_type: Task'"
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch projects: {e}", exc_info=True)
                    return f"Failed to fetch available projects: {str(e)}. Please try again or contact support."
            
            # If user provided project_key, validate it
            if extracted_project_key and not is_placeholder:
                try:
                    projects = get_jira_projects(
                        server_url=connection["server_url"],
                        username=connection["username"],
                        api_token=connection["api_token"]
                    )
                    # Try to find matching project by key or name (case-insensitive)
                    matched_project = None
                    for p in projects:
                        if p['key'].upper() == extracted_project_key.upper() or p['name'].lower() == extracted_project_key.lower():
                            matched_project = p
                            break
                    
                    if matched_project:
                        project_key = matched_project['key']  # Use the actual key
                        logger.info(f"Validated project_key: {project_key}")
                    else:
                        # Project not found, show available projects
                        projects_list = "\n".join([f"- {p['key']}: {p['name']}" for p in projects])
                        return (
                            f"Project '{extracted_project_key}' not found. Please select from the available projects:\n\n"
                            f"{projects_list}\n\n"
                            "Please specify the project key (e.g., 'QUEGAI') or project name you want to use for this ticket."
                        )
                except Exception as e:
                    logger.warning(f"Failed to validate project: {e}", exc_info=True)
                    return f"Failed to validate project. Please try again."
            
            # If project_key is set but issue_type is not, ask for issue type
            if project_key and (not extracted_issue_type or not extracted_issue_type.strip()):
                # Fetch available issue types for the selected project and ask user to select
                try:
                    issue_types = get_jira_issue_types(
                        server_url=connection["server_url"],
                        username=connection["username"],
                        api_token=connection["api_token"],
                        project_key=project_key
                    )
                    if not issue_types:
                        return f"No issue types found for project {project_key}. Please check the project configuration."
                    
                    issue_types_list = "\n".join([f"- {it['name']}: {it.get('description', 'No description')}" for it in issue_types])
                    return (
                        f"Great! I'll create the ticket in project '{project_key}'.\n\n"
                        "Now I need to know what type of issue this is.\n\n"
                        f"Available issue types for project {project_key}:\n{issue_types_list}\n\n"
                        "Please specify the issue type (e.g., 'Task', 'Bug', 'Story', 'Epic') you want to use for this ticket."
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch issue types: {e}", exc_info=True)
                    return f"Failed to fetch available issue types for project {project_key}: {str(e)}. Please try again or contact support."
            
            # If both are provided, validate issue_type
            if project_key and extracted_issue_type:
                try:
                    issue_types = get_jira_issue_types(
                        server_url=connection["server_url"],
                        username=connection["username"],
                        api_token=connection["api_token"],
                        project_key=project_key
                    )
                    # Try to find matching issue type by name (case-insensitive)
                    matched_issue_type = None
                    for it in issue_types:
                        if it['name'].lower() == extracted_issue_type.lower():
                            matched_issue_type = it['name']  # Use the actual name from Jira
                            break
                    
                    if matched_issue_type:
                        issue_type = matched_issue_type
                    else:
                        # Issue type not found, show available issue types
                        issue_types_list = "\n".join([f"- {it['name']}: {it.get('description', 'No description')}" for it in issue_types])
                        return (
                            f"Issue type '{extracted_issue_type}' not found for project '{project_key}'. Please select from the available issue types:\n\n"
                            f"{issue_types_list}\n\n"
                            "Please specify the issue type (e.g., 'Task', 'Bug', 'Story', 'Epic') you want to use for this ticket."
                        )
                except Exception as e:
                    logger.warning(f"Failed to validate issue type: {e}", exc_info=True)
                    return f"Failed to validate issue type. Please try again."

            # Generate title and description from PR analysis if available
            # Always use PR analysis format when PR analysis is available (as per user requirement)
            final_summary = summary if summary else ""
            final_description = description if description else ""
            
            # Use PR analysis format when PR analysis is available
            if pr_analysis:
                # Extract PR analysis data
                analysis_data = pr_analysis.get("analysis_data", {})
                pr_title = pr_analysis.get("pr_title", "")
                pr_url = pr_analysis.get("pr_url", "")
                files_data = analysis_data.get("files", []) if isinstance(analysis_data, dict) else []
                
                # Extract high-level change summary from files
                change_summary = ""
                impacted_tables = set()
                impacted_queries_count = 0
                change_type = ""
                affected_table = ""
                affected_column = ""
                
                for file_data in files_data:
                    # Extract impact analysis text to get high-level summary only
                    impact_analysis = file_data.get("impact_analysis", "")
                    if impact_analysis:
                        # Extract only the high-level change summary description
                        import re as _re_extract
                        # Try to extract change summary description
                        summary_match = _re_extract.search(r"Change summary description:\s*(.+?)(?:\n\n|\||Field|###)", impact_analysis, _re_extract.DOTALL)
                        if summary_match:
                            change_summary = summary_match.group(1).strip()
                        
                        # Extract change type and affected table/column from the table if present
                        type_match = _re_extract.search(r"Change Type\s*\|\s*(.+?)(?:\n|$)", impact_analysis)
                        if type_match:
                            change_type = type_match.group(1).strip()
                        
                        table_match = _re_extract.search(r"Affected Table\s*\|\s*(.+?)(?:\n|$)", impact_analysis)
                        if table_match:
                            affected_table = table_match.group(1).strip()
                        
                        column_match = _re_extract.search(r"Affected Column\(s\)\s*\|\s*(.+?)(?:\n|$)", impact_analysis)
                        if column_match:
                            affected_column = column_match.group(1).strip()
                    
                    # Collect impacted tables (high-level only)
                    affected_tables = file_data.get("affected_tables", [])
                    if isinstance(affected_tables, list):
                        impacted_tables.update(affected_tables)
                    affected_query_ids = file_data.get("affected_query_ids", [])
                    if isinstance(affected_query_ids, list):
                        impacted_queries_count += len(affected_query_ids)
                
                # Generate short description for summary
                short_description = change_summary[:80] if change_summary else pr_title[:80]
                if len(change_summary) > 80:
                    try:
                        shorten_prompt = f"Create a very short (max 60 chars) description of this change: {change_summary}"
                        llm_short = CHAT_LLM.invoke(shorten_prompt)
                        short_response = getattr(llm_short, "content", str(llm_short))
                        if short_response and len(short_response.strip()) < 80:
                            short_description = short_response.strip()
                    except Exception:
                        pass
                
                # Build summary: "Impact analysis for PR <PR_LINK or PR_ID> – <Short description>"
                pr_reference = pr_url if pr_url else f"#{pr_number}"
                final_summary = f"Impact analysis for PR {pr_reference} – {short_description}"
                
                # Build concise, low-format description: one table + plain actions
                description_parts = []
                
                # Build a single table with key details (no headings, no bullets, no separator row, no trailing pipes)
                description_parts.append("| Field | Description")
                description_parts.append(f"| Background | {change_summary or 'Change may impact downstream processes and queries.'}")
                description_parts.append(f"| PR Link | {pr_url or f'PR #{pr_number}'}")
                description_parts.append(f"| Repository | {repo_full_name}")
                description_parts.append(f"| PR Title | {pr_title}")
                if change_type:
                    description_parts.append(f"| Change Type | {change_type}")
                if affected_table:
                    description_parts.append(f"| Affected Table | {affected_table}")
                if affected_column:
                    description_parts.append(f"| Affected Column(s) | {affected_column}")
                if impacted_tables:
                    tables_preview = ", ".join(sorted(impacted_tables)[:5])
                    if len(impacted_tables) > 5:
                        tables_preview += f", +{len(impacted_tables)-5} more"
                    description_parts.append(f"| Impacted Tables | {tables_preview}")
                if impacted_queries_count > 0:
                    description_parts.append(f"| Impacted Queries | {impacted_queries_count}")
                
                description_parts.append("")  # spacer
                
                # Required actions as simple checklist (no bullets/indentation)
                description_parts.append("Required Actions:")
                if impacted_queries_count > 0:
                    description_parts.append("☐ Review and update impacted queries")
                if impacted_tables:
                    description_parts.append("☐ Update downstream models/queries that reference impacted tables")
                description_parts.append("☐ Update dashboards/reports if affected")
                description_parts.append("☐ Notify stakeholders of the change")
                description_parts.append("☐ Verify data integrity after changes are applied")
                
                final_description = "\n".join(description_parts)
            
            # Validate summary and description before creating ticket
            if not final_summary or not final_summary.strip():
                return "Error: A summary is required to create a Jira ticket. Please provide a brief summary of the issue."
            
            if not final_description or not final_description.strip():
                return "Error: A description is required to create a Jira ticket. Please provide a detailed description of the issue."
            
            # Ensure we have both project_key and issue_type before proceeding
            if not project_key:
                return "Error: Project key is required. Please specify the project key."
            
            if not issue_type:
                return "Error: Issue type is required. Please specify the issue type."
            
            # Create the ticket
            result = create_jira_ticket_for_org(
                org_id=org_id,
                project_key=project_key,
                summary=final_summary,
                description=final_description,
                issue_type=issue_type,
                priority=ticket_data.get("priority"),
                assignee=ticket_data.get("assignee"),
                pr_url=ticket_data.get("pr_url") or (pr_analysis.get("pr_url") if pr_analysis else None),
                analysis_report_url=ticket_data.get("analysis_report_url"),
                user_id=user_id,
            )
            
            if "error" in result:
                error_msg = result['error']
                # Make connection errors very explicit
                if "no active jira connection" in error_msg.lower() or "connection not found" in error_msg.lower():
                    return f"JIRA CONNECTION NOT CONFIGURED: {error_msg}. Please set up a Jira connection first using the /jira/save-connection endpoint. DO NOT try other tools - this is a configuration issue that must be resolved before creating tickets."
                # For other errors, also make it clear
                return f"JIRA ERROR: {error_msg}. The Jira ticket could not be created. DO NOT try other tools - this is a Jira-specific operation."
            
            # Format response with prominent ticket URL
            parts = ["✅ Jira ticket created successfully!"]
            parts.append("")
            
            if "jira_issue" in result:
                issue = result["jira_issue"]
                ticket_key = issue.get('key', '')
                ticket_url = issue.get('url', '')
                if ticket_key:
                    parts.append(f"**Ticket Key:** {ticket_key}")
                if ticket_url:
                    parts.append(f"**Ticket URL:** {ticket_url}")
                    parts.append("")
            
            if "ticket" in result and result["ticket"]:
                ticket = result["ticket"]
                parts.append(f"**Summary:** {ticket.get('summary', '')}")
                parts.append(f"**Type:** {ticket.get('issue_type', '')}")
                parts.append(f"**Priority:** {ticket.get('priority', '')}")
                if ticket.get("assignee"):
                    parts.append(f"**Assignee:** {ticket.get('assignee')}")
            
            if "warning" in result:
                parts.append("")
                parts.append(f"⚠️ **Warning:** {result['warning']}")
            
            # Add structured data for frontend (handle datetime serialization)
            def _json_serializer(obj):
                """JSON serializer for objects not serializable by default json code"""
                if hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                elif isinstance(obj, uuid.UUID):
                    return str(obj)
                raise TypeError(f"Type {type(obj)} not serializable")
            
            try:
                result_json = json.dumps(result, default=_json_serializer)[:2000]
                parts.append("\nDATA:\n" + result_json)
            except Exception as e:
                logger.warning(f"Failed to serialize result to JSON: {e}")
                # Fallback: convert datetime to string manually
                result_copy = {}
                for k, v in result.items():
                    if hasattr(v, 'isoformat'):
                        result_copy[k] = v.isoformat()
                    elif isinstance(v, dict):
                        result_copy[k] = {k2: (v2.isoformat() if hasattr(v2, 'isoformat') else v2) for k2, v2 in v.items()}
                    else:
                        result_copy[k] = v
                parts.append("\nDATA:\n" + json.dumps(result_copy, default=str)[:2000])
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Failed to create Jira ticket: {e}", exc_info=True)
            error_str = str(e).lower()
            # Check if it's a connection/configuration error
            if "connection" in error_str or "not found" in error_str or "not configured" in error_str:
                return f"JIRA CONNECTION NOT CONFIGURED: Failed to create Jira ticket - {str(e)}. Please set up a Jira connection first using the /jira/save-connection endpoint. DO NOT try other tools - this is a configuration issue."
            return f"JIRA ERROR: Failed to create Jira ticket: {str(e)}. DO NOT try other tools - this is a Jira-specific operation."
    
    return Tool(
        name="create_jira_ticket",
        func=_fn,
        description=(
            "Create a Jira ticket for tracking issues, bugs, or tasks. "
            "IMPORTANT: This tool requires a Jira connection to be configured. "
            "If the tool returns 'JIRA CONNECTION NOT CONFIGURED', do NOT try other tools - inform the user that Jira must be set up first. "
            "CRITICAL: This tool handles Jira ticket creation ONLY. DO NOT use code_suggestion or other tools when creating Jira tickets. "
            "CRITICAL: When the user provides project key and issue type (e.g., 'QUEGAI and Task'), you MUST include them in the Action Input as: 'project_key: QUEGAI, issue_type: Task'. "
            "DO NOT call the tool again without including project_key and issue_type if the user has already provided them. "
            "This tool will: 1) Fetch PR analysis if PR is mentioned, 2) Ask user for project and issue type if not provided, 3) Create the ticket. "
            "Input should include: summary, description, and when the user provides them: project_key, issue_type. "
            "Also optionally: priority, assignee (account_id), pr_url, analysis_report_url. "
            "The tool will fetch available projects and issue types from Jira and ask the user to select if not provided. "
            "Example Action Input when user provided project/type: 'summary: Fix issue, description: Downstream broken, project_key: QUEGAI, issue_type: Task'"
        ),
    )


__all__ = [
    "get_active_jira_connection",
    "create_jira_ticket_for_org",
    "build_org_jira_tool",
]

