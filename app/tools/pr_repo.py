from typing import List, Dict, Any, Optional, Tuple
import os
import json
import uuid
import requests
import psycopg2
import psycopg2.extras
from langchain.agents import Tool
try:
    from github import GithubIntegration, Github  # PyGithub
except Exception:  # pragma: no cover
    GithubIntegration = None  # type: ignore
    Github = None  # type: ignore

from app.vector_db import get_db_connection
from dotenv import load_dotenv
load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
PRIVATE_KEY = os.getenv("GITHUB_PRIVATE_KEY")

# Create global GitHub integration instance (matching pattern from app/api/github.py exactly)
git_integration: Optional[GithubIntegration] = None
if GITHUB_APP_ID and PRIVATE_KEY:
    try:
        git_integration = GithubIntegration(int(GITHUB_APP_ID), PRIVATE_KEY)
    except Exception:
        git_integration = None


def _fetch_rows(query: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Normalize params to primitive types psycopg2 can adapt
        norm_params = []
        for p in params:
            if isinstance(p, uuid.UUID):
                norm_params.append(str(p))
            else:
                norm_params.append(p)
        cur.execute(query, tuple(norm_params))
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def fetch_pr_analyses_for_org(
    org_id: str,
    repo_full_name: Optional[str] = None,
    pr_number: Optional[int] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    filters: List[str] = ["org_id = %s::uuid"]
    params: List[Any] = [str(org_id)]
    if repo_full_name:
        filters.append("repo_full_name = %s")
        params.append(repo_full_name)
    if pr_number is not None:
        filters.append("pr_number = %s")
        params.append(pr_number)
    where = " AND ".join(filters)
    q = f"""
        SELECT id, org_id, installation_id, repository_id, repo_full_name, pr_number, pr_title, analysis_data, created_at
        FROM github_pr_analyses
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(limit)
    return _fetch_rows(q, tuple(params))


def _get_installation_for_repo(org_id: str, repo_full_name: str) -> Optional[Dict[str, Any]]:
    q = """
        SELECT gi.id as installation_pk, gi.installation_id as installation_external_id, gr.id as repo_pk,
               gr.full_name, gr.default_branch
        FROM github_installations gi
        JOIN github_repositories gr ON gr.installation_id = gi.id
        WHERE gi.org_id = %s::uuid AND gr.full_name = %s AND gi.is_active = TRUE
        LIMIT 1
    """
    rows = _fetch_rows(q, (str(org_id), repo_full_name))
    return rows[0] if rows else None


def _get_installation_token(installation_external_id: str) -> Optional[str]:
    """
    Get installation access token using the global git_integration instance.
    Matches the pattern used in app/api/github.py
    """
    if not git_integration:
        import logging
        logger = logging.getLogger(__name__)
        if not GITHUB_APP_ID or not PRIVATE_KEY:
            logger.debug("GitHub credentials not configured: GITHUB_APP_ID=%s, PRIVATE_KEY=%s", 
                        bool(GITHUB_APP_ID), bool(PRIVATE_KEY))
        else:
            logger.debug("git_integration instance is None despite credentials being set")
        return None
    try:
        return git_integration.get_access_token(int(installation_external_id)).token
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to get installation token for {installation_external_id}: {e}")
        return None


def fetch_repo_tree(
    org_id: str,
    repo_full_name: str,
    recursive: bool = True,
) -> Dict[str, Any]:
    inst = _get_installation_for_repo(org_id, repo_full_name)
    if not inst:
        return {"error": "Installation/repository not found or not linked to org"}
    token = _get_installation_token(inst["installation_external_id"])
    if not token:
        return {"error": "GitHub Integration not configured"}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Get branch SHA
    default_branch = inst.get("default_branch") or "main"
    branch_resp = requests.get(f"{GITHUB_API_BASE}/repos/{repo_full_name}/branches/{default_branch}", headers=headers)
    if branch_resp.status_code != 200:
        return {"error": f"Failed to fetch branch: {branch_resp.text}"}
    sha = branch_resp.json().get("commit", {}).get("sha")
    if not sha:
        return {"error": "Failed to resolve commit sha"}

    params = {"recursive": 1} if recursive else {}
    tree_resp = requests.get(f"{GITHUB_API_BASE}/repos/{repo_full_name}/git/trees/{sha}", headers=headers, params=params)
    if tree_resp.status_code != 200:
        return {"error": f"Failed to fetch tree: {tree_resp.text}"}
    return tree_resp.json()


def build_org_pr_repo_tool(org_id: str, default_limit: int = 10) -> Tool:
    """
    Build a tool that can:
    - Fetch stored PR analyses for the org (optionally for a given repo/pr).
    - Fetch repository tree to provide codebase context.

    Input: natural language like "show pr 123 in org repo owner/name" or just a repo name.
    Output: concise text summary plus selected JSON blobs for the agent to reference.
    """
    def _safe_json(obj: Any) -> str:
        def _default(o):
            try:
                import uuid as _uuid
                if isinstance(o, _uuid.UUID):
                    return str(o)
            except Exception:
                pass
            return str(o)
        try:
            return json.dumps(obj, default=_default)
        except Exception:
            # Fallback stringify
            return json.dumps(json.loads(json.dumps(obj, default=_default)))

    def _fn(question: str) -> str:
        # Parse repo and PR number heuristically
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

        analyses = fetch_pr_analyses_for_org(org_id=org_id, repo_full_name=repo_full_name, pr_number=pr_number, limit=default_limit)
        tree = fetch_repo_tree(org_id=org_id, repo_full_name=repo_full_name) if repo_full_name else None

        parts: List[str] = []
        if analyses:
            parts.append(f"Found {len(analyses)} stored PR analysis record(s).")
            for idx, a in enumerate(analyses[:5], 1):
                title = a.get("pr_title") or ""
                parts.append(f"{idx}. {a.get('repo_full_name')} PR #{a.get('pr_number')} {('- ' + title) if title else ''}")
                # Try to surface a concise impact summary if present
                try:
                    files = a.get("analysis_data", {}).get("files", []) if isinstance(a.get("analysis_data"), dict) else []
                    if files:
                        first = files[0]
                        impact_preview = (first.get("impact_analysis", "") or "").strip().splitlines()[:6]
                        if impact_preview:
                            parts.append("   Impact Preview: " + " ".join(impact_preview))
                except Exception:
                    pass
        else:
            parts.append("No stored PR analyses found for the given filters.")

        # Append compact JSON for the agent/frontend
        # Flatten useful fields for frontend: files with impact, affected_query_ids, regression_queries
        flattened: List[Dict[str, Any]] = []
        for a in analyses:
            ad = a.get("analysis_data", {}) if isinstance(a.get("analysis_data"), dict) else {}
            files = ad.get("files", []) if isinstance(ad, dict) else []
            for f in files:
                flattened.append({
                    "repo_full_name": a.get("repo_full_name"),
                    "pr_number": a.get("pr_number"),
                    "pr_title": a.get("pr_title"),
                    "impact_analysis": f.get("impact_analysis"),
                    "affected_query_ids": f.get("affected_query_ids", []),
                    "regression_queries": f.get("regression_queries", []),
                    "source_metadata": f.get("source_metadata", []),
                    "sql_change": f.get("sql_change"),
                })

        payload = {
            "analyses": analyses,
            "repo_tree": tree,
            "files": flattened,
        }
        parts.append("\nDATA:\n" + _safe_json(payload)[:4000])
        return "\n".join(parts)

    return Tool(
        name="pr_repo_analysis",
        func=_fn,
        description=(
            "Fetch stored PR analyses for the org and (optionally) the GitHub repo tree. "
            "Include owner/repo and optional PR number in the input for precision."
        ),
    )


__all__ = [
    "fetch_pr_analyses_for_org",
    "fetch_repo_tree",
    "build_org_pr_repo_tool",
]


# -----------------------------
# Branch- and PR-aware code fetch utilities
# -----------------------------

def get_pr_branches_and_shas(
    org_id: str,
    repo_full_name: str,
    pr_number: int,
) -> Dict[str, Any]:
    """
    Return base/head refs and SHAs for a PR using PyGithub.
    Requires GitHub App configured for the repo.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not Github:
        return {"error": "PyGithub library not available"}
    
    inst = _get_installation_for_repo(org_id, repo_full_name)
    if not inst:
        logger.warning(f"Installation not found for org_id={org_id}, repo={repo_full_name}")
        return {"error": "Installation/repository not found or not linked to org"}
    
    token = _get_installation_token(inst["installation_external_id"])
    if not token:
        logger.warning(f"Failed to get installation token for installation_id={inst['installation_external_id']}")
        return {"error": "GitHub Integration not configured"}

    try:
        gh = Github(login_or_token=token)
        repo = gh.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        return {
            "base_ref": pr.base.ref,
            "base_sha": pr.base.sha,
            "head_ref": pr.head.ref,
            "head_sha": pr.head.sha,
        }
    except Exception as e:
        logger.error(f"Failed to fetch PR branches/shas: {e}", exc_info=True)
        return {"error": f"Failed to fetch PR: {str(e)}"}


def fetch_pr_changed_files(
    org_id: str,
    repo_full_name: str,
    pr_number: int,
    per_page: int = 100,
    max_pages: int = 5,
) -> List[Dict[str, Any]]:
    """
    List files changed in a PR using PyGithub (matching pattern from github.py).
    Returns metadata: filename, status, patch, additions, deletions.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not Github:
        logger.warning("PyGithub library not available")
        return []
    
    inst = _get_installation_for_repo(org_id, repo_full_name)
    if not inst:
        logger.warning(f"Installation not found for org_id={org_id}, repo={repo_full_name}")
        return []
    
    token = _get_installation_token(inst["installation_external_id"])
    if not token:
        logger.warning(f"Failed to get installation token for installation_id={inst['installation_external_id']}")
        return []

    try:
        gh = Github(login_or_token=token)
        repo = gh.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        # Use PyGithub's get_files() method (same as github.py)
        files = pr.get_files()
        normalized: List[Dict[str, Any]] = []
        for f in files:
            normalized.append(
                {
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "patch": f.patch,
                    "raw_url": f.raw_url if hasattr(f, "raw_url") else None,
                    "contents_url": f.contents_url if hasattr(f, "contents_url") else None,
                }
            )
        logger.info(f"Fetched {len(normalized)} changed files for PR #{pr_number} in {repo_full_name}")
        return normalized
    except Exception as e:
        logger.error(f"Failed to fetch PR changed files: {e}", exc_info=True)
        return []


def _extract_content_from_patch(patch: Optional[str], status: Optional[str]) -> Optional[str]:
    """
    Extract file content from a git patch for added/modified files.
    For added files, extract all lines starting with '+'.
    For modified files, this is less reliable, so returns None.
    """
    if not patch or status != "added":
        return None
    
    lines = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            # Remove the leading '+' from the patch line
            lines.append(line[1:])
    
    if lines:
        return "\n".join(lines)
    return None


def fetch_file_at_ref(
    org_id: str,
    repo_full_name: str,
    path: str,
    ref: str,
    patch: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch a single file's text content at the given ref/branch/sha using PyGithub.
    Falls back to extracting content from patch if API access fails (403 permission error).
    Returns {content, encoding, path, ref} or {error}.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not Github:
        # Try to extract from patch as fallback
        if patch and status == "added":
            content = _extract_content_from_patch(patch, status)
            if content:
                logger.info(f"Extracted content from patch for {path} (PyGithub not available)")
                return {"path": path, "ref": ref, "encoding": "patch", "content": content}
        return {"error": "PyGithub library not available", "path": path, "ref": ref}
    
    inst = _get_installation_for_repo(org_id, repo_full_name)
    if not inst:
        logger.warning(f"Installation not found for org_id={org_id}, repo={repo_full_name}")
        return {"error": "Installation/repository not found or not linked to org", "path": path, "ref": ref}
    
    token = _get_installation_token(inst["installation_external_id"])
    if not token:
        logger.warning(f"Failed to get installation token for installation_id={inst['installation_external_id']}")
        return {"error": "GitHub Integration not configured", "path": path, "ref": ref}

    try:
        gh = Github(login_or_token=token)
        repo = gh.get_repo(repo_full_name)
        # Use PyGithub's get_contents() method with ref parameter
        file_content = repo.get_contents(path, ref=ref)
        
        if file_content.content:
            try:
                import base64
                decoded = base64.b64decode(file_content.content).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Failed to decode file content for {path} at {ref}: {e}")
                decoded = None
        else:
            decoded = None
            
        logger.debug(f"Fetched file {path} at ref {ref}, size={file_content.size if hasattr(file_content, 'size') else 'unknown'}")
        return {"path": path, "ref": ref, "encoding": "base64", "content": decoded}
    except Exception as e:
        error_str = str(e)
        # Check if it's a 403 permission error
        if "403" in error_str or "not accessible by integration" in error_str.lower():
            logger.warning(f"Permission denied (403) for {path} at {ref}, attempting to extract from patch")
            # Try to extract content from patch as fallback
            if patch and status == "added":
                content = _extract_content_from_patch(patch, status)
                if content:
                    logger.info(f"Successfully extracted content from patch for {path}")
                    return {"path": path, "ref": ref, "encoding": "patch", "content": content}
            logger.warning(f"Could not extract content from patch for {path}, status={status}")
        logger.error(f"Failed to fetch file {path} at ref {ref}: {e}", exc_info=True)
        return {"error": f"Failed to fetch file: {str(e)}", "path": path, "ref": ref}


def fetch_pr_files_with_contents(
    org_id: str,
    repo_full_name: str,
    pr_number: int,
    which_ref: str = "head",
    include_binary: bool = False,
) -> Dict[str, Any]:
    """
    Fetch PR changed files and, for text files, fetch content at head/base ref using PyGithub.
    which_ref: 'head' or 'base'
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"fetch_pr_files_with_contents: org_id={org_id}, repo={repo_full_name}, pr_number={pr_number}, which_ref={which_ref}")
    
    # Step 1: Get PR branches and SHAs
    logger.debug("Step 1: Getting PR branches and SHAs...")
    meta = get_pr_branches_and_shas(org_id, repo_full_name, pr_number)
    if "error" in meta:
        logger.error(f"Failed to get PR branches/shas: {meta.get('error')}")
        return meta
    
    ref = meta.get("head_ref") if which_ref == "head" else meta.get("base_ref")
    ref_sha = meta.get("head_sha") if which_ref == "head" else meta.get("base_sha")
    logger.info(f"Using ref={ref} (sha={ref_sha}) for which_ref={which_ref}")
    
    # Step 2: Get changed files
    logger.debug("Step 2: Fetching PR changed files...")
    files = fetch_pr_changed_files(org_id, repo_full_name, pr_number)
    logger.info(f"Found {len(files)} changed files in PR")
    
    if not files:
        logger.warning("No changed files found in PR")
        return {"ref": ref, "ref_sha": ref_sha, "files": []}
    
    # Step 3: Fetch content for each file
    logger.debug("Step 3: Fetching file contents...")
    out: List[Dict[str, Any]] = []
    for idx, f in enumerate(files, 1):
        path = f.get("filename")
        logger.debug(f"Processing file {idx}/{len(files)}: {path}")
        
        # Skip likely binary by extension unless include_binary=True
        is_sql_like = path.lower().endswith((".sql", ".py", ".jinja", ".yml", ".yaml", ".md")) if path else False
        if not include_binary and not is_sql_like:
            logger.debug(f"Skipping binary file: {path}")
            out.append({**f, "content": None})
            continue
        
        if path and ref:
            # Pass patch and status to fetch_file_at_ref for fallback extraction
            fetched = fetch_file_at_ref(org_id, repo_full_name, path, ref, patch=f.get("patch"), status=f.get("status"))
            if "error" in fetched:
                logger.warning(f"Failed to fetch content for {path}: {fetched.get('error')}")
                out.append({**f, "content": None, "fetch_error": fetched.get("error")})
            else:
                content = fetched.get("content")
                content_size = len(content) if content else 0
                encoding = fetched.get("encoding", "unknown")
                logger.debug(f"Fetched {path}: {content_size} chars (encoding: {encoding})")
                out.append({**f, "content": content, "content_encoding": encoding})
        else:
            logger.warning(f"Missing path or ref for file: path={path}, ref={ref}")
            out.append({**f, "content": None})
    
    logger.info(f"Completed: fetched contents for {len([f for f in out if f.get('content')])}/{len(out)} files")
    return {"ref": ref, "ref_sha": ref_sha, "files": out}


def check_github_integration_status() -> Dict[str, Any]:
    """
    Diagnostic function to check GitHub integration status.
    Returns a dict with status information for debugging.
    """
    status = {
        "has_app_id": bool(GITHUB_APP_ID),
        "has_private_key": bool(PRIVATE_KEY),
        "has_git_integration": bool(git_integration),
        "app_id_value": GITHUB_APP_ID if GITHUB_APP_ID else None,
        "private_key_length": len(PRIVATE_KEY) if PRIVATE_KEY else 0,
    }
    if git_integration:
        status["integration_ready"] = True
    else:
        status["integration_ready"] = False
        if not GITHUB_APP_ID:
            status["error"] = "GITHUB_APP_ID not set"
        elif not PRIVATE_KEY:
            status["error"] = "GITHUB_PRIVATE_KEY not set"
        else:
            status["error"] = "Failed to initialize GithubIntegration (check PRIVATE_KEY format)"
    return status


__all__ = [
    "fetch_pr_analyses_for_org",
    "fetch_repo_tree",
    "build_org_pr_repo_tool",
    "get_pr_branches_and_shas",
    "fetch_pr_changed_files",
    "fetch_file_at_ref",
    "fetch_pr_files_with_contents",
    "check_github_integration_status",
]


