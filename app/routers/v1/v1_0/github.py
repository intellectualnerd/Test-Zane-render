# GitHub App Installation and Management Endpoints
# GET /github/install → redirect to GitHub App installation
# GET /github/callback → handle GitHub App installation callback
# GET /github/installations → list installations for organization
# GET /github/repositories/{installation_id} → list repositories for installation
# POST /github/webhook → handle GitHub webhook events (PR events)
# POST /github/process-pr → process PR changes and add comment
from fastapi import APIRouter, HTTPException, Depends, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from pydantic import BaseModel
import json
import requests
from app.database import get_db
from app.utils.models import GitHubInstallation, GitHubRepository, Organization, User, GitHubPullRequestAnalysis, ColumnLevelLineage
from app.utils.auth_deps import get_current_user
from app.utils.rbac import require_connector_access
import uuid
from uuid import UUID
from datetime import datetime
import os
import logging
from urllib.parse import unquote
from app.services.impact_analysis import schema_detection_rag, dbt_model_detection_rag, fetch_queries, store_pr_analysis, parse_schema_change
from github import GithubIntegration, Github
from sqlalchemy import and_, or_, func
from fastapi_versioning import version
router = APIRouter(prefix="/github", tags=["GitHub"])

# --- Configuration ---
# TODO: Update these values for your GitHub App
GITHUB_APP_URL = "https://github.com/apps/test-zane/"
CALLBACK_URL = "https://test-zane-render.onrender.com/api/v1.0/github/callback"
GITHUB_API_BASE = "https://api.github.com"
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
PRIVATE_KEY = os.getenv("GITHUB_PRIVATE_KEY")

logger = logging.getLogger("github")
logging.basicConfig(level=logging.INFO)

# GitHub App permissions and events
GITHUB_PERMISSIONS = {
    "contents": "read",
    "pull_requests": "read", 
    "metadata": "read"
}

GITHUB_EVENTS = [
    "pull_request"
]
git_integration: Optional[GithubIntegration] = None
if GITHUB_APP_ID and PRIVATE_KEY:
    try:
        git_integration = GithubIntegration(int(GITHUB_APP_ID), PRIVATE_KEY)
    except Exception:
        git_integration = None



# --- Models ---
class InstallationResponse(BaseModel):
    id: UUID
    installation_id: str
    account_type: str
    account_login: str
    repository_selection: str
    permissions: str | None
    events: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True

class RepositoryResponse(BaseModel):
    id: UUID
    repo_id: str
    repo_name: str
    full_name: str
    private: bool
    description: str | None
    default_branch: str | None
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True


class PRAAnalysisResponse(BaseModel):
    id: UUID
    org_id: UUID
    installation_id: UUID
    repository_id: UUID | None
    repo_full_name: str
    pr_number: int
    pr_title: str | None
    pr_description: str | None
    branch_name: str | None
    author_name: str | None
    pr_url: str | None
    total_impacted_queries: int | None
    analysis_data: dict
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True

class PRProcessRequest(BaseModel):
    installation_id: str
    repo_full_name: str
    pr_number: int
    pr_title: str
    pr_body: str | None = None


# --- Helpers ---
def get_github_installation_info(installation_id: str, access_token: str) -> dict:
    """Get installation information from GitHub API"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(f"{GITHUB_API_BASE}/app/installations/{installation_id}", headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to get installation info: {response.text}")
    
    return response.json()

def get_github_repositories(installation_id: str, access_token: str) -> List[dict]:
    """Get repositories for an installation from GitHub API"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(f"{GITHUB_API_BASE}/installation/repositories", headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to get repositories: {response.text}")
    
    data = response.json()
    return data.get("repositories", [])

def get_installation_access_token(installation_id: str, jwt_token: str) -> str:
    """Get installation access token using JWT"""
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.post(f"{GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens", headers=headers)
    if response.status_code != 201:
        raise HTTPException(status_code=400, detail=f"Failed to get access token: {response.text}")
    
    data = response.json()
    return data.get("token")

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature"""
    import hmac
    import hashlib
    
    if not signature.startswith("sha256="):
        return False
    
    expected_signature = signature[7:]  # Remove "sha256=" prefix
    calculated_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, calculated_signature)

def add_comment_to_pr(access_token: str, repo_full_name: str, pr_number: int, comment: str) -> bool:
    """Add a comment to a pull request"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {"body": comment}
    response = requests.post(
        f"{GITHUB_API_BASE}/repos/{repo_full_name}/issues/{pr_number}/comments",
        headers=headers,
        json=data
    )
    
    return response.status_code == 201


def extract_actual_sql_from_diff(sql_change: str) -> str:
    """
    Extract the actual SQL change from a diff format.
    Diff format example:
    File: file.sql (modified) [+1/-1]
    @@ -1 +1 @@
    -ALTER TABLE old.table DROP COLUMN col;
    +ALTER TABLE new.table DROP COLUMN col;
    
    Returns the line starting with '+' (the new change), or the original text if not a diff.
    """
    if not sql_change:
        return ""
    
    # Look for lines starting with '+' that contain ALTER TABLE
    lines = sql_change.split('\n')
    for line in lines:
        line_stripped = line.strip()
        # Check if it's a diff line with '+' and contains ALTER TABLE
        if line_stripped.startswith('+') and 'ALTER TABLE' in line_stripped.upper():
            # Remove the '+' prefix and return
            return line_stripped[1:].strip()
    
    # If no diff format found, return original (might be plain SQL)
    return sql_change


def get_upstream_lineage(
    db: Session,
    org_id: UUID,
    target_database: Optional[str],
    target_schema: Optional[str],
    target_table: str,
    target_column: Optional[str] = None
) -> List[dict]:
    """
    Query upstream lineage (what feeds into the target table/column).
    Returns all source columns that feed into the specified target.
    Uses case-insensitive matching for table/column names.
    """
    query = db.query(ColumnLevelLineage).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.target_table) == func.lower(target_table)
    )
    
    if target_database:
        query = query.filter(func.lower(ColumnLevelLineage.target_database) == func.lower(target_database))
    if target_schema:
        query = query.filter(func.lower(ColumnLevelLineage.target_schema) == func.lower(target_schema))
    if target_column:
        query = query.filter(func.lower(ColumnLevelLineage.target_column) == func.lower(target_column))
    
    lineage_records = query.all()
    
    result = []
    for record in lineage_records:
        result.append({
            "source_database": record.source_database,
            "source_schema": record.source_schema,
            "source_table": record.source_table,
            "source_column": record.source_column,
            "target_database": record.target_database,
            "target_schema": record.target_schema,
            "target_table": record.target_table,
            "target_column": record.target_column,
            "query_id": record.query_id,
            "query_type": record.query_type,
            "dbt_model_file_path": record.dbt_model_file_path,
        })
    
    return result


def get_downstream_lineage(
    db: Session,
    org_id: UUID,
    source_database: Optional[str],
    source_schema: Optional[str],
    source_table: str,
    source_column: Optional[str] = None
) -> List[dict]:
    """
    Query downstream lineage (what the source table/column feeds into).
    Returns all target columns that depend on the specified source.
    Uses case-insensitive matching for table/column names.
    """
    query = db.query(ColumnLevelLineage).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.source_table) == func.lower(source_table)
    )
    
    if source_database:
        query = query.filter(func.lower(ColumnLevelLineage.source_database) == func.lower(source_database))
    if source_schema:
        query = query.filter(func.lower(ColumnLevelLineage.source_schema) == func.lower(source_schema))
    if source_column:
        query = query.filter(func.lower(ColumnLevelLineage.source_column) == func.lower(source_column))
    
    lineage_records = query.all()
    
    result = []
    for record in lineage_records:
        result.append({
            "source_database": record.source_database,
            "source_schema": record.source_schema,
            "source_table": record.source_table,
            "source_column": record.source_column,
            "target_database": record.target_database,
            "target_schema": record.target_schema,
            "target_table": record.target_table,
            "target_column": record.target_column,
            "query_id": record.query_id,
            "query_type": record.query_type,
            "dbt_model_file_path": record.dbt_model_file_path,
        })
    
    return result


def get_recursive_downstream_lineage(
    db: Session,
    org_id: UUID,
    source_database: Optional[str],
    source_schema: Optional[str],
    source_table: str,
    source_column: Optional[str] = None,
    max_depth: int = 10,
    visited_nodes: Optional[set] = None,
    seen_records: Optional[set] = None
) -> List[dict]:
    """
    Recursively query downstream lineage (multi-hop dependencies).
    Traverses the lineage graph to find all downstream dependencies at any depth.
    Uses case-insensitive matching for table/column names.
    Deduplicates records to prevent the same lineage relationship from appearing multiple times.
    
    Args:
        db: Database session
        org_id: Organization ID
        source_database: Source database name (optional)
        source_schema: Source schema name (optional)
        source_table: Source table name
        source_column: Source column name (optional)
        max_depth: Maximum depth to traverse (default: 10)
        visited_nodes: Set of visited (table, column) tuples to prevent cycles
        seen_records: Set of seen lineage record keys to prevent duplicates
    
    Returns:
        List of all downstream lineage records (all depths combined, deduplicated)
    """
    if visited_nodes is None:
        visited_nodes = set()
    if seen_records is None:
        seen_records = set()
    
    if max_depth <= 0:
        return []
    
    # Create a key for this node to track visited nodes (prevents cycles)
    node_key = (
        (source_database or "").lower(),
        (source_schema or "").lower(),
        source_table.lower(),
        (source_column or "").lower()
    )
    
    if node_key in visited_nodes:
        # Cycle detected, return empty to prevent infinite recursion
        return []
    
    visited_nodes.add(node_key)
    
    # Get direct downstream dependencies (1-hop)
    direct_downstream = get_downstream_lineage(
        db=db,
        org_id=org_id,
        source_database=source_database,
        source_schema=source_schema,
        source_table=source_table,
        source_column=source_column
    )
    
    # Filter and deduplicate direct downstream records
    unique_direct_downstream = []
    for record in direct_downstream:
        # Create a unique key for this lineage record
        # Use 'or ""' to handle None values (dict.get returns None if key exists with None value)
        record_key = (
            (record.get("source_database") or "").lower(),
            (record.get("source_schema") or "").lower(),
            (record.get("source_table") or "").lower(),
            (record.get("source_column") or "").lower(),
            (record.get("target_database") or "").lower(),
            (record.get("target_schema") or "").lower(),
            (record.get("target_table") or "").lower(),
            (record.get("target_column") or "").lower(),
        )
        
        # Only add if we haven't seen this exact lineage relationship before
        if record_key not in seen_records:
            seen_records.add(record_key)
            unique_direct_downstream.append(record)
    
    # Start with unique direct downstream records
    all_downstream = unique_direct_downstream.copy()
    
    # For each unique target table, recursively get its downstream dependencies
    # Group by target table to avoid redundant recursive calls
    target_tables_seen = set()
    for record in unique_direct_downstream:
        target_db = record.get("target_database")
        target_schema = record.get("target_schema")
        target_table = record.get("target_table")
        
        # Skip if target_table is None (can't recurse on None table)
        if not target_table:
            continue
        
        # Create a key for this target table
        target_table_key = (
            (target_db or "").lower(),
            (target_schema or "").lower(),
            target_table.lower()
        )
        
        # Only recursively query each target table once (not per column)
        # This ensures we get ALL downstream dependencies of the table, not just specific columns
        if target_table_key not in target_tables_seen:
            target_tables_seen.add(target_table_key)
            
            # Recursively get downstream for this target table
            # Pass source_column=None to get ALL columns of this table's downstream dependencies
            # Pass the same seen_records set to ensure deduplication across all recursive calls
            nested_downstream = get_recursive_downstream_lineage(
                db=db,
                org_id=org_id,
                source_database=target_db,
                source_schema=target_schema,
                source_table=target_table,
                source_column=None,  # Query ALL columns to get complete downstream
                max_depth=max_depth - 1,
                visited_nodes=visited_nodes.copy(),  # Copy for cycle detection (allows different paths)
                seen_records=seen_records  # Share the same set to prevent duplicates
            )
            
            # Add nested downstream results (already deduplicated by seen_records)
            all_downstream.extend(nested_downstream)
    
    return all_downstream


# --- Endpoints ---
@router.get("/install")
@version(1, 0)
def github_install(org_id: str, request: Request):
    """Redirect to GitHub App installation with org_id as state"""
    try:
        # Validate org_id format
        uuid.UUID(org_id)
    except ValueError:
        logger.warning("/github/install - invalid org_id: %s", org_id)
        raise HTTPException(status_code=400, detail="Invalid organization ID format")
    
    # Build GitHub App installation URL with state parameter
    install_url = f"{GITHUB_APP_URL}/installations/new?state={org_id}"
    # https://github.com/apps/queryguardai-poc/installations/new?state=/76d33fb3-6062-456b-a211-4aec9971f8be
    logger.info("/github/install - redirecting to %s", install_url)
    return RedirectResponse(url=install_url)


@router.get("/callback")
@version(1, 0)
def github_callback(
    installation_id: Optional[str] = None,
    setup_action: Optional[str] = None,
    state: Optional[str] = None,
    code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Handle GitHub App installation callback"""
    logger.info("GitHub callback received installation_id=%s setup_action=%s state=%s code=%s", installation_id, setup_action, state, code)

    # Handle setup_action=request (pending approval) - GitHub doesn't send installation_id yet
    if setup_action == "request":
        logger.info("/github/callback - installation request pending approval (setup_action=request)")
        if not state:
            return {"message": "Installation request received but no state parameter - cannot associate with organization"}
        
        try:
            # Validate state (org_id) format
            normalized_state = unquote(state)
            if normalized_state.startswith('/'):
                normalized_state = normalized_state[1:]
            org_uuid = uuid.UUID(normalized_state)
            
            # Check if organization exists
            organization = db.query(Organization).filter(
                Organization.id == org_uuid,
                Organization.is_active == True
            ).first()

            if not organization:
                logger.warning("/github/callback - organization not found for pending request: %s", org_uuid)
                return {"message": "Organization not found", "status": "error"}
            
            return {
                "message": "GitHub App installation request received. Waiting for organization administrator approval.",
                "org_id": str(org_uuid),
                "status": "pending_approval",
                "setup_action": "request"
            }
        except ValueError:
            logger.warning("/github/callback - invalid state format in request: %s", state)
            return {"message": "Invalid state parameter", "status": "error"}
    
    # For setup_action=install or when installation_id is present, proceed with installation
    if not installation_id:
        logger.warning("/github/callback - missing installation_id and setup_action is not 'request'")
        return {
            "message": "Installation ID is required for installation completion",
            "status": "error"
        }

    # If no state parameter, ignore the installation (not from our flow)
    if not state:
        return {"message": "Installation ignored - no state parameter"}
    
    try:
        # Decode and normalize state; GitHub may send it URL-encoded and with a leading '/'
        normalized_state = unquote(state)
        if normalized_state.startswith('/'):
            normalized_state = normalized_state[1:]
        # Validate state (org_id) format
        org_uuid = uuid.UUID(normalized_state)
    except ValueError:
        logger.warning("/github/callback - invalid state format: %s", state)
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    # Check if organization exists
    organization = db.query(Organization).filter(
        Organization.id == org_uuid,
        Organization.is_active == True
    ).first()
    
    if not organization:
        logger.warning("/github/callback - organization not found: %s", org_uuid)
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Check if installation already exists
    existing_installation = db.query(GitHubInstallation).filter(
        GitHubInstallation.installation_id == installation_id
    ).first()
    
    if existing_installation:
        logger.info("/github/callback - installation already exists: %s", installation_id)
        raise HTTPException(status_code=400, detail="Installation already exists")
    
    try:
        # TODO: You'll need to implement JWT token generation for your GitHub App
        # For now, we'll store basic installation info
        # In production, you'd use the GitHub API to get full installation details
        
        new_installation = GitHubInstallation(
            installation_id=installation_id,
            org_id=org_uuid,
            account_type="Organization",  # Will be updated with actual data
            account_login="",  # Will be updated with actual data
            repository_selection="all",  # Will be updated with actual data
            permissions=json.dumps(GITHUB_PERMISSIONS),
            events=json.dumps(GITHUB_EVENTS)
        )
        
        db.add(new_installation)
        db.commit()
        db.refresh(new_installation)
        logger.info("/github/callback - installation saved id=%s org_id=%s", new_installation.id, org_uuid)

        # Best-effort: immediately sync repositories for this installation
        try:
            if git_integration:
                access_token = git_integration.get_access_token(int(installation_id)).token
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github.v3+json",
                }
                repos_resp = requests.get(f"{GITHUB_API_BASE}/installation/repositories", headers=headers)
                if repos_resp.status_code == 200:
                    repos_data = repos_resp.json().get("repositories", [])

                    existing_repos = db.query(GitHubRepository).filter(
                        GitHubRepository.installation_id == new_installation.id
                    ).all()
                    existing_by_repo_id = {r.repo_id: r for r in existing_repos}

                    seen_repo_ids: set[str] = set()
                    for r in repos_data:
                        repo_id = str(r.get("id"))
                        seen_repo_ids.add(repo_id)
                        repo_obj = existing_by_repo_id.get(repo_id)
                        if not repo_obj:
                            repo_obj = GitHubRepository(
                                installation_id=new_installation.id,
                                repo_id=repo_id,
                                repo_name=r.get("name") or "",
                                full_name=r.get("full_name") or "",
                                private=bool(r.get("private")),
                                description=r.get("description"),
                                default_branch=r.get("default_branch"),
                            )
                            db.add(repo_obj)
                        else:
                            repo_obj.repo_name = r.get("name") or repo_obj.repo_name
                            repo_obj.full_name = r.get("full_name") or repo_obj.full_name
                            repo_obj.private = bool(r.get("private"))
                            repo_obj.description = r.get("description")
                            repo_obj.default_branch = r.get("default_branch")

                    db.commit()
                else:
                    logger.warning(
                        "/github/callback - failed to fetch repositories for installation %s: %s",
                        installation_id,
                        repos_resp.text,
                    )
            else:
                logger.info("/github/callback - GitHub Integration not configured; skipping repo sync")
        except Exception:
            logger.exception("/github/callback - error while syncing repositories for installation %s", installation_id)
        
        # Optional: redirect to your frontend success page if CALLBACK_URL is set
        if CALLBACK_URL:
            # Avoid redirecting back to the same callback endpoint to prevent a second hit
            if CALLBACK_URL.rstrip("/").endswith("/github/callback"):
                return {
                    "message": "GitHub installation saved",
                    "org_id": str(org_uuid),
                    "installation_id": installation_id,
                    "status": "success",
                }
            # Append org_id and installation_id for UI to consume
            redirect_url = f"{CALLBACK_URL}?org_id={org_uuid}&installation_id={installation_id}&status=success"
            return RedirectResponse(url=str(redirect_url))

        return new_installation
        
    except IntegrityError:
        db.rollback()
        logger.exception("/github/callback - failed to save installation")
        raise HTTPException(status_code=400, detail="Failed to save installation")


@router.get("/installations", response_model=List[InstallationResponse])
@version(1, 0)
def list_installations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all GitHub installations for the organization"""
    installations = db.query(GitHubInstallation).filter(
        GitHubInstallation.org_id == current_user.org_id,
        GitHubInstallation.is_active == True
    ).all()
    
    return installations


@router.get("/repositories/{installation_id}", response_model=List[RepositoryResponse])
@version(1, 0)
def list_repositories(installation_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List repositories for a specific installation"""
    try:
        inst_uuid = uuid.UUID(installation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid installation ID format")
    
    # Verify installation belongs to user's org
    installation = db.query(GitHubInstallation).filter(
        GitHubInstallation.id == inst_uuid,
        GitHubInstallation.org_id == current_user.org_id,
        GitHubInstallation.is_active == True
    ).first()
    
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub installation not found")
    
    repositories = db.query(GitHubRepository).filter(
        GitHubRepository.installation_id == inst_uuid
    ).all()
    
    return repositories


@router.post("/sync-repositories/{installation_id}")
@version(1, 0)
def sync_repositories(installation_id: str, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db)):
    """Sync repositories for an installation (manual trigger)"""
    try:
        inst_uuid = uuid.UUID(installation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid installation ID format")
    
    # Verify installation belongs to user's org
    installation = db.query(GitHubInstallation).filter(
        GitHubInstallation.id == inst_uuid,
        GitHubInstallation.org_id == current_user.org_id,
        GitHubInstallation.is_active == True
    ).first()
    
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub installation not found")
    
    if not git_integration:
        raise HTTPException(status_code=400, detail="GitHub Integration not configured")

    try:
        access_token = git_integration.get_access_token(int(installation.installation_id)).token
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(f"{GITHUB_API_BASE}/installation/repositories", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to get repositories: {response.text}")

        repos = response.json().get("repositories", [])

        existing_repos = db.query(GitHubRepository).filter(
            GitHubRepository.installation_id == installation.id
        ).all()
        existing_by_repo_id = {r.repo_id: r for r in existing_repos}

        seen_repo_ids: set[str] = set()
        created_count = 0
        updated_count = 0

        for r in repos:
            repo_id = str(r.get("id"))
            seen_repo_ids.add(repo_id)
            repo_obj = existing_by_repo_id.get(repo_id)
            if not repo_obj:
                repo_obj = GitHubRepository(
                    installation_id=installation.id,
                    repo_id=repo_id,
                    repo_name=r.get("name") or "",
                    full_name=r.get("full_name") or "",
                    private=bool(r.get("private")),
                    description=r.get("description"),
                    default_branch=r.get("default_branch"),
                )
                db.add(repo_obj)
                created_count += 1
            else:
                repo_obj.repo_name = r.get("name") or repo_obj.repo_name
                repo_obj.full_name = r.get("full_name") or repo_obj.full_name
                repo_obj.private = bool(r.get("private"))
                repo_obj.description = r.get("description")
                repo_obj.default_branch = r.get("default_branch")
                updated_count += 1

        db.commit()

        return {
            "message": "Repository sync completed",
            "installation_id": installation_id,
            "created": created_count,
            "updated": updated_count,
            "total": len(repos),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/github/sync-repositories - error: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to sync repositories")


@router.delete("/installations/{installation_id}")
@version(1, 0)
def deactivate_installation(installation_id: str, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db)):
    """Deactivate a GitHub installation (soft delete)"""
    try:
        inst_uuid = uuid.UUID(installation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid installation ID format")
    
    # Verify installation belongs to user's org
    installation = db.query(GitHubInstallation).filter(
        GitHubInstallation.id == inst_uuid,
        GitHubInstallation.org_id == current_user.org_id,
        GitHubInstallation.is_active == True
    ).first()
    
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub installation not found")
    
    installation.is_active = False
    db.commit()
    
    return {"message": "GitHub installation deactivated successfully"}


def extract_added_lines_from_patch(patch: str) -> str:
    """
    Extract only the added lines (lines starting with '+') from a GitHub diff patch.
    Removes the '+' prefix and returns the new content.
    
    Args:
        patch: The diff patch string from GitHub
        
    Returns:
        String containing only the added lines (without the '+' prefix)
    """
    if not patch:
        return ""
    
    added_lines = []
    for line in patch.split('\n'):
        # Lines starting with '+' are added lines (but not the hunk header which is @@)
        if line.startswith('+') and not line.startswith('+++'):
            # Remove the '+' prefix and add the line
            added_lines.append(line[1:])
    
    return '\n'.join(added_lines)


@router.post("/webhook")
@version(1, 0)
async def github_webhook(request: Request, db=Depends(get_db)):
    """Handle GitHub webhook events (PR events)"""
    # Get the raw body for signature verification
    body = await request.body()
    
    # Get GitHub signature header
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        logger.warning("/github/webhook - missing signature header")
        raise HTTPException(status_code=401, detail="Missing signature")
    
    # Verify webhook signature
    if not verify_webhook_signature(body, signature):
        logger.warning("/github/webhook - invalid signature")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Extract headers and parse payload early so we can log for any event
    event_type = request.headers.get("X-GitHub-Event")
    delivery_id = request.headers.get("X-GitHub-Delivery")

    # Parse the webhook payload
    try:
        payload = json.loads(body.decode())
    except json.JSONDecodeError:
        logger.warning("/github/webhook - invalid JSON payload")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Log compact info for observability
    try:
        payload_preview = str(payload)[:2000]
        logger.info(
            "/github/webhook - delivery=%s event=%s payload_preview=%s",
            delivery_id,
            event_type,
            payload_preview,
        )
    except Exception:
        logger.debug("/github/webhook - failed to log payload preview")

    # If not interested in this event type, return early
    if event_type != "pull_request":
        logger.info("/github/webhook - ignoring event %s", event_type)
        return {"message": f"Ignoring {event_type} event"}
    
    # Extract PR information
    action = payload.get("action")
    if action not in ["opened", "reopened", "synchronize"]:
        logger.info("/github/webhook - ignoring PR action %s", action)
        return {"message": f"Ignoring PR {action} action"}
    
    pr_data = payload.get("pull_request", {})
    repo_data = payload.get("repository", {})
    installation_data = payload.get("installation", {})
    
    # Extract relevant information
    pr_number = pr_data.get("number")
    pr_title = pr_data.get("title")
    pr_body = pr_data.get("body")
    branch_name = pr_data.get("head", {}).get("ref")  # Branch name from PR head
    author_name = pr_data.get("user", {}).get("login")  # PR author username
    pr_url = pr_data.get("html_url")  # GitHub PR URL
    repo_full_name = repo_data.get("full_name")
    installation_id = str(installation_data.get("id"))

    # Validate installation id exists and is active in our DB
    if not installation_id:
        logger.info("/github/webhook - missing installation id in payload")
        return {"message": "Missing installation id"}

    installation = db.query(GitHubInstallation).filter(
        GitHubInstallation.installation_id == installation_id,
        GitHubInstallation.is_active == True
    ).first()

    if not installation:
        logger.info("/github/webhook - installation id %s not registered/active, ignoring", installation_id)
        return {"message": "Installation not recognized or inactive"}
    
    # Use GitHub App installation token and PyGithub to fetch files and post comment
    if not git_integration:
        logger.warning("/github/webhook - GitHub Integration not configured")
        return {"message": "GitHub Integration not configured"}

    try:
        access_token = git_integration.get_access_token(int(installation_id)).token
        gh = Github(login_or_token=access_token)
        repo = gh.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)

        # Collect relevant SQL file changes
        code_changes = []
        for file in pr.get_files():
            if getattr(file, "patch", None) and file.filename.lower().endswith(".sql"):
                code_changes.append(
                    {
                        "filename": file.filename,
                        "status": file.status,
                        "patch": file.patch,
                        "additions": file.additions,
                        "deletions": file.deletions,
                    }
                )

        if not code_changes:
            logger.info("/github/webhook - no relevant SQL changes found")
            return {"message": "No relevant SQL changes found"}

        # Analyze each SQL change
        results = []
        for c in code_changes:
            # Extract only the added lines (new content) from the patch
            added_content = extract_added_lines_from_patch(c.get("patch", ""))
            
            # Create full_diff for logging/storage (includes full patch)
            full_diff = (
                f"File: {c['filename']} ({c['status']}) [+{c['additions']}/-{c['deletions']}]\n{c['patch']}"
            )
            
            # Skip analysis if there are no added lines (only deletions)
            if not added_content.strip():
                logger.info("/github/webhook - no added content in %s, skipping analysis", c['filename'])
                results.append(
                    {
                        "sql_change": full_diff,
                        "impact_analysis": "No new content to analyze (file contains only deletions).",
                        "affected_query_ids": [],
                        "regression_queries": [],
                        "source_metadata": [],
                    }
                )
                continue
            
            # For analysis, use only the added content (new changes)
            # Format it similar to full_diff but with only new content
            analysis_input = (
                f"File: {c['filename']} ({c['status']}) [+{c['additions']}/-{c['deletions']}]\n"
                f"New changes:\n{added_content}"
            )
            
            if c["filename"].endswith(".sql") and "models/" in c["filename"]:
                analysis_result = dbt_model_detection_rag(analysis_input, c["filename"], str(installation.org_id), db)  # DBT model path
            else:
                analysis_result = schema_detection_rag(analysis_input, str(installation.org_id))

            regression_queries = fetch_queries(analysis_result.get("affected_query_ids", []))

            results.append(
                {
                    "sql_change": full_diff,
                    "impact_analysis": analysis_result.get("impact_report", ""),
                    "affected_query_ids": analysis_result.get("affected_query_ids", []),
                    "regression_queries": regression_queries,
                    "source_metadata": analysis_result.get("source_metadata", []),
                }
            )

        # Compose comment
        file_summaries = []
        for idx, r in enumerate(results, start=1):
            file_info = code_changes[idx - 1]
            file_summaries.append(
                f"""
<details>
<summary>📂 **{file_info['filename']}** ({file_info['status']}) [+{file_info['additions']}/-{file_info['deletions']}]
</summary>

**Impact Report:**
{r['impact_analysis']}

**Affected Query IDs:** {', '.join(r['affected_query_ids']) if r['affected_query_ids'] else 'None'}

</details>
"""
            )

        comment_text = f"## 🤖 **Impact Analysis Summary**\n\nAnalyzed {len(results)} SQL file(s) for potential downstream impact.\n\n*Analysis generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*\n\n---\n\n{chr(10).join(file_summaries)}"

        # Calculate total impacted queries (unique query IDs across all files)
        all_affected_query_ids = set()
        for r in results:
            all_affected_query_ids.update(r.get("affected_query_ids", []))
        total_impacted_queries = len(all_affected_query_ids)

        # Store results
        # New: store results via SQLAlchemy model with relationships
        analysis_id = store_pr_analysis(
            db,
            org_id=str(installation.org_id),
            installation_id_str=installation_id,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            pr_title=pr_title,
            pr_description=pr_body,
            branch_name=branch_name,
            author_name=author_name,
            pr_url=pr_url,
            total_impacted_queries=total_impacted_queries,
            analysis_data={"files": results},
        )

        issue = repo.get_issue(number=pr_number)
        issue.create_comment(comment_text)

        logger.info("/github/webhook - posted analysis comment to PR #%s", pr_number)

        return {
            "message": "PR webhook processed",
            "pr_number": pr_number,
            "repo": repo_full_name,
            "installation_id": installation_id,
            "action": action,
            "files_analyzed": len(results),
            "analysis_id": analysis_id,
        }

    except Exception as e:
        logger.exception("/github/webhook - analysis failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Webhook processing failed")


@router.post("/process-pr")
@version(1, 0)
def process_pr_changes(pr_request: PRProcessRequest, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db)):
    """Process PR changes and add comment to PR"""
    try:
        inst_uuid = uuid.UUID(pr_request.installation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid installation ID format")
    
    # Verify installation belongs to user's org
    installation = db.query(GitHubInstallation).filter(
        GitHubInstallation.id == inst_uuid,
        GitHubInstallation.org_id == current_user.org_id,
        GitHubInstallation.is_active == True
    ).first()
    
    if not installation:
        raise HTTPException(status_code=404, detail="GitHub installation not found")
    
    try:
        # TODO: Implement JWT token generation for your GitHub App
        # For now, we'll use a placeholder access token
        # In production, you'd generate a JWT token and get an access token
        
        # Placeholder for access token (replace with actual implementation)
        access_token = "placeholder_access_token"
        
        # Add comment to PR
        comment = "Changes Processed By Query Guard AI"
        success = add_comment_to_pr(
            access_token=access_token,
            repo_full_name=pr_request.repo_full_name,
            pr_number=pr_request.pr_number,
            comment=comment
        )
        
        if success:
            return {
                "message": "PR processed successfully",
                "comment_added": True,
                "pr_number": pr_request.pr_number,
                "repo": pr_request.repo_full_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add comment to PR")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PR: {str(e)}")


@router.get("/analyses/{analysis_id}", response_model=PRAAnalysisResponse)
@version(1, 0)
def get_pr_analysis(analysis_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Fetch stored PR analysis by its UUID with enhanced lineage data"""
    try:
        analysis_uuid = uuid.UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid analysis ID format")

    analysis = db.query(GitHubPullRequestAnalysis).filter(
        GitHubPullRequestAnalysis.id == analysis_uuid
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Org scoping: the analysis must belong to the user's org
    if analysis.org_id != current_user.org_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Enhance analysis_data with lineage information
    if analysis.analysis_data and "files" in analysis.analysis_data:
        enhanced_files = []
        
        for file_data in analysis.analysis_data["files"]:
            enhanced_file = file_data.copy()
            
            # Parse SQL change to extract table and column information
            sql_change = file_data.get("sql_change", "")
            # Extract actual SQL from diff format (if it's a diff)
            actual_sql = extract_actual_sql_from_diff(sql_change)
            parsed_change = parse_schema_change(actual_sql)
            
            # Extract table and column info
            changed_table = parsed_change.get("table_name")
            changed_column = parsed_change.get("column_name")
            changed_database = parsed_change.get("database")
            changed_schema = parsed_change.get("schema")
            
            # Enhance source_metadata with source information and is_impacted flag
            source_metadata = file_data.get("source_metadata", [])
            enhanced_source_metadata = []
            
            # Get the changed column name (normalized)
            changed_column_lower = changed_column.lower() if changed_column else None
            
            for metadata_entry in source_metadata:
                enhanced_entry = metadata_entry.copy()
                
                # Try to find the actual source from lineage table by matching the target
                target_db = metadata_entry.get("target_database")
                target_schema = metadata_entry.get("target_schema")
                target_table = metadata_entry.get("target_table")
                target_column = metadata_entry.get("target_column")
                
                # Query lineage to find the ACTUAL DIRECT source for this target
                # IMPORTANT: Do NOT filter by changed_table/changed_column here, as this target
                # may be at depth 2+ and its direct source is an intermediate table, not the changed table.
                # We want to find the actual direct dependency chain, not force it to match the changed table.
                lineage_match = db.query(ColumnLevelLineage).filter(
                    ColumnLevelLineage.org_id == analysis.org_id,
                    ColumnLevelLineage.is_active == 1,
                    func.lower(ColumnLevelLineage.target_table) == func.lower(target_table),
                    func.lower(ColumnLevelLineage.target_column) == func.lower(target_column)
                )
                
                if target_db:
                    lineage_match = lineage_match.filter(
                        func.lower(ColumnLevelLineage.target_database) == func.lower(target_db)
                    )
                if target_schema:
                    lineage_match = lineage_match.filter(
                        func.lower(ColumnLevelLineage.target_schema) == func.lower(target_schema)
                    )
                
                # Get the first matching record (the direct source for this target)
                # Note: If there are multiple sources for the same target, we take the first one.
                # In practice, there should typically be one direct source per target column.
                lineage_record = lineage_match.first()
                
                if lineage_record:
                    # Use actual source from lineage table
                    enhanced_entry["source_database"] = lineage_record.source_database
                    enhanced_entry["source_schema"] = lineage_record.source_schema
                    enhanced_entry["source_table"] = lineage_record.source_table
                    enhanced_entry["source_column"] = lineage_record.source_column
                else:
                    # Fallback to parsed change values if no lineage match found
                    enhanced_entry["source_database"] = changed_database
                    enhanced_entry["source_schema"] = changed_schema
                    enhanced_entry["source_table"] = changed_table
                    enhanced_entry["source_column"] = changed_column
                
                # Flag entries where the target column is directly impacted
                enhanced_entry["is_impacted"] = True
                enhanced_source_metadata.append(enhanced_entry)
            
            enhanced_file["source_metadata"] = enhanced_source_metadata
            
            # Get complete lineage data if we have table information
            complete_lineage = {
                "upstream": [],
                "downstream": [],
                "center_node": None
            }
            
            if changed_table:
                # Set center node - the table/column that has the change (starting point for lineage visualization)
                center_node = {
                    "database": changed_database,
                    "schema": changed_schema,
                    "table": changed_table,
                    "column": changed_column
                }
                complete_lineage["center_node"] = center_node
                
                # Get upstream lineage (what feeds into the changed table)
                # Query ALL columns for this table (not just the changed column)
                upstream_lineage = get_upstream_lineage(
                    db=db,
                    org_id=analysis.org_id,
                    target_database=None,  # Query all databases for this table
                    target_schema=None,   # Query all schemas for this table
                    target_table=changed_table,
                    target_column=None  # Get ALL columns, not just the changed one
                )
                complete_lineage["upstream"] = upstream_lineage
                
                # Get downstream lineage (what the changed table feeds into)
                # Use recursive function to get multi-hop downstream dependencies
                # Query ALL columns for this table (not just the changed column)
                downstream_lineage = get_recursive_downstream_lineage(
                    db=db,
                    org_id=analysis.org_id,
                    source_database=None,  # Query all databases for this table
                    source_schema=None,   # Query all schemas for this table
                    source_table=changed_table,
                    source_column=None  # Get ALL columns, not just the changed one
                )
                complete_lineage["downstream"] = downstream_lineage
            
            enhanced_file["complete_lineage"] = complete_lineage
            enhanced_files.append(enhanced_file)
        
        # Create enhanced analysis_data
        enhanced_analysis_data = analysis.analysis_data.copy()
        enhanced_analysis_data["files"] = enhanced_files
        
        # Create a new analysis object with enhanced data
        # We need to convert to dict, modify, and return
        analysis_dict = {
            "id": analysis.id,
            "org_id": analysis.org_id,
            "installation_id": analysis.installation_id,
            "repository_id": analysis.repository_id,
            "repo_full_name": analysis.repo_full_name,
            "pr_number": analysis.pr_number,
            "pr_title": analysis.pr_title,
            "pr_description": analysis.pr_description,
            "branch_name": analysis.branch_name,
            "author_name": analysis.author_name,
            "pr_url": analysis.pr_url,
            "total_impacted_queries": analysis.total_impacted_queries,
            "analysis_data": enhanced_analysis_data,
            "created_at": analysis.created_at,
            "updated_at": analysis.updated_at,
        }
        
        # Return using the response model
        return PRAAnalysisResponse(**analysis_dict)
    
    return analysis
