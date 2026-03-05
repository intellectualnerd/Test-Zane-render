from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from uuid import UUID
from app.utils.auth_deps import get_current_user
from app.utils.models import User, GitHubPullRequestAnalysis
from app.database import get_db
import logging
from fastapi_versioning import version

router = APIRouter(prefix="/overview-dashboard", tags=["Overview Dashboard"])
logger = logging.getLogger("overview_dashboard")

# --- Response Models ---
class PullRequestSummary(BaseModel):
    analysis_id: UUID
    pr_id: str
    title: str
    description: Optional[str] = ""
    branch_name: Optional[str] = ""
    repository_name: str
    author_name: Optional[str] = ""
    submitted_at: str
    total_impacted_queries: int


class DashboardSummary(BaseModel):
    total_prs: int
    impacted_queries: int


class DashboardOverviewResponse(BaseModel):
    summary: DashboardSummary
    pull_requests: List[PullRequestSummary]


# --- Endpoints ---
@router.get("/", response_model=DashboardOverviewResponse)
@version(1, 0)
def get_dashboard_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Get dashboard overview data with KPIs and pull request information.
    Accessible by any authenticated user (any role).
    """
    logger.info("/overview-dashboard - request by user_id=%s username=%s", 
                current_user.id, current_user.username)
    
    # Query PR analyses for the user's organization, ordered by created_at (newest first)
    pr_analyses = db.query(GitHubPullRequestAnalysis).filter(
        GitHubPullRequestAnalysis.org_id == current_user.org_id
    ).order_by(GitHubPullRequestAnalysis.created_at.desc()).all()
    
    # Calculate summary statistics
    total_prs = len(pr_analyses)
    total_impacted_queries = sum(
        pr.total_impacted_queries or 0 
        for pr in pr_analyses 
        if pr.total_impacted_queries is not None
    )
    
    # If total_impacted_queries is not stored, calculate from analysis_data
    if total_impacted_queries == 0:
        all_query_ids = set()
        for pr in pr_analyses:
            if pr.analysis_data and isinstance(pr.analysis_data, dict):
                files = pr.analysis_data.get("files", [])
                for file_data in files:
                    affected_ids = file_data.get("affected_query_ids", [])
                    all_query_ids.update(affected_ids)
        total_impacted_queries = len(all_query_ids)
    
    # Convert PR analyses to response format
    pull_requests = []
    for pr in pr_analyses:
        # Extract repository name from repo_full_name (format: "owner/repo")
        repo_name = pr.repo_full_name.split("/")[-1] if pr.repo_full_name else ""
        
        # Calculate total_impacted_queries if not stored
        pr_impacted_queries = pr.total_impacted_queries
        if pr_impacted_queries is None:
            pr_query_ids = set()
            if pr.analysis_data and isinstance(pr.analysis_data, dict):
                files = pr.analysis_data.get("files", [])
                for file_data in files:
                    affected_ids = file_data.get("affected_query_ids", [])
                    pr_query_ids.update(affected_ids)
            pr_impacted_queries = len(pr_query_ids)
        
        pull_requests.append(
            PullRequestSummary(
                analysis_id=pr.id,
                pr_id=f"PR-{pr.pr_number}",
                title=pr.pr_title or "",
                description=pr.pr_description or "",
                branch_name=pr.branch_name or "",
                repository_name=repo_name,
                author_name=pr.author_name or "",
                submitted_at=pr.created_at.isoformat() if pr.created_at else "",
                total_impacted_queries=pr_impacted_queries
            )
        )
    
    # Convert to response model
    response = DashboardOverviewResponse(
        summary=DashboardSummary(
            total_prs=total_prs,
            impacted_queries=total_impacted_queries
        ),
        pull_requests=pull_requests
    )
    
    logger.info("/overview-dashboard - returning data with %d PRs", 
               len(response.pull_requests))
    return response

