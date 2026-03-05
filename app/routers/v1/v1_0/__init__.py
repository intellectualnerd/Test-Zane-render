from fastapi import APIRouter

from .auth import router as auth_router
from .chat import router as chat_router
from .dbt_cloud import router as dbt_cloud_router
from .github import router as github_router
from .impact import router as impact_router
from .jira import router as jira_router
from .organizations import router as organizations_router
from .overview_dashboard import router as overview_dashboard_router
from .snowflake import router as snowflake_router
from .users import router as user_router

router = APIRouter()

router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(dbt_cloud_router)
router.include_router(github_router)
router.include_router(impact_router)
router.include_router(jira_router)
router.include_router(organizations_router)
router.include_router(overview_dashboard_router)
router.include_router(snowflake_router)
router.include_router(user_router)