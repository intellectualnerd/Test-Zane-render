from fastapi import APIRouter, HTTPException, Depends, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List
from pydantic import BaseModel, HttpUrl
import requests
from app.database import get_db
from app.utils.models import DbtCloudConnection, DbtJob, User
from app.utils.auth_deps import get_current_user
from app.utils.rbac import require_connector_access
import logging
from app.services.dbt_crawler import sync_dbt_metadata
from fastapi_versioning import version

router = APIRouter(prefix="/dbt-cloud", tags=["dbt Cloud"])
logger = logging.getLogger("dbt_cloud")


class DbtCloudConnRequest(BaseModel):
    connection_name: str
    api_key: str
    account_id: str
    base_url: HttpUrl


class DbtCloudConnectionResponse(BaseModel):
    from uuid import UUID as _UUID  # local import to avoid top-level shadowing
    id: _UUID
    org_id: _UUID
    connection_name: str
    account_id: str
    base_url: str
    is_active: bool

    class Config:
        from_attributes = True


class DbtScheduleRequest(BaseModel):
    cron_expression: str


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _test_dbt_cloud_connection(api_key: str, account_id: str, base_url: str) -> dict:
    """Attempt to validate connectivity to dbt Cloud by fetching account info.

    Tries common API base path variants and auth header schemes and reports
    the last error details if all attempts fail.
    """
    normalized = _normalize_base_url(base_url)
    paths = [
        f"{normalized}/api/v3/accounts/{account_id}",
        f"{normalized}/v3/accounts/{account_id}",
        f"{normalized}/api/v2/accounts/{account_id}",
        f"{normalized}/v2/accounts/{account_id}",
    ]
    header_variants = [
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
    ]

    last_status = None
    last_body = None
    last_url = None
    last_auth = None

    for url in paths:
        for headers in header_variants:
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                logger.info("dbt test %s -> %s", url, resp.status_code)
                if resp.status_code == 200:
                    return {"ok": True, "url": url, "status": 200}
                last_status = resp.status_code
                # Avoid logging huge payloads
                last_body = resp.text[:500] if resp.text else None
                last_url = url
                last_auth = "Bearer" if headers["Authorization"].startswith("Bearer") else "Token"
                # If unauthorized, no need to try same header again on same path
                # continue to next header/path
            except Exception as e:
                logger.exception("dbt cloud test exception on %s: %s", url, str(e))
                return {"ok": False, "error": str(e), "url": url}

    return {
        "ok": False,
        "status": last_status,
        "error": last_body,
        "url": last_url,
        "auth_scheme": last_auth,
    }


@router.post("/test-connection")
@version(1, 0)
def test_connection(
    conn: DbtCloudConnRequest,
    current_user: User = Depends(require_connector_access()),
    request: Request = None,
):
    result = _test_dbt_cloud_connection(conn.api_key, conn.account_id, str(conn.base_url))
    if result.get("ok"):
        return {"message": "Connection successful"}
    # Surface details to assist debugging
    return {
        "message": "Connection failed",
        "status": result.get("status"),
        "error": result.get("error"),
        "url": result.get("url"),
        "auth_scheme": result.get("auth_scheme"),
    }


@router.post("/save-connection", response_model=DbtCloudConnectionResponse, status_code=status.HTTP_201_CREATED)
@version(1, 0)
def save_connection(
    conn: DbtCloudConnRequest,
    current_user: User = Depends(require_connector_access()),
    db: Session = Depends(get_db),
    request: Request = None,
):
    # test before save
    test_result = _test_dbt_cloud_connection(conn.api_key, conn.account_id, str(conn.base_url))
    if not test_result.get("ok"):
        raise HTTPException(
            status_code=400,
            detail=f"Connection test failed (status={test_result.get('status')}): {test_result.get('error')}"
        )

    # ensure unique connection_name per org
    existing = (
        db.query(DbtCloudConnection)
        .filter(
            DbtCloudConnection.org_id == current_user.org_id,
            DbtCloudConnection.connection_name == conn.connection_name,
            DbtCloudConnection.is_active == True,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Connection name already exists for this organization")

    new_conn = DbtCloudConnection(
        org_id=current_user.org_id,
        connection_name=conn.connection_name,
        api_key=conn.api_key,
        account_id=conn.account_id,
        base_url=str(conn.base_url).rstrip("/"),
    )
    try:
        db.add(new_conn)
        db.commit()
        db.refresh(new_conn)
        return new_conn
    except IntegrityError:
        db.rollback()
        logger.exception("/dbt-cloud/save-connection - failed to save connection")
        raise HTTPException(status_code=400, detail="Failed to save connection")


@router.get("/connections", response_model=List[DbtCloudConnectionResponse])
@version(1, 0)
def list_connections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    rows = (
        db.query(DbtCloudConnection)
        .filter(
            DbtCloudConnection.org_id == current_user.org_id,
            DbtCloudConnection.is_active == True,
        )
        .order_by(DbtCloudConnection.created_at.desc())
        .all()
    )
    return rows


@router.post("/schedule/{connection_id}")
@version(1, 0)
def set_schedule(
    connection_id: str,
    body: DbtScheduleRequest,
    current_user: User = Depends(require_connector_access()),
    db: Session = Depends(get_db),
    request: Request = None,
):
    conn = db.query(DbtCloudConnection).filter(
        DbtCloudConnection.id == connection_id,
        DbtCloudConnection.org_id == current_user.org_id,
        DbtCloudConnection.is_active == True,
    ).first()
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")

    job = db.query(DbtJob).filter(DbtJob.connection_id == connection_id).first()
    if job:
        job.cron_expression = body.cron_expression
        job.is_active = True
        # leave last_run_time unchanged
    else:
        # create with last_run_time = NULL to indicate never run
        job = DbtJob(connection_id=connection_id, cron_expression=body.cron_expression, is_active=True)
        db.add(job)
    db.commit()
    return {"message": "Schedule set", "cron_expression": body.cron_expression}


@router.post("/sync/{connection_id}")
@version(1, 0)
def sync(connection_id: str, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db), request: Request = None):
    # Ensure connection belongs to org
    conn = db.query(DbtCloudConnection).filter(
        DbtCloudConnection.id == connection_id,
        DbtCloudConnection.org_id == current_user.org_id,
        DbtCloudConnection.is_active == True,
    ).first()
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")

    try:
        result = sync_dbt_metadata(db, connection_id)
        return {"message": "Sync completed", **result}
    except Exception as e:
        logger.exception("/dbt-cloud/sync failed: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Sync failed: {str(e)}")


