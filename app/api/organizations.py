# Organization management endpoints (Admin only)
# POST /organizations → create new organization
# GET /organizations → list all organizations
# GET /organizations/{org_id} → get organization details
# PUT /organizations/{org_id} → update organization
# DELETE /organizations/{org_id} → deactivate organization

from fastapi import APIRouter, HTTPException, Depends, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Dict
from pydantic import BaseModel, EmailStr
from app.database import get_db
from app.utils.models import Organization, User
from app.utils.auth_deps import get_current_user
from app.utils.rbac import require_organizations_endpoint_access, check_organization_access, PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER
from app.api.auth import hash_password, generate_reset_token
from app.utils.email_service import send_welcome_email, send_password_setup_email
import uuid
from uuid import UUID
from datetime import datetime, timedelta
import logging
from collections import Counter
import secrets

router = APIRouter(prefix="/organizations", tags=["Organizations"])
logger = logging.getLogger("organizations")


# --- Models ---
class OrganizationCreate(BaseModel):
    name: str
    username: str  # Username for the first SYSTEM_ADMIN user
    email: EmailStr  # Email for the first SYSTEM_ADMIN user

class OrganizationUpdate(BaseModel):
    name: str | None = None
    is_active: bool | None = None

class OrganizationResponse(BaseModel):
    id: UUID
    name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True

class UserKPIInfo(BaseModel):
    id: UUID
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class RoleCounts(BaseModel):
    PRODUCT_SUPPORT_ADMIN: int = 0
    SYSTEM_ADMIN: int = 0
    ORGANIZATION_ADMIN: int = 0
    MEMBER: int = 0
    total: int = 0

class OrganizationKPI(BaseModel):
    organization: OrganizationResponse
    users: List[UserKPIInfo]
    role_counts: RoleCounts

class KPIMetadata(BaseModel):
    total_organizations: int
    total_users: int
    organizations: List[OrganizationKPI]


# --- Endpoints ---
@router.post("/", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
def create_organization(
    org: OrganizationCreate,
    current_user: User = Depends(require_organizations_endpoint_access()),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Create a new organization with first SYSTEM_ADMIN user (PRODUCT_SUPPORT_ADMIN only).
    Automatically creates the first user with SYSTEM_ADMIN role using the provided username and email.
    No password is required - a password setup link will be sent via email.
    """
    # Check if organization name already exists
    existing_org = db.query(Organization).filter(Organization.name == org.name).first()
    if existing_org:
        logger.warning("/organizations - create: name exists %s", org.name)
        raise HTTPException(status_code=400, detail="Organization name already exists")
    
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == org.username).first()
    if existing_user:
        logger.warning("/organizations - create: username exists %s", org.username)
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == org.email).first()
    if existing_email:
        logger.warning("/organizations - create: email exists %s", org.email)
        raise HTTPException(status_code=400, detail="Email already exists")

    new_org = Organization(name=org.name)
    
    try:
        # Create organization first
        db.add(new_org)
        db.flush()  # Flush to get the org.id without committing
        logger.info("/organizations - created id=%s name=%s", new_org.id, new_org.name)
        
        # Generate reset token for first-time password setup
        reset_token = generate_reset_token()
        token_expires = datetime.utcnow() + timedelta(minutes=60)  # 60 minute expiry
        
        # Generate a temporary random password (user will set their own via reset link)
        temp_password = secrets.token_urlsafe(32)
        password_hash = hash_password(temp_password)
        
        # Create first user with SYSTEM_ADMIN role
        first_user = User(
            username=org.username,
            email=org.email,
            password_hash=password_hash,
            org_id=new_org.id,
            role=SYSTEM_ADMIN,
            is_active=True,
            password_reset_token=reset_token,
            password_reset_token_expires=token_expires
        )
        
        db.add(first_user)
        db.commit()  # Commit both organization and user together
        db.refresh(new_org)
        db.refresh(first_user)
        
        # Send welcome email
        welcome_email_sent = send_welcome_email(org.email, org.username)
        if not welcome_email_sent:
            logger.error("/organizations - create: failed to send welcome email to %s", org.email)
        
        # Send password setup email with reset link
        setup_email_sent = send_password_setup_email(org.email, org.username, reset_token)
        if not setup_email_sent:
            logger.error("/organizations - create: failed to send password setup email to %s", org.email)
        
        logger.info("/organizations - created org id=%s name=%s with first SYSTEM_ADMIN user id=%s username=%s", 
                   new_org.id, new_org.name, first_user.id, org.username)
        return new_org
    except IntegrityError:
        db.rollback()
        logger.exception("/organizations - create failed")
        raise HTTPException(status_code=400, detail="Failed to create organization")


@router.get("/", response_model=List[OrganizationResponse])
def list_organizations(
    current_user: User = Depends(require_organizations_endpoint_access()),
    db: Session = Depends(get_db),
    request: Request = None
):
    """List all organizations (PRODUCT_SUPPORT_ADMIN only)"""
    organizations = db.query(Organization).all()
    logger.debug("/organizations - list count=%d", len(organizations))
    return organizations


@router.get("/kpi", response_model=KPIMetadata)
def get_organizations_kpi(
    current_user: User = Depends(require_organizations_endpoint_access()),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Get KPI data for all organizations with users and role counts.
    Returns all organizations with their respective users and KPI metadata.
    (PRODUCT_SUPPORT_ADMIN only)
    """
    logger.info("/organizations/kpi - request by user_id=%s", current_user.id)
    
    # Get all organizations
    organizations = db.query(Organization).order_by(Organization.created_at).all()
    
    # Get all users with their organizations
    users = db.query(User).order_by(User.created_at).all()
    
    # Group users by organization
    users_by_org: Dict[UUID, List[User]] = {}
    for user in users:
        if user.org_id not in users_by_org:
            users_by_org[user.org_id] = []
        users_by_org[user.org_id].append(user)
    
    # Build response
    organization_kpis = []
    total_users_count = 0
    
    for org in organizations:
        org_users = users_by_org.get(org.id, [])
        total_users_count += len(org_users)
        
        # Count roles
        role_counter = Counter(user.role for user in org_users)
        
        role_counts = RoleCounts(
            PRODUCT_SUPPORT_ADMIN=role_counter.get(PRODUCT_SUPPORT_ADMIN, 0),
            SYSTEM_ADMIN=role_counter.get(SYSTEM_ADMIN, 0),
            ORGANIZATION_ADMIN=role_counter.get(ORGANIZATION_ADMIN, 0),
            MEMBER=role_counter.get(MEMBER, 0),
            total=len(org_users)
        )
        
        # Convert users to response format
        user_kpi_info = [
            UserKPIInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at
            )
            for user in org_users
        ]
        
        organization_kpis.append(
            OrganizationKPI(
                organization=OrganizationResponse(
                    id=org.id,
                    name=org.name,
                    is_active=org.is_active,
                    created_at=org.created_at,
                    updated_at=org.updated_at
                ),
                users=user_kpi_info,
                role_counts=role_counts
            )
        )
    
    logger.info("/organizations/kpi - returning %d organizations with %d total users", 
                len(organizations), total_users_count)
    
    return KPIMetadata(
        total_organizations=len(organizations),
        total_users=total_users_count,
        organizations=organization_kpis
    )


@router.get("/{org_id}", response_model=OrganizationResponse)
def get_organization(
    org_id: str,
    current_user: User = Depends(require_organizations_endpoint_access()),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Get organization details by ID (PRODUCT_SUPPORT_ADMIN only)"""
    try:
        org_uuid = uuid.UUID(org_id)
    except ValueError:
        logger.warning("/organizations - get: invalid id %s", org_id)
        raise HTTPException(status_code=400, detail="Invalid organization ID format")

    organization = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not organization:
        logger.warning("/organizations - get: not found %s", org_uuid)
        raise HTTPException(status_code=404, detail="Organization not found")
    
    return organization


@router.put("/{org_id}", response_model=OrganizationResponse)
def update_organization(
    org_id: str,
    org_update: OrganizationUpdate,
    current_user: User = Depends(require_organizations_endpoint_access()),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Update organization details (PRODUCT_SUPPORT_ADMIN only)"""
    try:
        org_uuid = uuid.UUID(org_id)
    except ValueError:
        logger.warning("/organizations - update: invalid id %s", org_id)
        raise HTTPException(status_code=400, detail="Invalid organization ID format")

    organization = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Update fields if provided
    if org_update.name is not None:
        # Check if new name already exists (excluding current org)
        existing_org = db.query(Organization).filter(
            Organization.name == org_update.name,
            Organization.id != org_uuid
        ).first()
        if existing_org:
            logger.warning("/organizations - update: name exists %s", org_update.name)
            raise HTTPException(status_code=400, detail="Organization name already exists")
        organization.name = org_update.name

    if org_update.is_active is not None:
        organization.is_active = org_update.is_active

    try:
        db.commit()
        db.refresh(organization)
        logger.info("/organizations - updated id=%s", organization.id)
        return organization
    except IntegrityError:
        db.rollback()
        logger.exception("/organizations - update failed")
        raise HTTPException(status_code=400, detail="Failed to update organization")


@router.delete("/{org_id}")
def deactivate_organization(
    org_id: str,
    current_user: User = Depends(require_organizations_endpoint_access()),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Deactivate organization (PRODUCT_SUPPORT_ADMIN only) - Soft delete"""
    try:
        org_uuid = uuid.UUID(org_id)
    except ValueError:
        logger.warning("/organizations - delete: invalid id %s", org_id)
        raise HTTPException(status_code=400, detail="Invalid organization ID format")

    organization = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    organization.is_active = False
    
    try:
        db.commit()
        logger.info("/organizations - deactivated id=%s", organization.id)
        return {"message": "Organization deactivated successfully"}
    except IntegrityError:
        db.rollback()
        logger.exception("/organizations - deactivate failed")
        raise HTTPException(status_code=400, detail="Failed to deactivate organization")
