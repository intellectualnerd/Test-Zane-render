# User Management Endpoints (Admin Only)
# POST /users → create new user with role assignment
# GET /users → list users (filtered by organization)
# GET /users/{user_id} → get user details
# PUT /users/{user_id} → update user (role, email, etc.)
# DELETE /users/{user_id} → deactivate user (soft delete)

from fastapi import APIRouter, HTTPException, Depends, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime, timedelta
from app.database import get_db
from app.utils.models import User, Organization
from app.utils.auth_deps import get_current_user
from app.api.auth import hash_password, generate_reset_token
from app.utils.email_service import send_welcome_email, send_password_setup_email
import secrets
from app.utils.rbac import (
    PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER,
    VALID_ROLES, require_minimum_role, check_role_assignment, 
    check_organization_access, can_access_organization
)
import uuid
import logging
from fastapi_versioning import version

router = APIRouter(prefix="/users", tags=["Users"])
logger = logging.getLogger("users")


# --- Models ---
class CreateUserRequest(BaseModel):
    username: str
    password: Optional[str] = None  # Optional - if not provided, password reset link will be sent for password setup
    email: EmailStr
    role: str = MEMBER  # Default to MEMBER if not specified

class UpdateUserRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None  # Optional password update

class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    org_id: UUID
    role: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- Helper Functions ---
def can_modify_user(modifier: User, target_user: User) -> bool:
    """
    Check if a user can modify another user.
    PRODUCT_SUPPORT_ADMIN can modify any user.
    Others can only modify users in their own organization.
    """
    if modifier.role == PRODUCT_SUPPORT_ADMIN:
        return True
    return str(modifier.org_id) == str(target_user.org_id)


# --- Endpoints ---
@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@version(1, 0)
def create_user(
    user: CreateUserRequest,
    org_id: str,
    current_user: User = Depends(require_minimum_role(ORGANIZATION_ADMIN)),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Create a new user with role assignment.
    Requires: PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, or ORGANIZATION_ADMIN
    org_id should be provided as a query parameter in the link.
    """
    logger.info("POST /users - attempt by user_id=%s for username=%s role=%s org_id=%s", 
                current_user.id, user.username, user.role, org_id)
    
    # Validate role
    if user.role not in VALID_ROLES:
        logger.warning("/users - create: invalid role %s", user.role)
        raise HTTPException(status_code=400, detail=f"Invalid role. Valid roles: {', '.join(VALID_ROLES)}")
    
    # Validate org_id format
    try:
        org_uuid = uuid.UUID(org_id)
    except ValueError:
        logger.warning("/users - create: invalid org_id format %s", org_id)
        raise HTTPException(status_code=400, detail="Invalid organization ID format")
    
    # Check if current user can assign the requested role
    check_role_assignment(current_user, user.role, org_id)
    
    # Check organization access (unless PRODUCT_SUPPORT_ADMIN)
    check_organization_access(current_user, str(org_uuid))
    
    # Check if organization exists and is active
    organization = db.query(Organization).filter(
        Organization.id == org_uuid,
        Organization.is_active == True
    ).first()
    if not organization:
        logger.warning("/users - create: org not found or inactive %s", org_uuid)
        raise HTTPException(status_code=400, detail="Invalid organization ID or organization is inactive")
    
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        logger.warning("/users - create: username exists %s", user.username)
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user.email).first()
    if existing_email:
        logger.warning("/users - create: email exists %s", user.email)
        raise HTTPException(status_code=400, detail="Email already exists")

    # Generate reset token for first-time password setup/reset
    # This allows new users to set their own password regardless of whether admin provided one
    reset_token = generate_reset_token()
    token_expires = datetime.utcnow() + timedelta(minutes=60)  # 60 minute expiry
    
    # Determine password hash
    if user.password is None:
        # Generate a temporary random password (user will set their own via reset link)
        temp_password = secrets.token_urlsafe(32)
        password_hash = hash_password(temp_password)
    else:
        # Use provided password
        password_hash = hash_password(user.password)
    
    # Create user first
    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=password_hash,
        org_id=org_uuid,
        role=user.role,
        is_active=True,
        password_reset_token=reset_token,
        password_reset_token_expires=token_expires
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Send welcome email first
        welcome_email_sent = send_welcome_email(user.email, user.username)
        if not welcome_email_sent:
            logger.error("/users - create: failed to send welcome email to %s", user.email)
        
        # Then send password setup email with reset link
        setup_email_sent = send_password_setup_email(user.email, user.username, reset_token)
        if not setup_email_sent:
            logger.error("/users - create: failed to send password setup email to %s", user.email)
        
        logger.info("/users - created id=%s role=%s by user_id=%s, welcome email and password setup link sent", 
                   new_user.id, new_user.role, current_user.id)
        return new_user
    except IntegrityError:
        db.rollback()
        logger.exception("/users - create failed due to IntegrityError")
        raise HTTPException(status_code=400, detail="User creation failed")


@router.get("/", response_model=List[UserResponse])
@version(1, 0)
def list_users(
    org_id: Optional[str] = None,
    current_user: User = Depends(require_minimum_role(ORGANIZATION_ADMIN)),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    List users.
    PRODUCT_SUPPORT_ADMIN can list users from any organization.
    Others can only list users from their own organization.
    """
    logger.info("GET /users - request by user_id=%s org_id=%s", current_user.id, org_id)
    
    query = db.query(User)
    
    # Filter by organization
    if org_id:
        try:
            org_uuid = uuid.UUID(org_id)
        except ValueError:
            logger.warning("/users - list: invalid org_id format %s", org_id)
            raise HTTPException(status_code=400, detail="Invalid organization ID format")
        
        # Check organization access
        check_organization_access(current_user, org_id)
        query = query.filter(User.org_id == org_uuid)
    else:
        # If no org_id specified, filter by current user's org (unless PRODUCT_SUPPORT_ADMIN)
        if current_user.role != PRODUCT_SUPPORT_ADMIN:
            query = query.filter(User.org_id == current_user.org_id)
    
    users = query.order_by(User.created_at.desc()).all()
    logger.debug("/users - list count=%d", len(users))
    return users


@router.get("/{user_id}", response_model=UserResponse)
@version(1, 0)
def get_user(
    user_id: str,
    current_user: User = Depends(require_minimum_role(ORGANIZATION_ADMIN)),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Get user details by ID.
    PRODUCT_SUPPORT_ADMIN can access any user.
    Others can only access users from their own organization.
    """
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        logger.warning("/users - get: invalid id %s", user_id)
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        logger.warning("/users - get: not found %s", user_uuid)
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check organization access
    check_organization_access(current_user, str(user.org_id))
    
    return user


@router.put("/{user_id}", response_model=UserResponse)
@version(1, 0)
def update_user(
    user_id: str,
    user_update: UpdateUserRequest,
    current_user: User = Depends(require_minimum_role(ORGANIZATION_ADMIN)),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Update user details.
    PRODUCT_SUPPORT_ADMIN can update any user.
    Others can only update users from their own organization.
    Role changes are validated based on assigner's permissions.
    """
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        logger.warning("/users - update: invalid id %s", user_id)
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        logger.warning("/users - update: not found %s", user_uuid)
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if current user can modify this user
    if not can_modify_user(current_user, user):
        logger.warning("/users - update: access denied user_id=%s target_user_id=%s", 
                      current_user.id, user_uuid)
        raise HTTPException(status_code=403, detail="You do not have permission to modify this user")
    
    # Validate role if being updated
    if user_update.role is not None:
        if user_update.role not in VALID_ROLES:
            logger.warning("/users - update: invalid role %s", user_update.role)
            raise HTTPException(status_code=400, detail=f"Invalid role. Valid roles: {', '.join(VALID_ROLES)}")
        
        # Check if current user can assign the new role
        check_role_assignment(current_user, user_update.role, str(user.org_id))
    
    # Update fields if provided
    if user_update.username is not None:
        # Check if new username already exists (excluding current user)
        existing_user = db.query(User).filter(
            User.username == user_update.username,
            User.id != user_uuid
        ).first()
        if existing_user:
            logger.warning("/users - update: username exists %s", user_update.username)
            raise HTTPException(status_code=400, detail="Username already exists")
        user.username = user_update.username

    if user_update.email is not None:
        # Check if new email already exists (excluding current user)
        existing_email = db.query(User).filter(
            User.email == user_update.email,
            User.id != user_uuid
        ).first()
        if existing_email:
            logger.warning("/users - update: email exists %s", user_update.email)
            raise HTTPException(status_code=400, detail="Email already exists")
        user.email = user_update.email

    if user_update.role is not None:
        user.role = user_update.role

    if user_update.is_active is not None:
        user.is_active = user_update.is_active

    if user_update.password is not None:
        user.password_hash = hash_password(user_update.password)

    try:
        db.commit()
        db.refresh(user)
        logger.info("/users - updated id=%s by user_id=%s", user.id, current_user.id)
        return user
    except IntegrityError:
        db.rollback()
        logger.exception("/users - update failed")
        raise HTTPException(status_code=400, detail="Failed to update user")


@router.delete("/{user_id}")
@version(1, 0)
def deactivate_user(
    user_id: str,
    current_user: User = Depends(require_minimum_role(ORGANIZATION_ADMIN)),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Deactivate user (soft delete).
    PRODUCT_SUPPORT_ADMIN can deactivate any user.
    Others can only deactivate users from their own organization.
    """
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        logger.warning("/users - delete: invalid id %s", user_id)
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        logger.warning("/users - delete: not found %s", user_uuid)
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if current user can modify this user
    if not can_modify_user(current_user, user):
        logger.warning("/users - delete: access denied user_id=%s target_user_id=%s", 
                      current_user.id, user_uuid)
        raise HTTPException(status_code=403, detail="You do not have permission to deactivate this user")
    
    # Prevent self-deactivation
    if user.id == current_user.id:
        logger.warning("/users - delete: self-deactivation attempt user_id=%s", user_uuid)
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
    
    user.is_active = False
    
    try:
        db.commit()
        logger.info("/users - deactivated id=%s by user_id=%s", user.id, current_user.id)
        return {"message": "User deactivated successfully"}
    except IntegrityError:
        db.rollback()
        logger.exception("/users - deactivate failed")
        raise HTTPException(status_code=400, detail="Failed to deactivate user")

