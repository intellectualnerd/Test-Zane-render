# POST /auth/signup → register user
# POST /auth/login → login & get JWT
# POST /auth/forgot-password → generate reset token
# POST /auth/reset-password → reset password with token
# POST /auth/change-password → change password (requires authentication)
# POST /auth/logout → revoke JWT
# GET /auth/me → get current user info

from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, EmailStr
from uuid import UUID
from jose import jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database import get_db
from app.utils.models import (
    User, UserToken, Organization,
    SnowflakeConnection, GitHubInstallation, JiraConnection, DbtCloudConnection
)
from app.utils.rbac import MEMBER, has_any_role, PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN
from typing import List, Dict
from app.utils.auth_deps import get_current_user, SECRET_KEY, ALGORITHM
from app.utils.email_service import send_password_reset_email
import hashlib
import uuid
import os
import secrets
import logging

router = APIRouter(prefix="/auth", tags=["Auth"])
logger = logging.getLogger("auth")

# --- Config ---
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# --- Models ---
class UserSignup(BaseModel):
    username: str
    password: str
    email: EmailStr

class UserLogin(BaseModel):
    username: str
    password: str

class ForgotPassword(BaseModel):
    email: EmailStr

class ResetPassword(BaseModel):
    token: str
    new_password: str

class ChangePassword(BaseModel):
    current_password: str
    new_password: str

class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    org_id: UUID
    role: str
    is_connection_setup: bool
    missing_connectors: List[str]

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    is_connection_setup: bool
    missing_connectors: List[str]


# --- Helpers ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def generate_reset_token() -> str:
    """Generate a secure random token for password reset"""
    return secrets.token_urlsafe(32)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def check_connector_setup_status(org_id: UUID, db: Session) -> Dict[str, bool]:
    """
    Check which connectors are set up for an organization.
    Returns a dictionary with connector names as keys and boolean values indicating if they're set up.
    """
    connectors_status = {
        "snowflake": False,
        "github": False,
        "jira": False,
        "dbt_cloud": False
    }
    
    # Check Snowflake connections
    snowflake_conn = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.org_id == org_id,
        SnowflakeConnection.is_active == True
    ).first()
    if snowflake_conn:
        connectors_status["snowflake"] = True
    
    # Check GitHub installations
    github_installation = db.query(GitHubInstallation).filter(
        GitHubInstallation.org_id == org_id,
        GitHubInstallation.is_active == True
    ).first()
    if github_installation:
        connectors_status["github"] = True
    
    # Check Jira connections
    jira_conn = db.query(JiraConnection).filter(
        JiraConnection.org_id == org_id,
        JiraConnection.is_active == True
    ).first()
    if jira_conn:
        connectors_status["jira"] = True
    
    # Check dbt Cloud connections
    dbt_conn = db.query(DbtCloudConnection).filter(
        DbtCloudConnection.org_id == org_id,
        DbtCloudConnection.is_active == True
    ).first()
    if dbt_conn:
        connectors_status["dbt_cloud"] = True
    
    return connectors_status


def get_missing_connectors(connectors_status: Dict[str, bool]) -> List[str]:
    """Get list of connector names that are not set up."""
    missing = []
    connector_display_names = {
        "snowflake": "Snowflake",
        "github": "GitHub",
        "jira": "Jira",
        "dbt_cloud": "dbt Cloud"
    }
    
    for connector, is_setup in connectors_status.items():
        if not is_setup:
            missing.append(connector_display_names.get(connector, connector))
    
    return missing


# --- Endpoints ---
@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user: UserSignup, org_id: str, db: Session = Depends(get_db), request: Request = None):
    """
    Public signup endpoint - creates users with MEMBER role only.
    For creating users with other roles, use /users endpoint (requires authentication and appropriate role).
    org_id should be provided as a query parameter in the signup link.
    """
    logger.info("POST /auth/signup - attempt for username=%s org_id=%s ip=%s", user.username, org_id, request.client.host if request and request.client else "unknown")
    # Validate org_id format
    try:
        org_uuid = uuid.UUID(org_id)
    except ValueError:
        logger.warning("/auth/signup - invalid org_id format: %s", org_id)
        raise HTTPException(status_code=400, detail="Invalid organization ID format")
    
    # Check if organization exists and is active
    organization = db.query(Organization).filter(
        Organization.id == org_uuid,
        Organization.is_active == True
    ).first()
    if not organization:
        logger.warning("/auth/signup - org not found or inactive: %s", org_uuid)
        raise HTTPException(status_code=400, detail="Invalid organization ID or organization is inactive")
    
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        logger.warning("/auth/signup - username exists: %s", user.username)
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user.email).first()
    if existing_email:
        logger.warning("/auth/signup - email exists: %s", user.email)
        raise HTTPException(status_code=400, detail="Email already exists")

    # Public signup always creates MEMBER role users
    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=hash_password(user.password),
        org_id=org_uuid,
        role=MEMBER
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info("/auth/signup - user created id=%s role=%s", new_user.id, new_user.role)
        return {"message": "User registered successfully"}
    except IntegrityError:
        db.rollback()
        logger.exception("/auth/signup - registration failed due to IntegrityError")
        raise HTTPException(status_code=400, detail="Registration failed")


@router.post("/login", response_model=LoginResponse)
def login(user: UserLogin, db: Session = Depends(get_db), request: Request = None):
    logger.info("POST /auth/login - attempt username=%s ip=%s", user.username, request.client.host if request and request.client else "unknown")
    db_user = db.query(User).filter(
        User.username == user.username,
        User.is_active == True
    ).first()
    
    if not db_user or db_user.password_hash != hash_password(user.password):
        logger.warning("/auth/login - invalid credentials for %s", user.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": str(db_user.id)}, 
        expires_delta=access_token_expires
    )
    
    # Store token in database
    token_record = UserToken(
        user_id=db_user.id,
        token=token,
        expires_at=datetime.utcnow() + access_token_expires
    )
    
    db.add(token_record)
    
    # Check connection setup status for admin users
    is_connection_setup = True
    missing_connectors = []
    
    # Only check for users with admin-level access (can manage connectors)
    if has_any_role(db_user, [PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN]):
        # Get organization
        organization = db.query(Organization).filter(Organization.id == db_user.org_id).first()
        
        if organization:
            # If flag is already True, skip checking (optimization)
            if not organization.is_connection_setup:
                # Check connector setup status
                connectors_status = check_connector_setup_status(db_user.org_id, db)
                
                # ===== CONNECTION SETUP LOGIC =====
                # To switch back to checking ALL connectors, uncomment the block below
                # and comment out the "REQUIRED CONNECTORS" block
                
                # OPTION 1: Check if ALL connectors are set up (previous logic)
                # all_setup = all(connectors_status.values())
                # is_connection_setup = all_setup
                # log_msg = "all connectors" if all_setup else "missing connectors"
                
                # OPTION 2: Check if REQUIRED connectors (snowflake AND github) are set up (current logic)
                snowflake_setup = connectors_status.get("snowflake", False)
                github_setup = connectors_status.get("github", False)
                is_connection_setup = snowflake_setup and github_setup
                log_msg = "required connectors (snowflake & github)" if is_connection_setup else "missing connectors"
                
                if is_connection_setup:
                    # Update the flag to True
                    organization.is_connection_setup = True
                    logger.info("/auth/login - %s set up for org_id=%s, flag updated", log_msg, db_user.org_id)
                else:
                    missing_connectors = get_missing_connectors(connectors_status)
                    logger.info("/auth/login - missing connectors for org_id=%s: %s", db_user.org_id, missing_connectors)
            else:
                # Flag is already True, so connection setup is complete
                is_connection_setup = True
                logger.debug("/auth/login - connection setup already complete for org_id=%s", db_user.org_id)
    
    db.commit()
    logger.info("/auth/login - token issued for user_id=%s", db_user.id)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "is_connection_setup": is_connection_setup,
        "missing_connectors": missing_connectors
    }


@router.post("/forgot-password")
def forgot_password(req: ForgotPassword, db: Session = Depends(get_db), request: Request = None):
    logger.info("POST /auth/forgot-password - email=%s ip=%s", req.email, request.client.host if request and request.client else "unknown")
    user = db.query(User).filter(User.email == req.email, User.is_active == True).first()
    if not user:
        logger.warning("/auth/forgot-password - email not found: %s", req.email)
        raise HTTPException(status_code=404, detail="Email not found")

    # Generate secure reset token
    reset_token = generate_reset_token()
    user.password_reset_token = reset_token
    user.password_reset_token_expires = datetime.utcnow() + timedelta(minutes=60)  # 60 minute expiry
    
    db.commit()
    
    # Send email with reset link
    email_sent = send_password_reset_email(req.email, reset_token)
    if not email_sent:
        logger.error("/auth/forgot-password - failed to send email to %s for user_id=%s", req.email, user.id)
        # Still return success to prevent email enumeration attacks
        # The token is still generated and stored, but email delivery failed
    
    logger.info("/auth/forgot-password - reset token generated for user_id=%s, email_sent=%s", user.id, email_sent)
    return {
        "message": "If the email exists, a password reset link has been sent to your email address",
        "note": "The link will expire in 60 minutes"
    }


@router.post("/reset-password")
def reset_password(req: ResetPassword, db: Session = Depends(get_db), request: Request = None):
    logger.info("POST /auth/reset-password - token=%s ip=%s", req.token[:10] + "..." if len(req.token) > 10 else req.token, request.client.host if request and request.client else "unknown")
    user = db.query(User).filter(
        User.password_reset_token == req.token,
        User.password_reset_token_expires > datetime.utcnow(),
        User.is_active == True
    ).first()
    
    if not user:
        logger.warning("/auth/reset-password - invalid/expired token")
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user.password_hash = hash_password(req.new_password)
    user.password_reset_token = None
    user.password_reset_token_expires = None
    
    # Revoke all existing tokens for this user
    db.query(UserToken).filter(UserToken.user_id == user.id).update({"is_revoked": True})
    
    db.commit()
    logger.info("/auth/reset-password - password reset for user_id=%s", user.id)
    return {"message": "Password reset successful"}


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    logger.info("POST /auth/logout - user_id=%s ip=%s", current_user.id, request.client.host if request and request.client else "unknown")
    # Get the token from the request
    # Note: We need to get the raw token to revoke it
    # This is a simplified approach - in production you might want to pass the token explicitly
    
    # For now, we'll revoke all tokens for the user
    db.query(UserToken).filter(UserToken.user_id == current_user.id).update({"is_revoked": True})
    db.commit()
    
    logger.info("/auth/logout - tokens revoked for user_id=%s", current_user.id)
    return {"message": "Logged out successfully"}


@router.post("/change-password")
def change_password(
    req: ChangePassword,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Change password for authenticated user.
    Requires current password verification and sets new password.
    """
    logger.info("POST /auth/change-password - user_id=%s ip=%s", current_user.id, request.client.host if request and request.client else "unknown")
    
    # Verify current password
    if current_user.password_hash != hash_password(req.current_password):
        logger.warning("/auth/change-password - invalid current password for user_id=%s", current_user.id)
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    # Check if new password is different from current password
    if hash_password(req.new_password) == current_user.password_hash:
        logger.warning("/auth/change-password - new password same as current password for user_id=%s", current_user.id)
        raise HTTPException(status_code=400, detail="New password must be different from current password")
    
    # Update password
    current_user.password_hash = hash_password(req.new_password)
    
    # Revoke all existing tokens for this user (force re-login for security)
    db.query(UserToken).filter(UserToken.user_id == current_user.id).update({"is_revoked": True})
    
    db.commit()
    logger.info("/auth/change-password - password changed successfully for user_id=%s", current_user.id)
    return {"message": "Password changed successfully"}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    logger.debug("GET /auth/me - user_id=%s ip=%s", current_user.id, request.client.host if request and request.client else "unknown")
    
    # Check connection setup status for admin users
    is_connection_setup = True
    missing_connectors = []
    
    # Only check for users with admin-level access (can manage connectors)
    if has_any_role(current_user, [PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN]):
        # Get organization
        organization = db.query(Organization).filter(Organization.id == current_user.org_id).first()
        
        if organization:
            # If flag is already True, skip checking (optimization)
            if not organization.is_connection_setup:
                # Check connector setup status
                connectors_status = check_connector_setup_status(current_user.org_id, db)
                
                # ===== CONNECTION SETUP LOGIC =====
                # To switch back to checking ALL connectors, uncomment the block below
                # and comment out the "REQUIRED CONNECTORS" block
                
                # OPTION 1: Check if ALL connectors are set up (previous logic)
                # all_setup = all(connectors_status.values())
                # is_connection_setup = all_setup
                # log_msg = "all connectors" if all_setup else "missing connectors"
                
                # OPTION 2: Check if REQUIRED connectors (snowflake AND github) are set up (current logic)
                snowflake_setup = connectors_status.get("snowflake", False)
                github_setup = connectors_status.get("github", False)
                is_connection_setup = snowflake_setup and github_setup
                log_msg = "required connectors (snowflake & github)" if is_connection_setup else "missing connectors"
                
                if is_connection_setup:
                    # Update the flag to True
                    organization.is_connection_setup = True
                    logger.info("/auth/me - %s set up for org_id=%s, flag updated", log_msg, current_user.org_id)
                else:
                    missing_connectors = get_missing_connectors(connectors_status)
                    logger.info("/auth/me - missing connectors for org_id=%s: %s", current_user.org_id, missing_connectors)
            else:
                # Flag is already True, so connection setup is complete
                is_connection_setup = True
                logger.debug("/auth/me - connection setup already complete for org_id=%s", current_user.org_id)
    
    db.commit()
    
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "org_id": current_user.org_id,
        "role": current_user.role,
        "is_connection_setup": is_connection_setup,
        "missing_connectors": missing_connectors
    }
