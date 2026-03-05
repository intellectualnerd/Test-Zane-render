"""
One-time initialization script and API endpoint to create the first PRODUCT_SUPPORT_ADMIN user and QueryGuardAI organization.

This can be used in two ways:
1. As a script: python scripts/init_product_support_admin.py
2. As an API endpoint: POST /init-setup

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (required)

Hardcoded Credentials:
    Username: admin
    Email: admin@queryguardai.com
    Password: Admin@123
    
⚠️  IMPORTANT: Change the default password after first login!
⚠️  WARNING: The API endpoint should be removed or protected after initial setup!
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database import SessionLocal, init_db, get_db
from app.utils.models import Organization, User
# from app.api.auth import hash_password
from app.routers.v1.v1_0.auth import hash_password
from app.utils.rbac import PRODUCT_SUPPORT_ADMIN
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional
import uuid
import logging
from fastapi_versioning import version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("init_script")

# FastAPI Router for endpoint usage
router = APIRouter(prefix="/init-setup", tags=["Initialization"])


class InitResponse(BaseModel):
    message: str
    organization_id: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    warning: Optional[str] = None


def create_initial_admin(db: Optional[Session] = None, return_response: bool = False):
    """
    Create the initial QueryGuardAI organization and PRODUCT_SUPPORT_ADMIN user.
    
    Args:
        db: Optional database session. If None, creates a new session.
        return_response: If True, returns InitResponse dict instead of logging.
    
    Returns:
        If return_response=True, returns InitResponse dict. Otherwise returns None.
    """
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        # Initialize database (create tables if they don't exist)
        logger.info("Initializing database...")
        init_db()
        
        # Check if QueryGuardAI organization already exists
        existing_org = db.query(Organization).filter(
            Organization.name == "QueryGuardAI"
        ).first()
        
        if existing_org:
            logger.warning(f"QueryGuardAI organization already exists with ID: {existing_org.id}")
            
            # Check if PRODUCT_SUPPORT_ADMIN user already exists
            existing_admin = db.query(User).filter(
                User.role == PRODUCT_SUPPORT_ADMIN,
                User.org_id == existing_org.id
            ).first()
            
            if existing_admin:
                logger.warning(f"PRODUCT_SUPPORT_ADMIN user already exists: {existing_admin.username} ({existing_admin.email})")
                if return_response:
                    return InitResponse(
                        message="System already initialized",
                        organization_id=str(existing_org.id),
                        user_id=str(existing_admin.id),
                        username=existing_admin.username,
                        email=existing_admin.email,
                        role=existing_admin.role,
                        warning="System was already initialized. No changes made."
                    )
                logger.info("Initialization already completed. Exiting.")
                return
            
            logger.info("Organization exists but no PRODUCT_SUPPORT_ADMIN user found. Creating admin user...")
            org_id = existing_org.id
        else:
            # Create QueryGuardAI organization
            logger.info("Creating QueryGuardAI organization...")
            org = Organization(
                name="QueryGuardAI",
                is_active=True
            )
            db.add(org)
            db.commit()
            db.refresh(org)
            org_id = org.id
            logger.info(f"✅ Created QueryGuardAI organization with ID: {org_id}")
        
        # Hardcoded admin credentials
        admin_username = "admin"
        admin_email = "admin@queryguardai.com"
        admin_password = "Admin@123"
        
        # Check if admin user already exists
        existing_user = db.query(User).filter(
            User.username == admin_username
        ).first()
        
        if existing_user:
            logger.warning(f"User with username '{admin_username}' already exists.")
            if return_response:
                return InitResponse(
                    message="User already exists",
                    organization_id=str(org_id),
                    user_id=str(existing_user.id),
                    username=existing_user.username,
                    email=existing_user.email,
                    role=existing_user.role,
                    warning=f"User '{admin_username}' already exists. No new user created."
                )
            logger.info("If you want to create a new admin user, use a different username.")
            return
        
        # Check if email already exists
        existing_email = db.query(User).filter(
            User.email == admin_email
        ).first()
        
        if existing_email:
            logger.warning(f"User with email '{admin_email}' already exists.")
            if return_response:
                return InitResponse(
                    message="Email already exists",
                    organization_id=str(org_id),
                    user_id=str(existing_email.id),
                    username=existing_email.username,
                    email=existing_email.email,
                    role=existing_email.role,
                    warning=f"Email '{admin_email}' already exists. No new user created."
                )
            logger.info("If you want to create a new admin user, use a different email.")
            return
        
        # Create PRODUCT_SUPPORT_ADMIN user
        logger.info(f"Creating PRODUCT_SUPPORT_ADMIN user: {admin_username}")
        admin_user = User(
            username=admin_username,
            email=admin_email,
            password_hash=hash_password(admin_password),
            org_id=org_id,
            role=PRODUCT_SUPPORT_ADMIN,
            is_active=True
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        logger.info("=" * 60)
        logger.info("✅ Initialization completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Organization: QueryGuardAI (ID: {org_id})")
        logger.info(f"Admin User: {admin_username}")
        logger.info(f"Email: {admin_email}")
        logger.info(f"Role: {PRODUCT_SUPPORT_ADMIN}")
        logger.info("=" * 60)
        logger.warning("⚠️  IMPORTANT: Change the default password after first login!")
        logger.info("=" * 60)
        
        if return_response:
            return InitResponse(
                message="System initialized successfully",
                organization_id=str(org_id),
                user_id=str(admin_user.id),
                username=admin_username,
                email=admin_email,
                role=PRODUCT_SUPPORT_ADMIN,
                warning="⚠️  IMPORTANT: Change the default password (Admin@123) after first login!"
            )
        
    except IntegrityError:
        db.rollback()
        logger.exception("Initialization failed due to IntegrityError")
        if return_response:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Initialization failed due to database constraint violation"
            )
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"❌ Error during initialization: {str(e)}")
        if return_response:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Initialization failed: {str(e)}"
            )
        raise
    finally:
        if close_db:
            db.close()


# FastAPI Endpoint
@router.post("/", response_model=InitResponse, status_code=status.HTTP_201_CREATED)
@version(1, 0)
def initialize_system_endpoint(db: Session = Depends(get_db)):
    """
    TEMPORARY ENDPOINT: Initialize QueryGuardAI system.
    
    Creates:
    1. QueryGuardAI organization (if it doesn't exist)
    2. First PRODUCT_SUPPORT_ADMIN user with hardcoded credentials
    
    ⚠️  WARNING: This endpoint should be removed or protected after initial setup!
    ⚠️  Default credentials: admin / admin@queryguardai.com / Admin@123
    
    This is a one-time setup endpoint. It will not create duplicate organizations or users.
    """
    logger.warning("⚠️  TEMPORARY INIT ENDPOINT CALLED - This should be removed after initial setup!")
    return create_initial_admin(db=db, return_response=True)


if __name__ == "__main__":
    logger.info("Starting QueryGuardAI initialization script...")
    logger.info("This script creates the initial PRODUCT_SUPPORT_ADMIN user and QueryGuardAI organization.")
    
    # Check if DATABASE_URL is set
    if not os.getenv("DATABASE_URL"):
        logger.error("❌ DATABASE_URL environment variable is not set!")
        logger.error("Please set DATABASE_URL before running this script.")
        sys.exit(1)
    
    try:
        create_initial_admin()
        logger.info("✅ Script completed successfully!")
    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}")
        sys.exit(1)

