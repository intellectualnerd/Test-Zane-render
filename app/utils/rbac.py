"""
Role-Based Access Control (RBAC) utilities for QueryGuardAI Backend

Role Hierarchy (from highest to lowest control):
1. PRODUCT_SUPPORT_ADMIN - Full product access, cross-org (special role to manage all clients)
2. SYSTEM_ADMIN - Full product access within own organization only
3. ORGANIZATION_ADMIN - Organization-level access within own organization (can only create MEMBER users)
4. MEMBER - View-only access within own organization

Role Assignment Rules:
- PRODUCT_SUPPORT_ADMIN: Can assign PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER
- SYSTEM_ADMIN: Can assign SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER (within their org)
- ORGANIZATION_ADMIN: Can only assign MEMBER (cannot create other Organization Admins)
- MEMBER: Cannot assign any roles
"""

from fastapi import HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.utils.models import User, Organization
from app.utils.auth_deps import get_current_user
import logging

logger = logging.getLogger("rbac")

# Role constants
PRODUCT_SUPPORT_ADMIN = "PRODUCT_SUPPORT_ADMIN"
SYSTEM_ADMIN = "SYSTEM_ADMIN"
ORGANIZATION_ADMIN = "ORGANIZATION_ADMIN"
MEMBER = "MEMBER"

# Valid roles list
VALID_ROLES = [PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER]

# Role hierarchy (higher number = more privileges)
ROLE_HIERARCHY = {
    PRODUCT_SUPPORT_ADMIN: 4,
    SYSTEM_ADMIN: 3,
    ORGANIZATION_ADMIN: 2,
    MEMBER: 1
}


def has_role(user: User, role: str) -> bool:
    """Check if user has a specific role"""
    return user.role == role


def has_any_role(user: User, roles: List[str]) -> bool:
    """Check if user has any of the specified roles"""
    return user.role in roles


def has_minimum_role(user: User, minimum_role: str) -> bool:
    """Check if user has at least the minimum role level"""
    user_level = ROLE_HIERARCHY.get(user.role, 0)
    min_level = ROLE_HIERARCHY.get(minimum_role, 0)
    return user_level >= min_level


def can_assign_role(assigner: User, target_role: str, target_org_id: Optional[str] = None) -> bool:
    """
    Check if a user can assign a specific role to another user.
    
    Args:
        assigner: User who wants to assign the role
        target_role: Role to be assigned
        target_org_id: Organization ID of the target user (for cross-org checks)
    
    Returns:
        True if assignment is allowed, False otherwise
    """
    # MEMBER cannot assign any roles
    if assigner.role == MEMBER:
        return False
    
    # PRODUCT_SUPPORT_ADMIN can assign PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER
    if assigner.role == PRODUCT_SUPPORT_ADMIN:
        return target_role in [PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER]
    
    # SYSTEM_ADMIN can assign SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER (within their org)
    if assigner.role == SYSTEM_ADMIN:
        if target_role == PRODUCT_SUPPORT_ADMIN:
            return False
        # Can only assign within their own organization
        if target_org_id and str(assigner.org_id) != str(target_org_id):
            return False
        return target_role in [SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER]
    
    # ORGANIZATION_ADMIN can only assign MEMBER (within their org)
    if assigner.role == ORGANIZATION_ADMIN:
        if target_role != MEMBER:
            return False
        # Can only assign within their own organization
        if target_org_id and str(assigner.org_id) != str(target_org_id):
            return False
        return True
    
    return False


def can_access_organization(user: User, org_id: str) -> bool:
    """
    Check if user can access a specific organization.
    
    PRODUCT_SUPPORT_ADMIN can access any organization.
    Others can only access their own organization.
    """
    if user.role == PRODUCT_SUPPORT_ADMIN:
        return True
    return str(user.org_id) == str(org_id)


def require_role(required_role: str):
    """
    Dependency to require a specific role.
    Usage: current_user: User = Depends(require_role(PRODUCT_SUPPORT_ADMIN))
    """
    def role_checker(current_user: User = Depends(get_current_user)):
        if not has_role(current_user, required_role):
            logger.warning(f"Role check failed: user {current_user.id} has role {current_user.role}, required {required_role}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {required_role}"
            )
        return current_user
    return role_checker


def require_any_role(required_roles: List[str]):
    """
    Dependency to require any of the specified roles.
    Usage: current_user: User = Depends(require_any_role([SYSTEM_ADMIN, ORGANIZATION_ADMIN]))
    """
    def role_checker(current_user: User = Depends(get_current_user)):
        if not has_any_role(current_user, required_roles):
            logger.warning(f"Role check failed: user {current_user.id} has role {current_user.role}, required one of {required_roles}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required one of roles: {', '.join(required_roles)}"
            )
        return current_user
    return role_checker


def require_minimum_role(minimum_role: str):
    """
    Dependency to require at least the minimum role level.
    Usage: current_user: User = Depends(require_minimum_role(SYSTEM_ADMIN))
    """
    def role_checker(current_user: User = Depends(get_current_user)):
        if not has_minimum_role(current_user, minimum_role):
            logger.warning(f"Role check failed: user {current_user.id} has role {current_user.role}, minimum required {minimum_role}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Minimum required role: {minimum_role}"
            )
        return current_user
    return role_checker


def require_organization_access(org_id_param: str = "org_id"):
    """
    Dependency to require organization access.
    Checks if user can access the organization specified in the request parameter.
    
    Usage: 
        @router.get("/organizations/{org_id}")
        def get_org(org_id: str, current_user: User = Depends(require_organization_access("org_id"))):
    """
    def org_access_checker(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        # This will be used as a dependency factory
        # The actual org_id will be checked in the endpoint
        return current_user
    
    return org_access_checker


def check_organization_access(user: User, org_id: str):
    """
    Helper function to check organization access and raise exception if denied.
    """
    if not can_access_organization(user, org_id):
        logger.warning(f"Organization access denied: user {user.id} (role: {user.role}, org: {user.org_id}) tried to access org {org_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You do not have permission to access this organization."
        )


def check_role_assignment(assigner: User, target_role: str, target_org_id: Optional[str] = None):
    """
    Helper function to check if role assignment is allowed and raise exception if denied.
    """
    if not can_assign_role(assigner, target_role, target_org_id):
        logger.warning(
            f"Role assignment denied: user {assigner.id} (role: {assigner.role}) "
            f"tried to assign role {target_role} to org {target_org_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to assign the role '{target_role}'."
        )


def require_connector_access():
    """
    Dependency to require access to connector management endpoints (Snowflake, dbt, GitHub, Jira).
    MEMBER role cannot access these endpoints.
    """
    def connector_access_checker(current_user: User = Depends(get_current_user)):
        if current_user.role == MEMBER:
            logger.warning(f"Connector access denied: user {current_user.id} has MEMBER role")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. MEMBER role cannot manage connectors."
            )
        return current_user
    return connector_access_checker


def require_organizations_endpoint_access():
    """
    Dependency to require access to organizations endpoints.
    Only PRODUCT_SUPPORT_ADMIN can access organizations endpoints.
    """
    def org_endpoint_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != PRODUCT_SUPPORT_ADMIN:
            logger.warning(f"Organizations endpoint access denied: user {current_user.id} has role {current_user.role}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only PRODUCT_SUPPORT_ADMIN can access organization management endpoints."
            )
        return current_user
    return org_endpoint_checker

