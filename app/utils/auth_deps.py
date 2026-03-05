# Authentication dependencies
# This module contains get_current_user to avoid circular imports with rbac.py

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from datetime import datetime
from app.database import get_db
from app.utils.models import User, UserToken
import os

security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"


def verify_token(raw_token: str):
    try:
        payload = jwt.decode(raw_token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get the current authenticated user from JWT token"""
    raw_token = credentials.credentials
    payload = verify_token(raw_token)
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Check if token exists and is not revoked
    token_record = db.query(UserToken).filter(
        UserToken.token == raw_token,
        UserToken.is_revoked == False,
        UserToken.expires_at > datetime.utcnow()
    ).first()
    
    if not token_record:
        raise HTTPException(status_code=401, detail="Token revoked or expired")

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user


def get_user_from_token(token: str, db: Session) -> User:
    """
    Get user from JWT token for WebSocket connections.
    This is a synchronous version that can be used in WebSocket handlers.
    
    Args:
        token: JWT token string
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid, expired, or user not found
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # Check if token exists and is not revoked
        token_record = db.query(UserToken).filter(
            UserToken.token == token,
            UserToken.is_revoked == False,
            UserToken.expires_at > datetime.utcnow()
        ).first()
        
        if not token_record:
            raise HTTPException(status_code=401, detail="Token revoked or expired")

        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


