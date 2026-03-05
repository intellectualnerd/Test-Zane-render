# WebSocket-specific Pydantic models for QueryGuard Chat
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid

class MessageType(str, Enum):
    """Enumeration of WebSocket message types"""
    CHAT_MESSAGE = "chat_message"
    SYSTEM_MESSAGE = "system_message"
    TYPING_INDICATOR = "typing"
    JOIN_ROOM = "join"
    LEAVE_ROOM = "leave"
    ERROR = "error"
    CONNECTION_STATUS = "connection_status"
    AI_RESPONSE = "ai_response"
    USER_STATUS = "user_status"

class UserStatus(str, Enum):
    """User online status"""
    ONLINE = "online"
    OFFLINE = "offline"
    TYPING = "typing"
    IDLE = "idle"

class ChatMessageRequest(BaseModel):
    """Request model for chat messages through WebSocket"""
    type: MessageType = MessageType.CHAT_MESSAGE
    content: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class ChatMessageResponse(BaseModel):
    """Response model for chat messages"""
    message_id: str
    type: MessageType
    content: str
    sender_id: str
    sender_name: Optional[str] = None
    timestamp: str
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class AIResponseData(BaseModel):
    """Model for AI response data - matches ChatResponse structure"""
    response: str
    sources: List[Dict[str, Any]] = []
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    conversation_id: Optional[str] = None
    impacted_query_ids: Optional[List[str]] = []
    impacted_queries: Optional[List[Dict[str, Any]]] = []
    pr_repo_data: Optional[Dict[str, Any]] = None
    code_suggestions: Optional[Dict[str, Any]] = None
    jira_ticket: Optional[Dict[str, Any]] = None
    sender_id: Optional[str] = "ai_assistant"
    sender_name: Optional[str] = "QueryGuard AI"
    message_type: Optional[str] = "assistant"

class SystemMessageData(BaseModel):
    """Model for system message data"""
    message: str
    status: Optional[str] = None
    session_id: Optional[str] = None
    user_count: Optional[int] = None

class TypingIndicatorData(BaseModel):
    """Model for typing indicator data"""
    is_typing: bool
    sender_id: str
    sender_name: Optional[str] = None

class ConnectionStatusData(BaseModel):
    """Model for connection status data"""
    status: str  # "connected", "disconnected", "reconnected"
    session_id: str
    timestamp: str
    user_count: Optional[int] = None

class ErrorMessageData(BaseModel):
    """Model for error message data"""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None

class WebSocketMessageData(BaseModel):
    """Generic WebSocket message data model"""
    type: MessageType
    data: Union[
        Dict[str, Any], 
        AIResponseData, 
        SystemMessageData, 
        TypingIndicatorData, 
        ConnectionStatusData, 
        ErrorMessageData,
        str
    ]
    sender_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    room_id: Optional[str] = None  # org_id or conversation_id

class ChatSessionInfo(BaseModel):
    """Model for chat session information"""
    session_id: str
    org_id: str
    user_id: str
    user_name: Optional[str] = None
    status: UserStatus = UserStatus.ONLINE
    created_at: str
    last_activity: str
    connection_count: int = 1

class ChatRoomInfo(BaseModel):
    """Model for chat room information"""
    room_id: str  # org_id or conversation_id
    room_name: Optional[str] = None
    participants: List[ChatSessionInfo] = []
    message_count: Optional[int] = None
    created_at: Optional[str] = None
    last_activity: Optional[str] = None

class WebSocketConnectionRequest(BaseModel):
    """Model for WebSocket connection parameters"""
    org_id: str = Field(..., description="Organization ID")
    user_id: str = Field(..., description="User ID")
    user_name: Optional[str] = Field(None, description="User display name")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    auth_token: Optional[str] = Field(None, description="Authentication token")

class ChatStatsResponse(BaseModel):
    """Model for chat statistics response"""
    active_connections: int
    active_sessions: int
    organizations_with_sessions: int
    users_with_sessions: int
    total_org_sessions: int
    total_user_sessions: int
    server_uptime: Optional[str] = None