# WebSocket Chat Integration - Complete Documentation

## 📋 Overview

This document explains how the REST chat endpoint logic has been fully integrated into the WebSocket implementation, providing real-time chat with all the sophisticated features including LLM agents, tools, and classification.

## 🔄 What Was Changed

### 1. Authentication Support for WebSocket (`app/utils/auth_deps.py`)

**Added Function**: `get_user_from_token(token: str, db: Session) -> User`

**Purpose**: 
- Enables authentication for WebSocket connections
- Validates JWT tokens synchronously (needed for WebSocket handlers)
- Returns authenticated user object with org_id

**How It Works**:
```python
# Validates token, checks if it's revoked/expired
# Returns User object with org_id, role, etc.
current_user = get_user_from_token(token, db)
```

---

### 2. Updated WebSocket Models (`app/utils/websocket_models.py`)

**Updated**: `AIResponseData` class

**Added Fields**:
- `impacted_query_ids`: List of query IDs affected by changes
- `impacted_queries`: Full query details
- `pr_repo_data`: PR and repository analysis data
- `code_suggestions`: Code suggestions from PR analysis
- `jira_ticket`: Created Jira ticket information
- `conversation_id`: Conversation tracking
- `sender_id`, `sender_name`, `message_type`: Metadata

**Why**: To match the full `ChatResponse` structure from REST endpoint, enabling all features via WebSocket.

---

### 3. WebSocket Endpoint Updates (`app/api/chat.py`)

#### 3.1 Authentication Integration

**Changed**: WebSocket endpoint now accepts `token` query parameter

**Before**:
```python
@router.websocket("/ws/{org_id}/{user_id}")
async def websocket_chat_endpoint(websocket, org_id, user_id, ...)
```

**After**:
```python
@router.websocket("/ws/{org_id}/{user_id}")
async def websocket_chat_endpoint(
    websocket, org_id, user_id, 
    token: Optional[str] = Query(None)  # NEW
)
```

**How It Works**:
1. If `token` is provided, user is authenticated
2. `org_id` is resolved from authenticated user (more secure)
3. `user_id` is set from authenticated user
4. If token is invalid, connection is rejected with code 4001
5. If no token, connection works but with limited functionality (fallback mode)

**Connection URL Example**:
```
ws://localhost:8000/chat/ws/{org_id}/{user_id}?token=YOUR_JWT_TOKEN&user_name=John
```

---

#### 3.2 Full Chat Logic Integration

**Changed**: `handle_chat_message()` function completely rewritten

**Before**: Simple QA chain with vector DB
```python
qa_chain = get_qa_chain(org_id, k=5)
result = qa_chain.invoke({"query": content})
```

**After**: Full chat logic with:
1. **LLM Classification**: Determines if message needs tools
2. **Agent-Based Processing**: Uses LangChain agents with multiple tools
3. **Tool Integration**: 
   - Lineage extraction
   - Query impact analysis
   - PR/Repo analysis
   - Code suggestions
   - Jira ticket creation
4. **Data Extraction**: Extracts impacted queries, PR data, etc.
5. **Fallback Mode**: If no auth, uses simple QA chain

**Flow Diagram**:
```
User Message
    ↓
Authentication Check
    ↓
┌─────────────────┐
│ Has Auth Token? │
└─────────────────┘
    ↓ Yes              ↓ No
LLM Classification    Simple QA Chain
    ↓
Tool Selection
    ↓
Agent Processing
    ↓
Data Extraction
    ↓
WebSocket Response (with all data)
```

---

## 🎯 Key Features Integrated

### 1. **LLM Classification**
- Analyzes user message to determine intent
- Categories: `lineage`, `impact`, `pr`, `code`, `jira`, `other`
- Routes to appropriate tools or conversational response

### 2. **Agent-Based Processing**
- Uses LangChain `ZERO_SHOT_REACT_DESCRIPTION` agent
- Automatically selects and uses appropriate tools
- Handles complex multi-step queries

### 3. **Tool Integration**
All tools are org-aware and user-aware:

- **`extract_lineage`**: Data lineage extraction
- **`query_history_search`**: Find impacted queries
- **`pr_repo_analysis`**: Analyze PRs and repositories
- **`code_suggestion`**: Generate code suggestions
- **`create_jira_ticket`**: Create Jira tickets

### 4. **Rich Response Data**
WebSocket responses now include:
- `response`: AI-generated text response
- `impacted_query_ids`: List of affected query IDs
- `impacted_queries`: Full query details with SQL
- `pr_repo_data`: PR analysis results
- `code_suggestions`: Code suggestions by file
- `jira_ticket`: Created Jira ticket info
- `processing_time`: Time taken to process
- `sources`: Source documents (for simple QA mode)

---

## 📡 WebSocket Message Format

### Client → Server (Chat Message)

**With Authentication (Full Features)**:
```json
{
  "type": "chat_message",
  "content": "What queries are impacted by changing the users table?",
  "conversation_id": "optional-conv-id",
  "k": 5,
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

**Without Authentication (Simple Mode)**:
```json
{
  "type": "chat_message",
  "content": "What is data lineage?"
}
```

### Server → Client (AI Response)

**Full Response (With Tools)**:
```json
{
  "type": "ai_response",
  "data": {
    "response": "Based on my analysis, here are the impacted queries...",
    "sources": [],
    "processing_time": 2.5,
    "conversation_id": "conv-123",
    "impacted_query_ids": ["uuid-1", "uuid-2"],
    "impacted_queries": [
      {
        "id": "uuid-1",
        "query_text": "SELECT * FROM users WHERE...",
        "database": "analytics",
        "schema": "public"
      }
    ],
    "pr_repo_data": null,
    "code_suggestions": null,
    "jira_ticket": null,
    "sender_id": "ai_assistant",
    "sender_name": "QueryGuard AI",
    "message_type": "assistant"
  },
  "sender_id": "ai_assistant",
  "timestamp": "2023-11-07T10:30:00Z",
  "message_id": "msg-123"
}
```

**Simple Response (No Auth)**:
```json
{
  "type": "ai_response",
  "data": {
    "response": "Data lineage is...",
    "sources": [
      {
        "content": "Document content...",
        "metadata": {"table": "users", "schema": "public"}
      }
    ],
    "processing_time": 1.2,
    "sender_id": "ai_assistant",
    "sender_name": "QueryGuard AI",
    "message_type": "assistant"
  }
}
```

---

## 🔐 Authentication Flow

### Step 1: Get JWT Token
```bash
POST /auth/login
{
  "username": "user",
  "password": "pass"
}
# Returns: {"access_token": "jwt_token_here"}
```

### Step 2: Connect WebSocket with Token
```javascript
const token = "your_jwt_token";
const ws = new WebSocket(
  `ws://localhost:8000/chat/ws/${orgId}/${userId}?token=${token}&user_name=John`
);
```

### Step 3: Server Validates Token
- Token is decoded and validated
- User is fetched from database
- `org_id` is resolved from user (more secure than URL param)
- Connection proceeds with authenticated user context

### Step 4: Full Features Available
- All tools are accessible
- Org-aware processing
- User-aware operations (e.g., Jira ticket creation)

---

## 🧪 Testing the Integration

### Test 1: With Authentication (Full Features)

```python
import asyncio
import websockets
import json

async def test_authenticated_chat():
    token = "your_jwt_token_here"
    org_id = "76d33fb3-6062-456b-a211-4aec9971f8be"
    user_id = "user-uuid"
    
    url = f"ws://localhost:8000/chat/ws/{org_id}/{user_id}?token={token}"
    
    async with websockets.connect(url) as ws:
        # Send message
        await ws.send(json.dumps({
            "type": "chat_message",
            "content": "What queries are impacted by changing the users table?"
        }))
        
        # Receive response
        response = await ws.recv()
        data = json.loads(response)
        print(f"Response: {data['data']['response']}")
        print(f"Impacted Queries: {data['data']['impacted_queries']}")
```

### Test 2: Without Authentication (Fallback Mode)

```python
# Same as above but without token parameter
url = f"ws://localhost:8000/chat/ws/{org_id}/{user_id}"
# Will use simple QA chain
```

### Test 3: Using Test Page

1. **Get token**: Login via `/auth/login`
2. **Open test page**: `http://localhost:8000/chat/test-page`
3. **Update connection URL** in browser console:
   ```javascript
   const token = "your_token";
   const wsUrl = `ws://localhost:8000/chat/ws/${orgId}/${userId}?token=${token}`;
   ```
4. **Send messages** and see full responses

---

## 🔍 Differences: REST vs WebSocket

| Feature | REST Endpoint | WebSocket |
|---------|--------------|-----------|
| **Authentication** | `Depends(get_current_user)` | Token in query param |
| **Real-time** | Request/Response | Bidirectional streaming |
| **Multi-user** | Single user | Broadcast to org |
| **Typing Indicators** | No | Yes |
| **Connection State** | Stateless | Stateful (session) |
| **Response Format** | Same | Same (with WebSocket wrapper) |
| **Tools & Agents** | ✅ Full support | ✅ Full support |
| **Data Extraction** | ✅ Yes | ✅ Yes |

---

## 🎨 Architecture Flow

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ WebSocket Connection (with token)
       ↓
┌─────────────────────────────────┐
│  WebSocket Endpoint             │
│  - Validate token               │
│  - Resolve org_id from user     │
│  - Create session               │
└──────┬──────────────────────────┘
       │
       ↓
┌─────────────────────────────────┐
│  handle_chat_message()          │
│  - Check authentication         │
│  - Route to appropriate handler │
└──────┬──────────────────────────┘
       │
       ├─── No Auth ──→ Simple QA Chain
       │
       └─── Has Auth ──→ Full Chat Logic
                          │
                          ├─── LLM Classification
                          │
                          ├─── Agent Processing
                          │
                          ├─── Tool Execution
                          │
                          └─── Data Extraction
                                │
                                ↓
                          WebSocket Response
                                │
                                ↓
                          Broadcast to Org
```

---

## 📝 Important Notes

### 1. **Organization ID Resolution**
- **With Auth**: `org_id` comes from authenticated user (secure)
- **Without Auth**: `org_id` from URL parameter (less secure)
- Always prefer authenticated connections for production

### 2. **Fallback Behavior**
- If no token provided, uses simple QA chain
- Still works but without tools/agents
- Good for testing or public access

### 3. **Error Handling**
- Authentication errors: Connection closed with code 4001
- Processing errors: Error message sent via WebSocket
- All errors are logged for debugging

### 4. **Performance**
- Agent processing can take 2-10 seconds
- Typing indicators show during processing
- Responses are broadcasted to all org users

### 5. **Security Considerations**
- Token is passed in query parameter (visible in logs)
- Consider using subprotocol or initial message for token in production
- Token is validated on every connection
- User's org_id is enforced (can't access other orgs)

---

## 🚀 Production Recommendations

1. **Use WSS (WebSocket Secure)** instead of WS
2. **Pass token in initial message** instead of query parameter
3. **Add rate limiting** per user/session
4. **Monitor agent processing times**
5. **Cache tool results** when appropriate
6. **Add connection pooling** for database sessions
7. **Implement message queuing** for high load

---

## 🔧 Troubleshooting

### Issue: "Authentication failed"
- **Cause**: Invalid or expired token
- **Solution**: Get new token via `/auth/login`

### Issue: "No authenticated user" warning
- **Cause**: Token not provided
- **Solution**: Add `token` query parameter

### Issue: Agent takes too long
- **Cause**: Complex query or tool execution
- **Solution**: Normal behavior, consider timeout handling

### Issue: Missing data in response
- **Cause**: Tool didn't return data or extraction failed
- **Solution**: Check logs, verify tool execution

---

## 📚 Related Files

- `app/api/chat.py` - Main chat endpoint and WebSocket handler
- `app/utils/auth_deps.py` - Authentication helpers
- `app/utils/websocket_models.py` - WebSocket message models
- `app/utils/websocket_manager.py` - WebSocket connection manager
- `app/tools/` - Tool implementations (lineage, impact, etc.)

---

**Last Updated**: Integration completed with full REST endpoint logic

