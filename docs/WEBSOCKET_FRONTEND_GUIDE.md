# WebSocket Chat Implementation Guide for Frontend Developers

This document provides a comprehensive step-by-step guide for implementing the QueryGuard WebSocket chat functionality in your frontend application.

## Quick Reference

**Production API Base URL:** `https://queryguard-backend-dev.onrender.com`  
**Production WebSocket URL:** `wss://queryguard-backend-dev.onrender.com/chat/ws/{org_id}/{user_id}?{params}`

**⚠️ IMPORTANT: JWT Token is REQUIRED for Chatbot Tools**

**Quick Connection Example:**
```javascript
// JWT token is REQUIRED for chatbot tools (lineage, query history, PR/repo, code suggestions, Jira)
const token = localStorage.getItem('jwt_token'); // Get from login endpoint
const ws = new WebSocket(
  `wss://queryguard-backend-dev.onrender.com/chat/ws/YOUR_ORG_ID/YOUR_USER_ID?session_id=abc123&user_name=John%20Doe&token=${encodeURIComponent(token)}`
);
```

## Table of Contents
1. [WebSocket Connection Overview](#websocket-connection-overview)
2. [Connection URL and Parameters](#connection-url-and-parameters)
3. [Step-by-Step Connection Flow](#step-by-step-connection-flow)
4. [Message Types and Formats](#message-types-and-formats)
5. [Authentication Flow](#authentication-flow)
6. [Thread Management](#thread-management)
7. [Complete Implementation Example](#complete-implementation-example)

---

## WebSocket Connection Overview

The QueryGuard chat system uses WebSockets for real-time bidirectional communication. The connection enables:
- Real-time chat messages
- AI responses with streaming support
- Typing indicators
- User presence (online/offline status)
- Chat history persistence
- Multi-user collaboration within an organization

---

## Connection URL and Parameters

### Base URL Structure

```
ws://<host>:<port>/chat/ws/<org_id>/<user_id>?<query_parameters>
```

**Production Host:**
- **API Base URL:** `https://queryguard-backend-dev.onrender.com`
- **WebSocket URL:** `wss://queryguard-backend-dev.onrender.com/chat/ws/<org_id>/<user_id>?<query_parameters>`

**Examples:**

**Production:**
```
wss://queryguard-backend-dev.onrender.com/chat/ws/76d33fb3-6062-456b-a211-4aec9971f8be/test-user?session_id=abc123&user_name=Test%20User&token=eyJhbGc...&thread_id=xyz789
```

**Local Development:**
```
ws://localhost:8000/chat/ws/76d33fb3-6062-456b-a211-4aec9971f8be/test-user?session_id=abc123&user_name=Test%20User&token=eyJhbGc...&thread_id=xyz789
```

**Note:** Use `wss://` (secure WebSocket) for HTTPS hosts and `ws://` for HTTP hosts.

### URL Path Parameters (Required)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `org_id` | String (UUID) | Organization ID - identifies which organization the user belongs to | `76d33fb3-6062-456b-a211-4aec9971f8be` |
| `user_id` | String | User ID - unique identifier for the user | `test-user` or `user-uuid-123` |

### Query Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `session_id` | String | No* | Unique session identifier. If not provided, server generates one. | `test-session-abc123` |
| `user_name` | String | No | Display name for the user (URL encoded) | `John%20Doe` |
| `token` | String (JWT) | **Yes** | **JWT authentication token. REQUIRED for:** chatbot tools (lineage, query history, PR/repo, code suggestions, Jira), chat history, thread management, and organization-specific features | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |
| `thread_id` | String (UUID) | No | Existing chat thread ID. If provided, messages are saved to this thread. If not provided and user is authenticated, a new thread is created automatically. | `550e8400-e29b-41d4-a716-446655440000` |

**Notes:**
- *`session_id`*: If omitted, server auto-generates a UUID. However, it's recommended to generate one client-side for better tracking.
- **`token` (REQUIRED):** The JWT authentication token is **REQUIRED** to use chatbot tools. Without token, the system works in "Simple QA mode" with only basic vector search (no tools, no history, no thread persistence). With token, you get:
  - ✅ All chatbot tools (lineage, query history, PR/repo analysis, code suggestions, Jira ticket creation)
  - ✅ Agent-based reasoning and tool selection
  - ✅ Chat history persistence
  - ✅ Thread management
  - ✅ Organization-specific data access

---

## Step-by-Step Connection Flow

### Step 1: Prepare Connection Parameters

```javascript
// 1. Get user credentials (from your auth system)
const orgId = "76d33fb3-6062-456b-a211-4aec9971f8be";  // Required
const userId = "user-123";                              // Required
const userName = "John Doe";                            // Optional but recommended
const token = localStorage.getItem("jwt_token");        // REQUIRED for chatbot tools and full features
const threadId = selectedThreadId || null;              // Optional - existing thread or null for new

// IMPORTANT: Token is REQUIRED for chatbot tools to work
if (!token) {
    console.warn('JWT token is required for chatbot tools (lineage, query history, PR/repo, code suggestions, Jira)');
}

// 2. Generate session ID (or let server generate it)
const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// 3. Build WebSocket URL
// Option 1: Use current window location (for same-origin)
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const host = window.location.host; // e.g., "localhost:8000" or "queryguard-backend-dev.onrender.com"
const baseUrl = `${protocol}//${host}`;

// Option 2: Use production host directly
// const baseUrl = 'wss://queryguard-backend-dev.onrender.com';

let wsUrl = `${baseUrl}/chat/ws/${orgId}/${userId}?session_id=${sessionId}&user_name=${encodeURIComponent(userName)}`;

// Add JWT token (REQUIRED for chatbot tools)
if (token) {
    wsUrl += `&token=${encodeURIComponent(token)}`;
} else {
    console.warn('WARNING: No JWT token provided. Chatbot tools will not be available. Only basic Q&A will work.');
}

if (threadId) {
    wsUrl += `&thread_id=${encodeURIComponent(threadId)}`;
}
```

### Step 2: Establish WebSocket Connection

```javascript
const ws = new WebSocket(wsUrl);

// Connection opened
ws.onopen = function(event) {
    console.log('WebSocket connected');
    // Enable UI elements for sending messages
    enableChatInput();
};

// Handle incoming messages
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    handleIncomingMessage(data);
};

// Connection closed
ws.onclose = function(event) {
    console.log('WebSocket disconnected');
    // Disable UI elements
    disableChatInput();
    // Optionally attempt reconnection
};

// Connection error
ws.onerror = function(error) {
    console.error('WebSocket error:', error);
    // Show error to user
};
```

### Step 3: Handle Initial Connection Messages

When the connection is established, you'll receive a welcome message:

```javascript
function handleIncomingMessage(data) {
    switch(data.type) {
        case 'system_message':
            // Welcome message or system notifications
            if (data.data.status === 'connected') {
                console.log('Connected! Session:', data.data.session_id);
            }
            // Thread creation notification
            if (data.data.status === 'thread_created' && data.data.thread_id) {
                const newThreadId = data.data.thread_id;
                console.log('New thread created:', newThreadId);
                // Update your UI with the new thread ID
                updateCurrentThread(newThreadId);
            }
            break;
            
        case 'user_status':
            // User joined/left notifications
            console.log(data.data.message);
            break;
            
        // ... other message types (see below)
    }
}
```

---

## Message Types and Formats

### Outgoing Messages (Client → Server)

#### 1. Chat Message

Send a user message to the AI:

```javascript
const message = {
    type: 'chat_message',
    content: 'What queries are impacted by this schema change?',
    thread_id: currentThreadId  // Optional: include if you want to save to a specific thread
};

ws.send(JSON.stringify(message));
```

**Fields:**
- `type` (required): `"chat_message"`
- `content` (required): The user's message text
- `thread_id` (optional): Thread ID to save message to (if authenticated)

#### 2. Typing Indicator

Send typing status:

```javascript
// User started typing
const typingStart = {
    type: 'typing',
    data: {
        is_typing: true
    }
};
ws.send(JSON.stringify(typingStart));

// User stopped typing
const typingStop = {
    type: 'typing',
    data: {
        is_typing: false
    }
};
ws.send(JSON.stringify(typingStop));
```

**Fields:**
- `type` (required): `"typing"`
- `data.is_typing` (required): `true` or `false`

#### 3. Ping (Keep-Alive)

Optional ping to keep connection alive:

```javascript
const ping = {
    type: 'ping'
};
ws.send(JSON.stringify(ping));
```

### Incoming Messages (Server → Client)

All incoming messages follow this structure:

```typescript
interface WebSocketMessage {
    type: string;
    data: any;
    sender_id?: string;
    sender_name?: string;
    room_id?: string;
}
```

#### 1. System Message

System notifications and status updates:

```javascript
{
    type: 'system_message',
    data: {
        message: 'Welcome to QueryGuard chat! Session abc123 established.',
        session_id: 'abc123',
        user_name: 'John Doe',
        status: 'connected' | 'thread_created' | 'error',
        thread_id: '550e8400-e29b-41d4-a716-446655440000'  // Present if status is 'thread_created'
    },
    sender_id: 'system',
    room_id: 'org-uuid'
}
```

**When received:**
- On connection: `status: 'connected'`
- When new thread is created: `status: 'thread_created'` with `thread_id`
- On errors: `status: 'error'` with error details

#### 2. Chat Message (Echo)

Echo of user's message (broadcast to all users in org):

```javascript
{
    type: 'chat_message',
    data: {
        content: 'What queries are impacted?',
        sender_id: 'user-123',
        sender_name: 'John Doe',
        message_type: 'user',
        thread_id: '550e8400-e29b-41d4-a716-446655440000'
    },
    sender_id: 'user-123',
    room_id: 'org-uuid'
}
```

#### 3. AI Response

AI-generated response:

```javascript
{
    type: 'ai_response',
    data: {
        response: 'Based on the schema change, I found 5 impacted queries...',
        thread_id: '550e8400-e29b-41d4-a716-446655440000',  // Thread ID (new or existing)
        impacted_queries: [...],  // Optional: array of impacted queries
        pr_repo_data: {...},      // Optional: PR/repo analysis data
        code_suggestions: [...],   // Optional: code suggestions
        jira_ticket: {...},       // Optional: created Jira ticket info
        processing_time: 2.45     // Processing time in seconds
    },
    sender_id: 'ai_assistant',
    room_id: 'org-uuid'
}
```

**Important:** The `thread_id` in the response is the thread where the message was saved. If you didn't provide a `thread_id` in the connection or message, and the user is authenticated, a new thread will be created and returned here.

#### 4. Typing Indicator

Typing status from other users or AI:

```javascript
{
    type: 'typing',
    data: {
        is_typing: true,
        sender_id: 'user-123' | 'ai_assistant',
        sender_name: 'John Doe' | 'QueryGuard AI'
    },
    sender_id: 'user-123',
    room_id: 'org-uuid'
}
```

#### 5. User Status

User join/leave notifications:

```javascript
{
    type: 'user_status',
    data: {
        message: 'User John Doe joined the chat',
        user_id: 'user-123',
        user_name: 'John Doe',
        status: 'online' | 'offline',
        action: 'joined' | 'left'
    },
    sender_id: 'user-123',
    room_id: 'org-uuid'
}
```

#### 6. Error Message

Error notifications:

```javascript
{
    type: 'error',
    data: {
        error_code: 'INVALID_JSON' | 'PROCESSING_ERROR' | 'AUTH_ERROR',
        error_message: 'Invalid message format. Please send valid JSON.'
    },
    sender_id: 'system'
}
```

---

## Authentication Flow

### ⚠️ IMPORTANT: JWT Token is REQUIRED for Chatbot Tools

**The JWT authentication token is REQUIRED to use any chatbot tools.** Without it, the chatbot will only provide basic Q&A from the vector database and cannot access any tools.

### Without Token (Simple QA Mode - Limited Functionality)

**What works:**
- Basic chat functionality
- Simple Q&A responses from vector database
- Real-time messaging
- Typing indicators

**What DOESN'T work (REQUIRES JWT TOKEN):**
- ❌ **Chatbot tools** (lineage, query history, PR/repo analysis, code suggestions, Jira)
- ❌ Agent-based reasoning and tool selection
- ❌ Chat history persistence
- ❌ Thread management
- ❌ User authentication
- ❌ Organization-specific data access

**Use case:** Only for quick testing without tool functionality. **Not recommended for production use.**

### With Token (Full Features - REQUIRED for Tools)

**Steps:**

1. **Get JWT Token** (via login endpoint):
```javascript
const response = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        email: 'user@example.com',
        password: 'password123'
    })
});

const { access_token } = await response.json();
localStorage.setItem('jwt_token', access_token);
```

2. **Include Token in WebSocket URL:**
```javascript
const token = localStorage.getItem('jwt_token');
const wsUrl = `${baseUrl}/chat/ws/${orgId}/${userId}?session_id=${sessionId}&user_name=${encodeURIComponent(userName)}&token=${encodeURIComponent(token)}`;
```

3. **Server Behavior:**
   - Validates token
   - Resolves `org_id` from authenticated user (overrides URL parameter)
   - Enables chat history persistence
   - Creates/manages threads automatically
   - Provides organization-specific context

**What works:**
- Everything from Simple QA mode, plus:
- ✅ **All chatbot tools** (lineage extraction, query impact analysis, PR/repo analysis, code suggestions, Jira ticket creation)
- ✅ Agent-based reasoning and intelligent tool selection
- ✅ Chat history across sessions
- ✅ Thread creation and management
- ✅ Message persistence
- ✅ User-specific context
- ✅ Organization-specific data access

**⚠️ Without JWT token, chatbot tools are NOT available.**

---

## Thread Management

### Understanding Threads

A **thread** is a conversation container that groups related messages together. Threads enable:
- Chat history persistence
- Context continuity across sessions
- Organization of conversations

### Thread Lifecycle

#### 1. Create New Thread (REST API)

Before connecting, you can create a thread:

```javascript
const response = await fetch('/chat/threads', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        title: 'New Chat'  // Optional: thread title
    })
});

const thread = await response.json();
// thread.id is the thread_id to use in WebSocket connection
```

#### 2. Connect with Existing Thread

```javascript
const wsUrl = `${baseUrl}/chat/ws/${orgId}/${userId}?session_id=${sessionId}&user_name=${encodeURIComponent(userName)}&token=${encodeURIComponent(token)}&thread_id=${existingThreadId}`;
```

#### 3. Auto-Create Thread on First Message

If you connect without a `thread_id` but with a `token`, the server will automatically create a new thread when you send the first message. The `thread_id` will be returned in the `ai_response` or `system_message`:

```javascript
// Connect without thread_id
const wsUrl = `${baseUrl}/chat/ws/${orgId}/${userId}?session_id=${sessionId}&token=${encodeURIComponent(token)}`;

// Send first message
ws.send(JSON.stringify({
    type: 'chat_message',
    content: 'Hello!'
}));

// Receive thread_id in response
// In ai_response: data.thread_id
// Or in system_message: data.thread_id (if status is 'thread_created')
```

#### 4. Load Thread History (REST API)

```javascript
// Get all threads for user
const response = await fetch('/chat/threads', {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});

const threads = await response.json();
// Returns: [{ id, title, message_count, created_at, last_message_at, ... }]

// Get specific thread with messages
const threadResponse = await fetch(`/chat/threads/${threadId}`, {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});

const thread = await threadResponse.json();
// Returns: { id, title, messages: [{ role, content, created_at }, ...], ... }
```

---

## Complete Implementation Example

Here's a complete React/TypeScript example:

```typescript
import { useState, useEffect, useRef } from 'react';

interface WebSocketMessage {
    type: string;
    data: any;
    sender_id?: string;
    sender_name?: string;
}

export const useChatWebSocket = (
    orgId: string,
    userId: string,
    userName: string,
    token: string | null,  // REQUIRED for chatbot tools
    threadId: string | null
) => {
    const [ws, setWs] = useState<WebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [messages, setMessages] = useState<any[]>([]);
    const [currentThreadId, setCurrentThreadId] = useState<string | null>(threadId);
    const sessionIdRef = useRef<string>(`session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);

    const connect = () => {
        // Use environment variable or default to production
        const API_HOST = process.env.REACT_APP_API_HOST || 'queryguard-backend-dev.onrender.com';
        const protocol = API_HOST.includes('localhost') ? 'ws:' : 'wss:';
        const baseUrl = `${protocol}//${API_HOST}`;

        let wsUrl = `${baseUrl}/chat/ws/${orgId}/${userId}?session_id=${sessionIdRef.current}&user_name=${encodeURIComponent(userName)}`;

        // JWT token is REQUIRED for chatbot tools
        if (token) {
            wsUrl += `&token=${encodeURIComponent(token)}`;
        } else {
            console.warn('WARNING: No JWT token provided. Chatbot tools will not be available.');
        }

        if (currentThreadId) {
            wsUrl += `&thread_id=${encodeURIComponent(currentThreadId)}`;
        }

        const websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            setIsConnected(true);
            setWs(websocket);
            console.log('WebSocket connected');
        };

        websocket.onmessage = (event) => {
            const data: WebSocketMessage = JSON.parse(event.data);
            handleMessage(data);
        };

        websocket.onclose = () => {
            setIsConnected(false);
            setWs(null);
            console.log('WebSocket disconnected');
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    };

    const handleMessage = (data: WebSocketMessage) => {
        switch (data.type) {
            case 'system_message':
                if (data.data.status === 'thread_created' && data.data.thread_id) {
                    setCurrentThreadId(data.data.thread_id);
                }
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: data.data.message,
                    timestamp: new Date()
                }]);
                break;

            case 'chat_message':
                setMessages(prev => [...prev, {
                    type: 'user',
                    content: data.data.content,
                    sender: data.data.sender_name || data.data.sender_id,
                    timestamp: new Date()
                }]);
                break;

            case 'ai_response':
                if (data.data.thread_id && !currentThreadId) {
                    setCurrentThreadId(data.data.thread_id);
                }
                setMessages(prev => [...prev, {
                    type: 'ai',
                    content: data.data.response,
                    threadId: data.data.thread_id,
                    metadata: {
                        impactedQueries: data.data.impacted_queries,
                        prRepoData: data.data.pr_repo_data,
                        codeSuggestions: data.data.code_suggestions,
                        jiraTicket: data.data.jira_ticket,
                        processingTime: data.data.processing_time
                    },
                    timestamp: new Date()
                }]);
                break;

            case 'typing':
                // Handle typing indicator
                break;

            case 'user_status':
                // Handle user join/leave
                break;

            case 'error':
                console.error('WebSocket error:', data.data);
                break;
        }
    };

    const sendMessage = (content: string) => {
        if (!ws || !isConnected) return;

        const message = {
            type: 'chat_message',
            content: content,
            thread_id: currentThreadId
        };

        ws.send(JSON.stringify(message));
    };

    const sendTyping = (isTyping: boolean) => {
        if (!ws || !isConnected) return;

        ws.send(JSON.stringify({
            type: 'typing',
            data: { is_typing: isTyping }
        }));
    };

    const disconnect = () => {
        if (ws) {
            ws.close();
        }
    };

    useEffect(() => {
        return () => {
            disconnect();
        };
    }, []);

    return {
        connect,
        disconnect,
        sendMessage,
        sendTyping,
        isConnected,
        messages,
        currentThreadId
    };
};
```

---

## Best Practices

1. **Always handle reconnection:** Implement automatic reconnection logic for dropped connections.

2. **Store thread_id:** When you receive a `thread_id` from the server, store it and use it for subsequent connections.

3. **Error handling:** Always handle WebSocket errors gracefully and show user-friendly messages.

4. **Token management:** Store JWT tokens securely (httpOnly cookies or secure storage) and refresh them before expiration.

5. **Message queuing:** If the connection is not ready, queue messages and send them when connected.

6. **Cleanup:** Always close WebSocket connections when components unmount or users navigate away.

7. **Typing indicators:** Debounce typing indicators to avoid sending too many messages.

---

## REST API Endpoints (Related)

These REST endpoints work alongside the WebSocket connection:

**Production Base URL:** `https://queryguard-backend-dev.onrender.com`

- `GET /chat/threads` - Get all threads for authenticated user
- `POST /chat/threads` - Create a new thread
- `GET /chat/threads/{thread_id}` - Get thread with messages
- `POST /auth/login` - Get JWT token for authentication

**Example API Calls:**
```javascript
// Production
const API_BASE = 'https://queryguard-backend-dev.onrender.com';

// Login
const loginResponse = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: 'user@example.com', password: 'password' })
});

// Get threads
const threadsResponse = await fetch(`${API_BASE}/chat/threads`, {
    headers: { 'Authorization': `Bearer ${token}` }
});
```

---

## Troubleshooting

### Connection Fails
- Check that the WebSocket URL is correct (ws:// or wss://)
- Verify `org_id` and `user_id` are valid
- Check network/firewall settings

### Chatbot Tools Not Working
- **JWT token is REQUIRED** for chatbot tools to function
- Ensure `token` is included in WebSocket connection URL
- Verify token is valid and not expired
- Check that user is authenticated and belongs to the organization
- Without token, only basic Q&A is available (no tools)

### No Chat History
- Ensure `token` is included in connection URL
- Verify token is valid and not expired
- Check that `thread_id` is correct (if using existing thread)

### Messages Not Saving
- Token is required for message persistence
- Verify user has proper permissions
- Check server logs for errors

### Thread Not Created
- Token must be provided
- User must be authenticated
- Check that user belongs to the organization

---

## Summary

### Production Configuration

**API Base URL:** `https://queryguard-backend-dev.onrender.com`  
**WebSocket URL:** `wss://queryguard-backend-dev.onrender.com/chat/ws/{org_id}/{user_id}?{params}`

### Connection Requirements

**Required for Connection:**
- `org_id` (path parameter)
- `user_id` (path parameter)

**⚠️ REQUIRED for Chatbot Tools:**
- `token` (query parameter) - **JWT authentication token is REQUIRED** for chatbot tools (lineage, query history, PR/repo, code suggestions, Jira). Without token, only basic Q&A is available.

**Optional but Recommended:**
- `session_id` - For better session tracking
- `user_name` - For display purposes
- `thread_id` - To continue existing conversation

**Feature Comparison:**

| Feature | Without JWT Token | With JWT Token |
|---------|-------------------|----------------|
| Basic Q&A | ✅ Yes | ✅ Yes |
| Lineage Tool | ❌ **No** | ✅ Yes |
| Query History Tool | ❌ **No** | ✅ Yes |
| PR/Repo Tool | ❌ **No** | ✅ Yes |
| Code Suggestion Tool | ❌ **No** | ✅ Yes |
| Jira Tool | ❌ **No** | ✅ Yes |
| Agent Reasoning | ❌ **No** | ✅ Yes |
| Chat History | ❌ **No** | ✅ Yes |
| Thread Management | ❌ **No** | ✅ Yes |

### Connection URL Pattern

**Production:**
```
wss://queryguard-backend-dev.onrender.com/chat/ws/{org_id}/{user_id}?session_id={session_id}&user_name={user_name}&token={token}&thread_id={thread_id}
```

**Local Development:**
```
ws://localhost:8000/chat/ws/{org_id}/{user_id}?session_id={session_id}&user_name={user_name}&token={token}&thread_id={thread_id}
```

### Environment Configuration

For frontend applications, configure the API host:

```javascript
// .env file
REACT_APP_API_HOST=queryguard-backend-dev.onrender.com

// Or for local development
REACT_APP_API_HOST=localhost:8000
```

The WebSocket connection enables real-time, bidirectional communication with automatic thread management, chat history, and AI-powered responses.

