# Chat History Implementation

## 📋 Overview

This document explains the chat history feature that allows users to maintain conversation threads, similar to ChatGPT. Users can create new chats, access previous conversations, and delete chats.

## 🗄️ Database Schema

### ChatThread Table
Stores conversation threads (chat sessions):

```sql
CREATE TABLE chat_threads (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    user_id UUID NOT NULL REFERENCES users(id),
    title VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    last_message_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_chat_threads_user_org ON chat_threads(user_id, org_id);
CREATE INDEX idx_chat_threads_org ON chat_threads(org_id);
```

**Fields**:
- `id`: Unique thread identifier
- `org_id`: Organization ID (for multi-tenancy)
- `user_id`: User who owns the thread
- `title`: Thread title (auto-generated from first message or user-provided)
- `is_active`: Soft delete flag (False = deleted)
- `created_at`: Thread creation timestamp
- `updated_at`: Last update timestamp
- `last_message_at`: Timestamp of most recent message

### ChatMessage Table
Stores individual messages within threads:

```sql
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    org_id UUID NOT NULL REFERENCES organizations(id),
    user_id UUID NOT NULL REFERENCES users(id),
    role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    metadata JSONB,  -- Stores impacted_queries, pr_repo_data, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chat_messages_thread ON chat_messages(thread_id);
CREATE INDEX idx_chat_messages_user_org ON chat_messages(user_id, org_id);
CREATE INDEX idx_chat_messages_created ON chat_messages(created_at);
```

**Fields**:
- `id`: Unique message identifier
- `thread_id`: Parent thread
- `org_id`: Organization ID
- `user_id`: User who owns the message
- `role`: "user" or "assistant"
- `content`: Message text content
- `metadata`: JSONB field storing additional data:
  - `impacted_query_ids`: List of query IDs
  - `impacted_queries`: Full query details
  - `pr_repo_data`: PR analysis data
  - `code_suggestions`: Code suggestions
  - `jira_ticket`: Jira ticket info
  - `processing_time`: Time taken to process
  - `sources`: Source documents
- `created_at`: Message timestamp

## 🔌 API Endpoints

### 1. Create New Chat Thread
**POST** `/chat/threads`

**Request Body**:
```json
{
  "title": "Optional custom title"
}
```

**Response**:
```json
{
  "id": "thread-uuid",
  "org_id": "org-uuid",
  "user_id": "user-uuid",
  "title": "New Chat",
  "is_active": true,
  "created_at": "2023-11-07T10:00:00Z",
  "updated_at": null,
  "last_message_at": null,
  "message_count": 0
}
```

### 2. Get User's Chat Threads
**GET** `/chat/threads`

**Response**:
```json
[
  {
    "id": "thread-uuid-1",
    "title": "What is data lineage?",
    "last_message_at": "2023-11-07T10:30:00Z",
    "message_count": 5,
    "created_at": "2023-11-07T10:00:00Z"
  },
  {
    "id": "thread-uuid-2",
    "title": "Impact analysis",
    "last_message_at": "2023-11-07T09:15:00Z",
    "message_count": 3,
    "created_at": "2023-11-07T09:00:00Z"
  }
]
```

**Ordered by**: `last_message_at` DESC (most recent first)

### 3. Get Thread with Messages
**GET** `/chat/threads/{thread_id}`

**Response**:
```json
{
  "id": "thread-uuid",
  "title": "What is data lineage?",
  "created_at": "2023-11-07T10:00:00Z",
  "last_message_at": "2023-11-07T10:30:00Z",
  "messages": [
    {
      "id": "msg-uuid-1",
      "thread_id": "thread-uuid",
      "role": "user",
      "content": "What is data lineage?",
      "metadata": null,
      "created_at": "2023-11-07T10:00:00Z"
    },
    {
      "id": "msg-uuid-2",
      "thread_id": "thread-uuid",
      "role": "assistant",
      "content": "Data lineage is...",
      "metadata": {
        "processing_time": 2.5,
        "sources": [...]
      },
      "created_at": "2023-11-07T10:00:05Z"
    }
  ]
}
```

### 4. Update Thread Title
**PUT** `/chat/threads/{thread_id}/title?title=New Title`

**Response**:
```json
{
  "message": "Thread title updated",
  "title": "New Title"
}
```

### 5. Delete Thread
**DELETE** `/chat/threads/{thread_id}`

**Response**:
```json
{
  "message": "Chat thread deleted successfully"
}
```

**Note**: Soft delete (sets `is_active = False`)

## 🔌 WebSocket Integration

### Connection with Thread ID

**URL**: `ws://localhost:8000/chat/ws/{org_id}/{user_id}?token=JWT_TOKEN&thread_id=THREAD_ID`

**Query Parameters**:
- `token`: JWT authentication token (required for history)
- `thread_id`: Optional thread ID
  - If provided: Messages saved to this thread
  - If not provided: New thread created automatically

### Message Format

**Client → Server**:
```json
{
  "type": "chat_message",
  "content": "What is data lineage?",
  "thread_id": "optional-thread-id"  // Can override connection thread_id
}
```

**Server → Client** (with thread_id):
```json
{
  "type": "ai_response",
  "data": {
    "response": "...",
    "thread_id": "thread-uuid",
    "impacted_queries": [...],
    ...
  }
}
```

**Server → Client** (new thread created):
```json
{
  "type": "system_message",
  "data": {
    "message": "New chat thread created: thread-uuid",
    "thread_id": "thread-uuid",
    "status": "thread_created"
  }
}
```

## 🔄 How It Works

### Flow Diagram

```
User Opens Chat
    ↓
GET /chat/threads → List of previous chats
    ↓
User Selects Thread OR Creates New
    ↓
POST /chat/threads (if new) → Get thread_id
    ↓
Connect WebSocket with thread_id
    ↓
Send Messages → Saved to thread automatically
    ↓
AI Responses → Saved to thread automatically
    ↓
Thread title auto-generated from first message
```

### Auto-Thread Creation

1. **User connects without thread_id**:
   - First message triggers thread creation
   - Thread created with title "New Chat"
   - `thread_id` sent to client via system message

2. **User connects with thread_id**:
   - Messages saved to existing thread
   - Thread title updated if still "New Chat"

3. **Title Auto-Generation**:
   - First user message used to generate title
   - First 50 characters of message
   - User can update title later via API

### Message Saving

**User Messages**:
- Saved immediately when received
- Updates `thread.last_message_at`
- Auto-generates title if needed

**Assistant Messages**:
- Saved after AI response generated
- Includes full metadata (impacted queries, PR data, etc.)
- Updates `thread.last_message_at`

## 📝 Usage Examples

### Example 1: Create New Chat

```bash
# 1. Create new thread
curl -X POST http://localhost:8000/chat/threads \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "My New Chat"}'

# Response: {"id": "thread-uuid", ...}

# 2. Connect WebSocket with thread_id
ws://localhost:8000/chat/ws/{org_id}/{user_id}?token=TOKEN&thread_id=thread-uuid

# 3. Send messages - they're automatically saved
```

### Example 2: Continue Existing Chat

```bash
# 1. Get user's threads
curl -X GET http://localhost:8000/chat/threads \
  -H "Authorization: Bearer YOUR_TOKEN"

# 2. Select a thread and get its messages
curl -X GET http://localhost:8000/chat/threads/{thread_id} \
  -H "Authorization: Bearer YOUR_TOKEN"

# 3. Connect WebSocket with thread_id
ws://localhost:8000/chat/ws/{org_id}/{user_id}?token=TOKEN&thread_id=thread-uuid

# 4. Continue conversation - messages saved automatically
```

### Example 3: Delete Chat

```bash
curl -X DELETE http://localhost:8000/chat/threads/{thread_id} \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🔐 Security

- **User Isolation**: Users can only access their own threads
- **Org Isolation**: Threads are scoped to user's organization
- **Authentication Required**: All endpoints require JWT token
- **Soft Delete**: Threads are soft-deleted (can be restored if needed)

## 🎯 Features

✅ **Create New Chats**: Users can create new conversation threads
✅ **View Chat History**: List all previous conversations
✅ **Continue Conversations**: Load and continue previous chats
✅ **Delete Chats**: Soft delete conversations
✅ **Auto-Title Generation**: Thread titles auto-generated from first message
✅ **Rich Metadata**: All AI response data saved (impacted queries, PR data, etc.)
✅ **WebSocket Integration**: Messages saved automatically during WebSocket chat
✅ **REST Integration**: Messages can be saved via REST endpoint too

## 📊 Data Structure

### Thread Response
```typescript
interface ChatThread {
  id: string;
  org_id: string;
  user_id: string;
  title: string;
  is_active: boolean;
  created_at: string;
  updated_at: string | null;
  last_message_at: string | null;
  message_count: number;
}
```

### Message Response
```typescript
interface ChatMessage {
  id: string;
  thread_id: string;
  role: "user" | "assistant";
  content: string;
  metadata: {
    impacted_query_ids?: string[];
    impacted_queries?: Array<{...}>;
    pr_repo_data?: {...};
    code_suggestions?: {...};
    jira_ticket?: {...};
    processing_time?: number;
    sources?: Array<{...}>;
  } | null;
  created_at: string;
}
```

## 🚀 Frontend Integration

### Typical Flow

1. **On Page Load**:
   ```javascript
   // Get user's threads
   const threads = await fetch('/chat/threads', {
     headers: { 'Authorization': `Bearer ${token}` }
   }).then(r => r.json());
   
   // Display in sidebar
   threads.forEach(thread => {
     displayThread(thread);
   });
   ```

2. **Create New Chat**:
   ```javascript
   const newThread = await fetch('/chat/threads', {
     method: 'POST',
     headers: { 'Authorization': `Bearer ${token}` },
     body: JSON.stringify({ title: 'New Chat' })
   }).then(r => r.json());
   
   // Connect WebSocket with thread_id
   const ws = new WebSocket(
     `ws://localhost:8000/chat/ws/${orgId}/${userId}?token=${token}&thread_id=${newThread.id}`
   );
   ```

3. **Load Existing Chat**:
   ```javascript
   // Get thread with messages
   const thread = await fetch(`/chat/threads/${threadId}`, {
     headers: { 'Authorization': `Bearer ${token}` }
   }).then(r => r.json());
   
   // Display messages
   thread.messages.forEach(msg => {
     displayMessage(msg);
   });
   
   // Connect WebSocket to continue
   const ws = new WebSocket(
     `ws://localhost:8000/chat/ws/${orgId}/${userId}?token=${token}&thread_id=${threadId}`
   );
   ```

4. **Delete Chat**:
   ```javascript
   await fetch(`/chat/threads/${threadId}`, {
     method: 'DELETE',
     headers: { 'Authorization': `Bearer ${token}` }
   });
   ```

## 🔧 Database Migration

The tables will be created automatically when you run the application (via `init_db()`). However, if you need to create them manually:

```sql
-- Run the init_db() function or create tables manually
-- The models are defined in app/utils/models.py
```

## 📝 Notes

- **Thread ID in WebSocket**: Can be provided in connection URL or in message payload
- **Auto-Creation**: If no thread_id provided, new thread created on first message
- **Title Generation**: First 50 characters of first user message
- **Metadata Storage**: All AI response data stored in JSONB for easy querying
- **Soft Delete**: Threads marked as inactive, not physically deleted
- **Ordering**: Threads ordered by `last_message_at` (most recent first)

---

**Implementation Complete!** 🎉

