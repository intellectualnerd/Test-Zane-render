# QueryGuardAI Backend

A FastAPI-based backend for QueryGuardAI - a data lineage and impact analysis tool for Snowflake queries with multi-tenant Snowflake crawler service and real-time WebSocket chat functionality.

## Features

- User authentication with JWT tokens
- **Role-Based Access Control (RBAC)** with four role levels
- Database integration with PostgreSQL
- RESTful API endpoints for auth operations
- Secure password hashing and token management
- Multi-tenant Snowflake crawler service with cron-based scheduling
- Background polling worker for automated query history mining
- Comprehensive audit trail for all crawler operations
- GitHub App integration for repository monitoring
- Jira integration for ticket management
- **Real-time WebSocket chat with AI assistant**
- **Multi-user chat rooms organized by organization**
- **Typing indicators and user status tracking**
- **Automatic session management and cleanup**

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database
- pip

### Installation

1. Clone the repository and navigate to the main directory:
```bash
cd main
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export DATABASE_URL="postgresql+psycopg2://username:password@localhost:5432/queryguard"
export SECRET_KEY="your-secret-key-here"
```

5. Create the database:
```bash
createdb queryguard
```

6. Initialize the first PRODUCT_SUPPORT_ADMIN user:
```bash
python scripts/init_product_support_admin.py
```

7. Run the application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Authentication

- `POST /auth/signup` - Register a new user (creates MEMBER role, requires valid org_id)
- `POST /auth/login` - Login and get JWT token
- `POST /auth/forgot-password` - Generate password reset token
- `POST /auth/reset-password` - Reset password with token
- `POST /auth/logout` - Logout and revoke JWT
- `GET /auth/me` - Get current user information (includes role)

### User Management (Admin Only)

- `POST /users` - Create a new user with role assignment (requires PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, or ORGANIZATION_ADMIN)
- `GET /users` - List users (filtered by organization, optional org_id query parameter)
- `GET /users/{user_id}` - Get user details by ID
- `PUT /users/{user_id}` - Update user (username, email, role, password, is_active)
- `DELETE /users/{user_id}` - Deactivate user (soft delete)

### Organizations (PRODUCT_SUPPORT_ADMIN Only)

- `POST /organizations` - Create a new organization
- `GET /organizations` - List all organizations
- `GET /organizations/{org_id}` - Get organization details
- `PUT /organizations/{org_id}` - Update organization
- `DELETE /organizations/{org_id}` - Deactivate organization

### Snowflake Management

- `POST /snowflake/test-connection` - Test Snowflake connection
- `POST /snowflake/save-connection` - Save Snowflake connection (includes cron expression for automated crawling)
- `GET /snowflake/connections` - List all connections for organization
- `GET /snowflake/fetch-databases/{connection_id}` - Fetch databases from Snowflake
- `GET /snowflake/fetch-schemas/{connection_id}/{database_name}` - Fetch schemas for database
- `POST /snowflake/save-database-selection` - Save database selections
- `POST /snowflake/save-schema-selection` - Save schema selections
- `GET /snowflake/selected-databases/{connection_id}` - Get selected databases
- `GET /snowflake/selected-schemas/{connection_id}/{database_name}` - Get selected schemas

### GitHub App Management

- `GET /github/install?org_id={org_id}` - Redirect to GitHub App installation
- `GET /github/callback` - Handle GitHub App installation callback
- `GET /github/installations` - List all installations for organization
- `GET /github/repositories/{installation_id}` - List repositories for installation
- `POST /github/sync-repositories/{installation_id}` - Sync repositories (manual trigger)
- `DELETE /github/installations/{installation_id}` - Deactivate installation
- `POST /github/webhook` - Handle GitHub webhook events (PR events)
- `POST /github/process-pr` - Process PR changes and add comment

### WebSocket Chat (Real-time)

- `WS /chat/ws/{org_id}/{user_id}` - WebSocket connection for real-time chat
  - Query Parameters: `session_id` (optional), `user_name` (optional)
  - Supports: chat messages, typing indicators, user status, AI responses
- `GET /chat/sessions/{org_id}` - Get active chat sessions for organization
- `GET /chat/stats` - Get WebSocket connection statistics
- `POST /chat/cleanup?timeout_minutes=30` - Manual cleanup of inactive sessions
- `GET /chat/test-page` - Built-in WebSocket test interface
- `POST /chat/query` - Traditional REST API for chat (existing endpoint)

### Health Check & Monitoring

- `GET /` - API information
- `GET /health` - Health check endpoint
- `GET /worker-status` - Check background worker status (includes WebSocket stats)

## Database Schema

### Organizations Table
- `id` (UUID) - Primary key
- `name` (String) - Organization name
- `is_active` (Boolean) - Organization status
- `created_at` (DateTime) - Creation timestamp
- `updated_at` (DateTime) - Last update timestamp

### Users Table
- `id` (UUID) - Primary key
- `username` (String) - Unique username
- `email` (String) - Unique email
- `password_hash` (String) - Hashed password
- `org_id` (UUID) - Foreign key to organizations
- `password_reset_token` (String) - Password reset token
- `password_reset_token_expires` (DateTime) - Token expiry
- `is_active` (Boolean) - User status
- `created_at` (DateTime) - Creation timestamp
- `updated_at` (DateTime) - Last update timestamp

### User Tokens Table
- `id` (UUID) - Primary key
- `user_id` (UUID) - Foreign key to users
- `token` (Text) - JWT token
- `expires_at` (DateTime) - Token expiry
- `is_revoked` (Boolean) - Token status
- `created_at` (DateTime) - Creation timestamp

### Snowflake Connections Table
- `id` (UUID) - Primary key
- `org_id` (UUID) - Foreign key to organizations
- `connection_name` (String) - Connection name
- `account` (String) - Snowflake account
- `username` (String) - Snowflake username
- `password` (String) - Snowflake password
- `warehouse` (String) - Snowflake warehouse
- `cron_expression` (String) - Mining schedule
- `is_active` (Boolean) - Connection status
- `created_at` (DateTime) - Creation timestamp
- `updated_at` (DateTime) - Last update timestamp

### Snowflake Databases Table
- `id` (UUID) - Primary key
- `connection_id` (UUID) - Foreign key to snowflake_connections
- `database_name` (String) - Database name
- `is_selected` (Boolean) - Selection status
- `created_at` (DateTime) - Creation timestamp

### Snowflake Schemas Table
- `id` (UUID) - Primary key
- `database_id` (UUID) - Foreign key to snowflake_databases
- `schema_name` (String) - Schema name
- `is_selected` (Boolean) - Selection status
- `created_at` (DateTime) - Creation timestamp

### Snowflake Jobs Table (Crawler Management)
- `id` (UUID) - Primary key
- `connection_id` (UUID) - Foreign key to snowflake_connections (unique)
- `cron_expression` (String) - Cron expression for scheduling
- `last_run_time` (DateTime) - Watermark for incremental fetching
- `is_active` (Boolean) - Job status
- `created_at` (DateTime) - Creation timestamp
- `updated_at` (DateTime) - Last update timestamp

### Snowflake Crawl Audits Table (Audit Trail)
- `id` (UUID) - Primary key
- `batch_id` (UUID) - Unique identifier for each crawl batch
- `job_id` (UUID) - Foreign key to snowflake_jobs
- `connection_id` (UUID) - Foreign key to snowflake_connections
- `start_time` (DateTime) - Crawl execution start time
- `end_time` (DateTime) - Crawl execution end time
- `status` (String) - RUNNING/COMPLETED/FAILED
- `rows_fetched` (Integer) - Number of records processed
- `error_message` (Text) - Error details if failed
- `created_at` (DateTime) - Creation timestamp

### Snowflake Query Records Table (Query History)
- `id` (UUID) - Primary key
- `batch_id` (UUID) - Foreign key to snowflake_crawl_audits
- `query_id` (String) - Snowflake query identifier
- `query_text` (Text) - Full query text
- `start_time` (DateTime) - Query execution start time
- `end_time` (DateTime) - Query execution end time
- `user_name` (String) - User who executed the query
- `database_name` (String) - Database context
- `schema_name` (String) - Schema context
- `query_type` (String) - Type of query (SELECT, INSERT, etc.)
- `execution_status` (String) - Query execution status
- `created_at` (DateTime) - Creation timestamp

### GitHub Installations Table
- `id` (UUID) - Primary key
- `installation_id` (String) - GitHub installation ID
- `org_id` (UUID) - Foreign key to organizations
- `account_type` (String) - User or Organization
- `account_login` (String) - GitHub username/org name
- `repository_selection` (String) - all or selected
- `permissions` (Text) - JSON string of permissions
- `events` (Text) - JSON string of events
- `is_active` (Boolean) - Installation status
- `created_at` (DateTime) - Creation timestamp
- `updated_at` (DateTime) - Last update timestamp

### GitHub Repositories Table
- `id` (UUID) - Primary key
- `installation_id` (UUID) - Foreign key to github_installations
- `repo_id` (String) - GitHub repository ID
- `repo_name` (String) - Repository name
- `full_name` (String) - Full repository name (owner/repo)
- `private` (Boolean) - Repository visibility
- `description` (Text) - Repository description
- `default_branch` (String) - Default branch name
- `created_at` (DateTime) - Creation timestamp
- `updated_at` (DateTime) - Last update timestamp

## Snowflake Crawler Service

### Overview
The multi-tenant Snowflake crawler service automatically mines query history from Snowflake accounts based on cron expressions. Each client can create multiple Snowflake connections with independent scheduling.

### Key Features
- **Multi-tenant Architecture**: Each client's connections are isolated
- **Cron-based Scheduling**: Flexible scheduling using standard cron expressions
- **Incremental Processing**: Only fetches data since last successful run
- **Background Worker**: Runs independently every 5 minutes
- **Comprehensive Audit Trail**: Tracks all operations with detailed logging
- **Database/Schema Filtering**: Only processes queries from selected databases/schemas

### How It Works

#### 1. Job Creation & Synchronization
When a Snowflake connection is saved with a `cron_expression`, a `SnowflakeJob` record is automatically created. Additionally, on every application startup, the system automatically synchronizes the jobs table with existing connections to ensure no jobs are missed:
```json
{
  "connection_name": "Production DB",
  "account": "your-account.snowflakecomputing.com",
  "username": "your-username",
  "password": "your-password",
  "warehouse": "COMPUTE_WH",
  "role": "ACCOUNTADMIN",
  "cron_expression": "0 */6 * * *"  // Every 6 hours
}
```

#### 2. Startup Synchronization
On every application startup, the system automatically:
- Scans all active connections with cron expressions
- Creates missing job entries for connections without jobs
- Updates existing jobs if cron expressions have changed
- Ensures the jobs table is always in sync with connections

#### 3. Background Worker
- Runs every 10 minutes in a background thread (configurable)
- Checks all active `SnowflakeJob` records
- Uses `croniter` library to evaluate if jobs are due
- Triggers crawler for due jobs
- Enhanced logging with emojis for better visibility

#### 4. Data Crawling Process
For each due job:
1. **Creates Audit Record**: Tracks the crawl operation
2. **Determines Fetch Window**: 
   - First run: Last 30 days
   - Subsequent runs: Since `last_run_time`
3. **Connects to Snowflake**: Using stored connection credentials
4. **Queries History**: Fetches from `account_usage.query_history`
5. **Applies Filters**: Only queries from selected databases/schemas
6. **Stores Results**: Inserts query records into database
7. **Updates Watermark**: Sets new `last_run_time` for next run

#### 5. Cron Expression Examples
- `0 */6 * * *` - Every 6 hours
- `0 0 * * *` - Daily at midnight
- `0 */2 * * *` - Every 2 hours
- `0 9 * * 1` - Weekly on Monday at 9 AM

### Database Schema Relationships
```
SnowflakeConnection (1:1) SnowflakeJob (1:many) SnowflakeCrawlAudit (1:many) SnowflakeQueryRecord
```

### Error Handling
- **Invalid Cron Expressions**: Jobs are deactivated automatically
- **Connection Failures**: Detailed error messages in audit records
- **Partial Failures**: Database rollback ensures consistency
- **Worker Resilience**: Continues processing other jobs if one fails

### Monitoring and Logging
- **Application Level**: Startup/shutdown events with emoji indicators
- **Worker Level**: Job processing and cron evaluation with visual status
- **Crawler Level**: Snowflake connections and data fetching with progress indicators
- **Audit Trail**: Complete operation history with timestamps

#### Enhanced Logging Features
- **🚀 Startup**: Application initialization and worker startup
- **⏰ Job Due**: When jobs are triggered based on cron expressions
- **📊 Data Fetching**: Success with row counts and watermarks
- **ℹ️ No Data**: When no new data is found since last run
- **✅ Success**: Successful operations and completions
- **❌ Errors**: Connection failures and processing errors
- **🔄 Updates**: Job synchronization and worker status changes
- **🛑 Shutdown**: Graceful application termination

### Security Considerations
- **Credential Isolation**: Each client's Snowflake credentials are encrypted and isolated
- **Data Filtering**: Only processes queries from explicitly selected databases/schemas
- **Audit Logging**: All operations are logged for compliance and debugging

## Development

The application uses SQLAlchemy for database operations and JWT for authentication. The database models are defined in `app/utils/models.py` and the authentication logic is in `app/api/auth.py`. The Snowflake crawler service is implemented in `app/snowflake_crawler.py`.

## Crawler Configuration

### Worker Settings
- **Polling Interval**: 10 minutes (600 seconds) - configurable in `app/snowflake_crawler.py`
- **Job Evaluation**: Uses `croniter` library for precise cron expression parsing
- **Error Handling**: Automatic retry and graceful failure handling

### Logging Configuration
The crawler uses enhanced logging with emojis for better visibility:

```
🚀 Starting Snowflake crawler worker (interval: 600 seconds)
⏰ Job due: fd4b938c (cron: */5 * * * *)
📊 Crawl completed: 178 rows fetched, watermark: 2025-09-23 09:15:30
✅ Processed 1 due jobs
```

### Cron Expression Examples
- `0 */6 * * *`: Every 6 hours
- `0 0 * * *`: Daily at midnight
- `0 */2 * * *`: Every 2 hours
- `0 9 * * 1`: Weekly on Monday at 9 AM
- `*/5 * * * *`: Every 5 minutes (for testing)

## WebSocket Real-Time Chat

### Overview
The WebSocket integration provides real-time bidirectional communication for chat functionality with the QueryGuard AI assistant. Users can send messages and receive immediate AI responses with typing indicators and user status updates.

### WebSocket Connection
**URL**: `ws://localhost:8000/chat/ws/{org_id}/{user_id}`

**Query Parameters**:
- `session_id` (optional): Custom session ID, auto-generated if not provided
- `user_name` (optional): Display name for the user

**Example**:
```
ws://localhost:8000/chat/ws/76d33fb3-6062-456b-a211-4aec9971f8be/user123?user_name=John%20Doe
```

### Message Types

#### Outgoing (Client → Server)
```json
// Chat Message
{
  "type": "chat_message",
  "content": "Your question here"
}

// Typing Indicator
{
  "type": "typing", 
  "data": {"is_typing": true}
}

// Ping (Keep-alive)
{
  "type": "ping"
}
```

#### Incoming (Server → Client)
- `system_message` - Connection status, welcome messages
- `chat_message` - User messages broadcast to all org users
- `ai_response` - AI assistant responses with sources
- `typing` - Typing indicators from users or AI
- `user_status` - User join/leave notifications
- `error` - Error messages
- `pong` - Response to ping

### Features
- **Multi-user chat rooms** organized by organization
- **Real-time AI responses** integrated with vector database
- **Typing indicators** for users and AI assistant
- **User status tracking** (join/leave events)
- **Automatic session cleanup** for inactive connections
- **Error handling** with graceful fallbacks
- **Broadcasting** messages to all users in organization

### Testing
1. **Built-in test page**: Navigate to `http://localhost:8000/chat/test-page`
2. **Python client**: Use `websocket_client_example.py`
3. **JavaScript**: See `WEBSOCKET_INTEGRATION.md` for client examples

### Documentation
See `WEBSOCKET_INTEGRATION.md` for comprehensive documentation including:
- Message format specifications
- Client implementation examples
- Architecture overview
- Security considerations
- Troubleshooting guide

## GitHub App Setup

### Required Permissions
- `contents: read` - Read repository contents for PR file changes
- `pull_requests: read` - Read pull request events and data
- `metadata: read` - Read repository metadata

### Required Events
- `pull_request` - Receive events when PRs are opened, closed, reopened, etc.

### Configuration
Update the following values in `app/api/github.py`:
- `GITHUB_APP_URL` - Your GitHub App's installation URL
- `CALLBACK_URL` - Your backend's callback URL for installations
- `WEBHOOK_SECRET` - Your GitHub App's webhook secret for signature verification

### Installation Flow
1. Customer signs into your SaaS (gets org_id)
2. Frontend redirects to: `GET /github/install?org_id={org_id}`
3. Backend redirects to GitHub App installation with state parameter
4. Customer installs app on their repository/organization
5. GitHub redirects back to callback with installation details
6. Backend validates state (org_id) and stores installation data

### Security Notes
- Only installations with valid state parameter (org_id) are processed
- Installations without state are ignored (prevents unauthorized installations)
- All operations are scoped to the user's organization
- Webhook signatures are verified to ensure requests come from GitHub

### Webhook Events
The webhook endpoint processes the following events:
- `pull_request` events with actions: `opened`, `reopened`
- Automatically extracts PR information for downstream processing
- Verifies webhook signature for security

### PR Processing
The PR processing endpoint:
- Validates installation ownership
- Adds comment "Changes Processed By Query Guard AI" to PR
- Requires proper GitHub App access token (needs JWT implementation)

## Role-Based Access Control (RBAC)

QueryGuardAI Backend implements a comprehensive RBAC system with four role levels:

1. **PRODUCT_SUPPORT_ADMIN** - Full product access, cross-organization (QueryGuardAI product owners)
2. **SYSTEM_ADMIN** - Full product access within own organization only (client administrators)
3. **ORGANIZATION_ADMIN** - Organization-level access, can create MEMBER users
4. **MEMBER** - View-only access within own organization

### Key RBAC Features:
- Role-based endpoint access control
- Organization isolation (users can only access their own org unless PRODUCT_SUPPORT_ADMIN)
- Role assignment validation (users can only assign roles they're permitted to assign)
- Connector management restrictions (MEMBER role cannot create/manage connectors)

For detailed RBAC documentation, see [docs/RBAC.md](docs/RBAC.md).

### Initialization

To create the first PRODUCT_SUPPORT_ADMIN user and QueryGuardAI organization:

**Option 1: Run the script**
```bash
python scripts/init_product_support_admin.py
```

**Option 2: Use the API endpoint (Temporary)**
```bash
POST /init-setup
```

⚠️ **WARNING:** The API endpoint should be removed or protected after initial setup!

**Default Credentials:**
- Username: `admin`
- Email: `admin@queryguardai.com`
- Password: `Admin@123`
- Organization: `QueryGuardAI`

⚠️ **IMPORTANT:** Change the default password after first login!

**Note:** The initialization is idempotent - running it multiple times will not create duplicate organizations or users.

## Security Notes

- Passwords are hashed using SHA-256
- JWT tokens are stored in the database for revocation capability
- Password reset tokens expire after 60 minutes
- All sensitive operations require authentication
- Role-based access control enforced on all endpoints
- Organization-level data isolation
