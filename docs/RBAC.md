# Role-Based Access Control (RBAC) Documentation

## Overview

QueryGuardAI Backend implements a comprehensive Role-Based Access Control (RBAC) system to manage user permissions across the platform. The RBAC system ensures that users can only access and perform actions appropriate to their role level.

## Role Hierarchy

The system defines four roles, ordered from highest to lowest privilege:

### 1. PRODUCT_SUPPORT_ADMIN
- **Full product access, cross-organization**
- Special role for QueryGuardAI product owners to manage all clients
- **Can access:**
  - All organization management endpoints (`/organizations/*`)
  - All connector management endpoints (Snowflake, dbt Cloud, GitHub, Jira)
  - All other endpoints across all organizations
- **Can assign roles:** PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER
- **Organization access:** Can access any organization

### 2. SYSTEM_ADMIN
- **Full product access within own organization only**
- Typically assigned to client administrators
- **Can access:**
  - All connector management endpoints (Snowflake, dbt Cloud, GitHub, Jira)
  - All other endpoints within their own organization
  - **Cannot access:** Organization management endpoints (`/organizations/*`)
- **Can assign roles:** SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER (within their org only)
- **Organization access:** Can only access their own organization

### 3. ORGANIZATION_ADMIN
- **Organization-level access within own organization**
- Can manage organization users but with limited role assignment
- **Can access:**
  - All connector management endpoints (Snowflake, dbt Cloud, GitHub, Jira)
  - All other endpoints within their own organization
  - **Cannot access:** Organization management endpoints (`/organizations/*`)
- **Can assign roles:** MEMBER only (cannot create other Organization Admins)
- **Organization access:** Can only access their own organization

### 4. MEMBER
- **View-only access within own organization**
- Basic user role with read-only permissions
- **Can access:**
  - View endpoints (GET requests) for data within their organization
  - Impact analysis endpoints (`/simulate/*`)
  - Chat endpoints (`/chat/*`)
  - **Cannot access:**
    - Organization management endpoints (`/organizations/*`)
    - Connector management endpoints (cannot create/manage Snowflake, dbt, GitHub, Jira connections)
    - User creation endpoints
- **Can assign roles:** None
- **Organization access:** Can only access their own organization

## Role Assignment Rules

### PRODUCT_SUPPORT_ADMIN
- ✅ Can assign: PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER
- ✅ Can assign roles across any organization
- ✅ Can create organizations

### SYSTEM_ADMIN
- ✅ Can assign: SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER
- ❌ Cannot assign: PRODUCT_SUPPORT_ADMIN
- ✅ Can only assign roles within their own organization

### ORGANIZATION_ADMIN
- ✅ Can assign: MEMBER only
- ❌ Cannot assign: PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN
- ✅ Can only assign roles within their own organization

### MEMBER
- ❌ Cannot assign any roles

## API Endpoint Access Matrix

| Endpoint Category | PRODUCT_SUPPORT_ADMIN | SYSTEM_ADMIN | ORGANIZATION_ADMIN | MEMBER |
|------------------|----------------------|--------------|-------------------|--------|
| `/organizations/*` | ✅ Full Access | ❌ No Access | ❌ No Access | ❌ No Access |
| `/users/*` (POST/PUT/DELETE) | ✅ All Roles | ✅ SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER | ✅ MEMBER only | ❌ No Access |
| `/users/*` (GET) | ✅ All Orgs | ✅ Own Org | ✅ Own Org | ❌ No Access |
| `/snowflake/*` (POST/PUT/DELETE) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ❌ No Access |
| `/snowflake/*` (GET) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ✅ Own Org |
| `/dbt-cloud/*` (POST/PUT/DELETE) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ❌ No Access |
| `/dbt-cloud/*` (GET) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ✅ Own Org |
| `/github/*` (POST/DELETE) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ❌ No Access |
| `/github/*` (GET) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ✅ Own Org |
| `/jira/*` (POST/PUT/DELETE) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ❌ No Access |
| `/jira/*` (GET) | ✅ Full Access | ✅ Own Org | ✅ Own Org | ✅ Own Org |
| `/simulate/*` | ✅ Full Access | ✅ Own Org | ✅ Own Org | ✅ Own Org |
| `/chat/*` | ✅ Full Access | ✅ Own Org | ✅ Own Org | ✅ Own Org |

## Implementation Details

### RBAC Utilities Module

The RBAC system is implemented in `app/utils/rbac.py` with the following key functions:

**Note:** Authentication dependencies (`get_current_user`) are located in `app/utils/auth_deps.py` to avoid circular imports.

#### Role Checking Functions
- `has_role(user, role)` - Check if user has a specific role
- `has_any_role(user, roles)` - Check if user has any of the specified roles
- `has_minimum_role(user, minimum_role)` - Check if user has at least the minimum role level
- `can_assign_role(assigner, target_role, target_org_id)` - Check if user can assign a role
- `can_access_organization(user, org_id)` - Check if user can access an organization

#### Dependency Functions (FastAPI)
- `require_role(required_role)` - Require a specific role (factory function, must be called with argument)
- `require_any_role(required_roles)` - Require any of the specified roles (factory function, must be called with argument)
- `require_minimum_role(minimum_role)` - Require at least the minimum role level (factory function, must be called with argument)
- `require_connector_access()` - Require connector management access (excludes MEMBER) - **Must be called with `()`**
- `require_organizations_endpoint_access()` - Require organization endpoint access (PRODUCT_SUPPORT_ADMIN only) - **Must be called with `()`**
- `require_organization_access(org_id_param)` - Require organization access (factory function, optional parameter)

#### Helper Functions
- `check_organization_access(user, org_id)` - Check and raise exception if access denied
- `check_role_assignment(assigner, target_role, target_org_id)` - Check and raise exception if assignment denied

### Usage Examples

#### Protecting an endpoint with role requirement:
```python
from app.utils.rbac import require_role, PRODUCT_SUPPORT_ADMIN

@router.post("/organizations")
def create_organization(
    org: OrganizationCreate,
    current_user: User = Depends(require_role(PRODUCT_SUPPORT_ADMIN)),
    db: Session = Depends(get_db)
):
    # Only PRODUCT_SUPPORT_ADMIN can access this endpoint
    ...
```

#### Protecting connector endpoints:
```python
from app.utils.rbac import require_connector_access

@router.post("/snowflake/save-connection")
def save_connection(
    conn: SnowflakeConn,
    current_user: User = Depends(require_connector_access()),  # Note: () is required
    db: Session = Depends(get_db)
):
    # MEMBER role cannot access this endpoint
    ...
```

#### Protecting organization endpoints:
```python
from app.utils.rbac import require_organizations_endpoint_access

@router.get("/organizations")
def list_organizations(
    current_user: User = Depends(require_organizations_endpoint_access()),  # Note: () is required
    db: Session = Depends(get_db)
):
    # Only PRODUCT_SUPPORT_ADMIN can access this endpoint
    ...
```

#### Using minimum role requirement:
```python
from app.utils.rbac import require_minimum_role, ORGANIZATION_ADMIN

@router.post("/users")
def create_user(
    user: CreateUserRequest,
    current_user: User = Depends(require_minimum_role(ORGANIZATION_ADMIN)),  # Note: argument is required
    db: Session = Depends(get_db)
):
    # ORGANIZATION_ADMIN, SYSTEM_ADMIN, and PRODUCT_SUPPORT_ADMIN can access
    ...
```

#### Checking role assignment:
```python
from app.utils.rbac import check_role_assignment
from app.utils.auth_deps import get_current_user

@router.post("/users")
def create_user(
    user: CreateUserRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if current user can assign the requested role
    check_role_assignment(current_user, user.role, user.org_id)
    ...
```

### Important Notes on Dependency Usage

⚠️ **Critical:** Factory functions that return dependencies must be called correctly:

1. **Functions with parameters** (factory functions):
   - `require_role(role)` - Must be called with role argument
   - `require_any_role([roles])` - Must be called with roles list
   - `require_minimum_role(role)` - Must be called with role argument
   - `require_organization_access(param)` - Optional parameter

2. **Functions without parameters** (must use `()`):
   - `require_connector_access()` - **Must include `()`**
   - `require_organizations_endpoint_access()` - **Must include `()`**

**Incorrect usage (will not enforce RBAC):**
```python
# ❌ WRONG - Missing parentheses
current_user: User = Depends(require_connector_access)

# ❌ WRONG - Missing argument
current_user: User = Depends(require_minimum_role)
```

**Correct usage:**
```python
# ✅ CORRECT
current_user: User = Depends(require_connector_access())

# ✅ CORRECT
current_user: User = Depends(require_minimum_role(ORGANIZATION_ADMIN))
```

## Database Schema

### User Model
The `User` model includes a `role` field:
```python
class User(Base):
    ...
    role = Column(String(50), nullable=False, default="MEMBER", index=True)
    ...
```

Valid role values:
- `PRODUCT_SUPPORT_ADMIN`
- `SYSTEM_ADMIN`
- `ORGANIZATION_ADMIN`
- `MEMBER`

## Initialization

### Creating the First PRODUCT_SUPPORT_ADMIN User

You can initialize the system in two ways:

#### Option 1: Run the Script
```bash
python scripts/init_product_support_admin.py
```

#### Option 2: Use the API Endpoint (Temporary)
```bash
POST /init-setup
```

⚠️ **WARNING:** The API endpoint should be removed or protected after initial setup!

**Hardcoded Credentials:**
- Username: `admin`
- Email: `admin@queryguardai.com`
- Password: `Admin@123`
- Organization: `QueryGuardAI`

⚠️ **IMPORTANT:** Change the default password after first login!

**Note:** The initialization is idempotent - running it multiple times will not create duplicate organizations or users.

### Environment Variables Required
- `DATABASE_URL` - PostgreSQL connection string (required)

## User Creation Flow

### Public Signup (`/auth/signup`)
- Creates users with **MEMBER** role only
- Requires valid `org_id`
- No authentication required

### Admin User Creation (`/users`)
- **Endpoint:** `POST /users` (moved from `/auth/create-user`)
- Requires authentication with minimum role: `ORGANIZATION_ADMIN`
- Validates role assignment permissions
- Validates organization access
- Can create users with appropriate roles based on assigner's role
- Additional endpoints:
  - `GET /users` - List users (requires ORGANIZATION_ADMIN+)
  - `GET /users/{user_id}` - Get user details (requires ORGANIZATION_ADMIN+)
  - `PUT /users/{user_id}` - Update user (requires ORGANIZATION_ADMIN+)
  - `DELETE /users/{user_id}` - Deactivate user (requires ORGANIZATION_ADMIN+)

### Example: Creating a SYSTEM_ADMIN User

**By PRODUCT_SUPPORT_ADMIN:**
```json
POST /users
Authorization: Bearer <token>
Content-Type: application/json

{
  "username": "client_admin",
  "email": "admin@client.com",
  "password": "SecurePassword123",
  "org_id": "<client_org_id>",
  "role": "SYSTEM_ADMIN"
}
```

**By SYSTEM_ADMIN (within their org):**
```json
POST /users
Authorization: Bearer <token>
Content-Type: application/json

{
  "username": "org_admin",
  "email": "orgadmin@client.com",
  "password": "SecurePassword123",
  "org_id": "<same_org_id>",
  "role": "ORGANIZATION_ADMIN"
}
```

## Security Considerations

1. **Role Validation**: All role assignments are validated against the role assignment rules
2. **Organization Isolation**: Users can only access resources within their own organization (unless PRODUCT_SUPPORT_ADMIN)
3. **Token-Based Authentication**: All protected endpoints require valid JWT tokens
4. **Default Role**: New users created via public signup default to MEMBER role
5. **Audit Logging**: All role assignments and access attempts are logged

## Common Use Cases

### Use Case 1: QueryGuardAI Product Owner Setup
1. Run initialization script to create PRODUCT_SUPPORT_ADMIN user
2. Login as PRODUCT_SUPPORT_ADMIN
3. Create new organization for client
4. Create SYSTEM_ADMIN user for client
5. Client uses SYSTEM_ADMIN account to manage their organization

### Use Case 2: Client Organization Management
1. Client logs in as SYSTEM_ADMIN
2. Creates ORGANIZATION_ADMIN users for team leads
3. ORGANIZATION_ADMIN users create MEMBER users for team members
4. All users can view data, but only admins can manage connectors

### Use Case 3: View-Only Access
1. MEMBER users can view all data within their organization
2. MEMBER users can use impact analysis and chat features
3. MEMBER users cannot create or modify connectors
4. MEMBER users cannot create other users

## Troubleshooting

### "Access denied" Errors
- Check user's role: `GET /auth/me` returns user role
- Verify role has required permissions (see Role Hierarchy section)
- Check organization access (users can only access their own org unless PRODUCT_SUPPORT_ADMIN)

### "Cannot assign role" Errors
- Verify assigner's role allows assigning target role (see Role Assignment Rules)
- Check if assignment is within same organization (unless PRODUCT_SUPPORT_ADMIN)
- Ensure target role is valid (one of: PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER)

### Role Not Updating
- Ensure user has permission to assign the role
- Verify organization access
- Check database constraints and logs

## Migration Notes

If upgrading from a version without RBAC:

1. **Database Migration**: Add `role` column to `users` table:
   ```sql
   ALTER TABLE users ADD COLUMN role VARCHAR(50) NOT NULL DEFAULT 'MEMBER';
   CREATE INDEX idx_users_role ON users(role);
   ```

2. **Existing Users**: All existing users will default to MEMBER role. Update manually:
   ```sql
   UPDATE users SET role = 'SYSTEM_ADMIN' WHERE <condition>;
   ```

3. **Run Initialization Script**: Create the first PRODUCT_SUPPORT_ADMIN user:
   ```bash
   python scripts/init_product_support_admin.py
   ```

## API Reference

### Get Current User Info
```http
GET /auth/me
Authorization: Bearer <token>

Response:
{
  "id": "uuid",
  "username": "admin",
  "email": "admin@queryguardai.com",
  "org_id": "uuid",
  "role": "PRODUCT_SUPPORT_ADMIN"
}
```

### Create User (Admin Only)
```http
POST /users
Authorization: Bearer <token>
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "SecurePassword123",
  "org_id": "uuid",
  "role": "MEMBER"
}
```

**Response:**
```json
{
  "id": "uuid",
  "username": "newuser",
  "email": "user@example.com",
  "org_id": "uuid",
  "role": "MEMBER"
}
```

### List Users (Admin Only)
```http
GET /users?org_id=<optional_org_id>
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": "uuid",
    "username": "user1",
    "email": "user1@example.com",
    "org_id": "uuid",
    "role": "MEMBER"
  },
  ...
]
```

### Update User (Admin Only)
```http
PUT /users/{user_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "username": "updated_username",
  "email": "updated@example.com",
  "role": "ORGANIZATION_ADMIN"
}
```

### Deactivate User (Admin Only)
```http
DELETE /users/{user_id}
Authorization: Bearer <token>
```

## Support

For questions or issues related to RBAC:
1. Check this documentation
2. Review logs for access denied messages
3. Verify user roles and organization assignments
4. Contact QueryGuardAI support team

## Architecture Notes

### Module Structure

- **`app/utils/rbac.py`** - RBAC utilities, role checking, and dependency functions
- **`app/utils/auth_deps.py`** - Authentication dependencies (`get_current_user`) to avoid circular imports
- **`app/api/users.py`** - User management endpoints (CRUD operations)
- **`app/api/organizations.py`** - Organization management endpoints (PRODUCT_SUPPORT_ADMIN only)
- **`scripts/init_product_support_admin.py`** - Initialization script and temporary API endpoint

### Circular Import Resolution

The RBAC system uses a separate `auth_deps.py` module to avoid circular imports:
- `rbac.py` imports `get_current_user` from `auth_deps.py`
- `auth.py` imports `MEMBER` from `rbac.py`
- `auth_deps.py` does not import from `rbac.py`, breaking the circular dependency

---

**Last Updated:** 2024
**Version:** 1.1.0

