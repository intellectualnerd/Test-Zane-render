# POST /snowflake/test-connection → test connection
# POST /snowflake/save-connection → save connection (after successful test)
# GET /snowflake/fetch-databases → fetch all databases (legacy)
# GET /snowflake/fetch-schemas/{database} → fetch schemas for selected DB (legacy)
# POST /snowflake/save-schema-selection → save DB + schema selections (legacy)
# GET /snowflake/database-schema-structure/{connection_id} → fetch all databases with schemas in hierarchical format (NEW)
# POST /snowflake/save-database-schema-selections → save database and schema selections from UI structure (NEW)

"""
Detailed Usage Scenarios:
1. Select Specific Schemas Only - User selects only PUBLIC schema
2. Deselect All Schemas - User deselects all schemas in a database
3. Partial Schema Selection - Database has 4 schemas, user selects only 2
4. Database-Only Update - Change database selection without affecting schemas
5. Deselect Everything - User deselects all databases and schemas
6. Invalid State Prevention - Handles invalid attempts gracefully
7. Efficient UI Updates - UI only sends changed items

Key Behaviors Documented:
1. Database selection is ALWAYS determined by schema selections
2. If schemas field is null, existing schema selections are preserved
3. If schema is not mentioned in request, it's set to not selected
4. Invalid states are automatically corrected
5. UI can send minimal payload with only changed items

CONSOLIDATED API USAGE EXAMPLES

=== FETCH ENDPOINT ===
GET /snowflake/database-schema-structure/{connection_id}

Response Format:
[
  {
    "id": "uuid",
    "database_name": "SALES_DB",
    "is_selected": false,
    "created_at": "2024-01-01T00:00:00",
    "schemas": [
      {
        "id": "uuid",
        "schema_name": "PUBLIC",
        "is_selected": false,
        "created_at": "2024-01-01T00:00:00"
      },
      {
        "id": "uuid",
        "schema_name": "STAGING",
        "is_selected": false,
        "created_at": "2024-01-01T00:00:00"
      }
    ]
  }
]

=== SAVE ENDPOINT ===
POST /snowflake/save-database-schema-selections

IMPORTANT RULE: A database can only be selected if at least one schema within it is selected.
If no schemas are selected in a database, the database will automatically be deselected.

=== USAGE SCENARIOS ===

1. SELECT SPECIFIC SCHEMAS ONLY:
   User wants to select only PUBLIC schema from SALES_DB
   
   Request:
   {
     "databases": [
       {
         "database_name": "SALES_DB",
         "is_selected": true,  // Will be ignored - determined by schema selection
         "schemas": [
           {"schema_name": "PUBLIC", "is_selected": true},
           {"schema_name": "STAGING", "is_selected": false}
         ]
       }
     ]
   }
   
   Result: SALES_DB is selected (because PUBLIC schema is selected)

2. DESELECT ALL SCHEMAS IN A DATABASE:
   User wants to deselect all schemas in SALES_DB
   
   Request:
   {
     "databases": [
       {
         "database_name": "SALES_DB",
         "is_selected": false,  // Will be ignored - determined by schema selection
         "schemas": [
           {"schema_name": "PUBLIC", "is_selected": false},
           {"schema_name": "STAGING", "is_selected": false}
         ]
       }
     ]
   }
   
   Result: SALES_DB is deselected (because no schemas are selected)

3. PARTIAL SCHEMA SELECTION:
   Database has 4 schemas, user wants to select only 2
   
   Request:
   {
     "databases": [
       {
         "database_name": "ANALYTICS_DB",
         "is_selected": true,  // Will be ignored - determined by schema selection
         "schemas": [
           {"schema_name": "RAW", "is_selected": true},
           {"schema_name": "CLEANED", "is_selected": true}
           // SCHEMA3 and SCHEMA4 not mentioned = automatically deselected
         ]
       }
     ]
   }
   
   Result: ANALYTICS_DB is selected (because RAW and CLEANED schemas are selected)

4. DATABASE-ONLY UPDATE (NO SCHEMA CHANGES):
   User wants to change database selection without affecting schemas
   
   Request:
   {
     "databases": [
       {
         "database_name": "MARKETING_DB",
         "is_selected": true,  // Will be validated against existing schema selections
         "schemas": null  // No schema changes
       }
     ]
   }
   
   Result: MARKETING_DB selection depends on existing schema selections

5. DESELECT EVERYTHING:
   User wants to deselect all databases and schemas
   
   Request:
   {
     "databases": [
       {
         "database_name": "SALES_DB",
         "is_selected": false,  // Will be ignored - determined by schema selection
         "schemas": [
           {"schema_name": "PUBLIC", "is_selected": false},
           {"schema_name": "STAGING", "is_selected": false}
         ]
       },
       {
         "database_name": "ANALYTICS_DB",
         "is_selected": false,  // Will be ignored - determined by schema selection
         "schemas": [
           {"schema_name": "RAW", "is_selected": false},
           {"schema_name": "CLEANED", "is_selected": false}
         ]
       }
     ]
   }
   
   Result: All databases and schemas are deselected

6. INVALID STATE PREVENTION:
   User tries to select database without selecting any schemas
   
   Request:
   {
     "databases": [
       {
         "database_name": "SALES_DB",
         "is_selected": true,  // Invalid - no schemas selected
         "schemas": [
           {"schema_name": "PUBLIC", "is_selected": false},
           {"schema_name": "STAGING", "is_selected": false}
         ]
       }
     ]
   }
   
   Result: SALES_DB is automatically deselected (warning logged)

7. EFFICIENT UI UPDATES:
   UI only sends changed items to minimize payload
   
   Request (only changed schemas):
   {
     "databases": [
       {
         "database_name": "SALES_DB",
         "is_selected": true,  // Will be ignored - determined by schema selection
         "schemas": [
           {"schema_name": "PUBLIC", "is_selected": true}  // Only changed schema
           // Other schemas not mentioned = maintain current state
         ]
       }
     ]
   }
   
   Result: Only PUBLIC schema selection is updated, database selection follows

=== KEY BEHAVIORS ===
- Database selection is ALWAYS determined by schema selections
- If schemas field is null, existing schema selections are preserved
- If schema is not mentioned in request, it's set to not selected
- Invalid states (database selected without schemas) are automatically corrected
- UI can send minimal payload with only changed items
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from pydantic import BaseModel, validator
import snowflake.connector
from app.database import get_db
from app.utils.models import (
    ColumnLevelLineage,
    # FilterClauseColumnLineage,
    InformationSchemacolumns,
    LineageLoadWatermark,
    SnowflakeConnection,
    SnowflakeCrawlAudit,
    SnowflakeDatabase,
    SnowflakeJob,
    SnowflakeQueryRecord,
    SnowflakeSchema,
    User,
)
from app.utils.auth_deps import get_current_user
from app.utils.rbac import require_connector_access, check_organization_access
import uuid
from uuid import UUID
from datetime import datetime
import logging
from croniter import croniter
from fastapi_versioning import version

router = APIRouter(prefix="/snowflake", tags=["Snowflake"])
logger = logging.getLogger("snowflake")


# --- Helpers ---
def validate_cron_expression(cron_expr: Optional[str]) -> bool:
    """Validate cron expression using croniter library"""
    if not cron_expr or cron_expr.strip() == "":
        return True  # Empty cron expression is valid (no scheduling)
    
    try:
        # Test if the cron expression is valid by trying to get next run time
        croniter(cron_expr.strip())
        return True
    except Exception as e:
        logger.warning("Invalid cron expression '%s': %s", cron_expr, str(e))
        return False

def test_connection(account, username, password, warehouse=None, database=None, schema=None, role=None):
    try:
        conn = snowflake.connector.connect(
            user=username,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema,
            role=role
        )
        
        # Test that the role and warehouse are valid by executing queries
        # This will fail if the role/warehouse is invalid or doesn't have necessary permissions
        cur = conn.cursor()
        try:
            # Check current warehouse to verify it's set correctly
            if warehouse and warehouse.strip():
                cur.execute("SELECT CURRENT_WAREHOUSE()")
                current_warehouse = cur.fetchone()[0]
                
                if current_warehouse is None:
                    cur.close()
                    conn.close()
                    logger.warning("/snowflake/test-connection - warehouse not set: expected=%s", warehouse)
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Warehouse validation failed: Warehouse '{warehouse}' was not set. Please check if the warehouse name is correct and the user has permission to use it."
                    )
                
                if current_warehouse.upper() != warehouse.upper():
                    cur.close()
                    conn.close()
                    logger.warning("/snowflake/test-connection - warehouse mismatch: expected=%s, got=%s", warehouse, current_warehouse)
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Warehouse validation failed: Expected warehouse '{warehouse}' but got '{current_warehouse}'. Please check if the warehouse name is correct and the user has permission to use it."
                    )
            
            # Check current role to verify it's set correctly
            cur.execute("SELECT CURRENT_ROLE()")
            current_role = cur.fetchone()[0]
            
            # If a role was specified, verify it matches what we got back
            if role and role.strip():
                if current_role.upper() != role.upper():
                    cur.close()
                    conn.close()
                    logger.warning("/snowflake/test-connection - role mismatch: expected=%s, got=%s", role, current_role)
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Role validation failed: Expected role '{role}' but got '{current_role}'. Please check if the role name is correct and the user has permission to use it."
                    )
            
            # Test basic query execution to ensure role and warehouse have necessary permissions
            cur.execute("SHOW DATABASES")
            cur.fetchall()  # Consume the result
            
            logger.info("/snowflake/test-connection - success for user=%s account=%s warehouse=%s role=%s", 
                       username, account, warehouse or "default", role or "default")
        finally:
            cur.close()
        
        conn.close()
        return True
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_msg = str(e)
        error_lower = error_msg.lower()
        
        # Check if it's a warehouse-related error
        if "warehouse" in error_lower or ("does not exist" in error_lower and warehouse):
            logger.warning("/snowflake/test-connection - warehouse validation failed: %s", error_msg)
            raise HTTPException(
                status_code=400, 
                detail=f"Warehouse validation failed: {error_msg}. Please verify that the warehouse '{warehouse}' exists and the user has permission to use it."
            )
        
        # Check if it's a role-related error
        if "role" in error_lower or ("does not exist" in error_lower and role) or "not authorized" in error_lower:
            logger.warning("/snowflake/test-connection - role validation failed: %s", error_msg)
            raise HTTPException(
                status_code=400, 
                detail=f"Role validation failed: {error_msg}. Please verify that the role '{role}' exists and the user has permission to use it."
            )
        
        logger.warning("/snowflake/test-connection - connection failed: %s", error_msg)
        raise HTTPException(status_code=400, detail=f"Connection failed: {error_msg}")


# --- Models ---
class SnowflakeConn(BaseModel):
    connection_name: str
    account: str
    username: str
    password: str
    warehouse: str = None
    role: str = None
    cron_expression: Optional[str] = None
    
    @validator('cron_expression')
    def validate_cron_expression(cls, v):
        """Validate cron expression if provided"""
        if v is not None and v.strip() != "":
            if not validate_cron_expression(v):
                raise ValueError(f"Invalid cron expression: '{v}'. Please use standard cron format (e.g., '0 */6 * * *' for every 6 hours)")
        return v.strip() if v else None

class DatabaseSelection(BaseModel):
    database_names: List[str]

class SchemaSelection(BaseModel):
    database_name: str
    schema_names: List[str]

class ConnectionResponse(BaseModel):
    id: UUID
    connection_name: str
    account: str
    username: str
    warehouse: str | None
    cron_expression: str | None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class DatabaseResponse(BaseModel):
    id: UUID
    database_name: str
    is_selected: bool
    created_at: datetime

    class Config:
        from_attributes = True

class SchemaResponse(BaseModel):
    id: UUID
    schema_name: str
    is_selected: bool
    created_at: datetime

    class Config:
        from_attributes = True

class SchemaNode(BaseModel):
    id: UUID
    schema_name: str
    is_selected: bool
    created_at: datetime

    class Config:
        from_attributes = True

class DatabaseNode(BaseModel):
    id: UUID
    database_name: str
    is_selected: bool
    created_at: datetime
    schemas: List[SchemaNode] = []

    class Config:
        from_attributes = True

class SchemaSelectionItem(BaseModel):
    schema_name: str
    is_selected: bool

class DatabaseSelectionItem(BaseModel):
    database_name: str
    is_selected: bool
    schemas: Optional[List[SchemaSelectionItem]] = None  # Only include if database is selected

class DatabaseSchemaSelectionRequest(BaseModel):
    databases: List[DatabaseSelectionItem]


# --- Endpoints ---
@router.post("/test-connection")
@version(1, 0)
def snowflake_test_connection(conn: SnowflakeConn, current_user: User = Depends(require_connector_access()), request: Request = None):
    """Test Snowflake connection before saving, including role validation"""
    success = test_connection(
        account=conn.account,
        username=conn.username,
        password=conn.password,
        warehouse=conn.warehouse,
        role=conn.role
    )
    return {"message": "Connection successful"} if success else {"message": "Connection failed"}


@router.post("/save-connection", response_model=ConnectionResponse, status_code=status.HTTP_201_CREATED)
@version(1, 0)
def save_connection(conn: SnowflakeConn, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db), request: Request = None):
    """Save Snowflake connection after successful test"""
    # Test connection before saving, including role validation
    test_connection(conn.account, conn.username, conn.password, conn.warehouse, role=conn.role)

    # Validate cron expression if provided
    if conn.cron_expression and conn.cron_expression.strip():
        if not validate_cron_expression(conn.cron_expression):
            logger.warning("/snowflake/save-connection - invalid cron expression: %s", conn.cron_expression)
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid cron expression: '{conn.cron_expression}'. Please use standard cron format (e.g., '0 */6 * * *' for every 6 hours)"
            )

    # Check if connection name already exists for this org
    existing_conn = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.connection_name == conn.connection_name,
        SnowflakeConnection.is_active == True
    ).first()
    
    if existing_conn:
        logger.warning("/snowflake/save-connection - name exists %s", conn.connection_name)
        raise HTTPException(status_code=400, detail="Connection name already exists for this organization")

    new_connection = SnowflakeConnection(
        org_id=current_user.org_id,
        connection_name=conn.connection_name,
        account=conn.account,
        username=conn.username,
        password=conn.password,
        warehouse=conn.warehouse,
        role=conn.role,
        cron_expression=conn.cron_expression
    )
    
    try:
        db.add(new_connection)
        db.commit()
        db.refresh(new_connection)
        
        # Handle job creation/update based on cron expression
        if conn.cron_expression and conn.cron_expression.strip():
            # Create or update job for scheduled crawling
            existing_job = db.query(SnowflakeJob).filter(SnowflakeJob.connection_id == new_connection.id).first()
            if existing_job:
                existing_job.cron_expression = conn.cron_expression
                existing_job.is_active = True
            else:
                db.add(SnowflakeJob(
                    connection_id=new_connection.id,
                    cron_expression=conn.cron_expression,
                    last_run_time=None,
                    is_active=True
                ))
            db.commit()
            logger.info("/snowflake/save-connection - created/updated job with cron: %s", conn.cron_expression)
        else:
            # Deactivate any existing job if cron expression is empty
            existing_job = db.query(SnowflakeJob).filter(SnowflakeJob.connection_id == new_connection.id).first()
            if existing_job:
                existing_job.is_active = False
                db.commit()
                logger.info("/snowflake/save-connection - deactivated job (no cron expression)")
        
        logger.info("/snowflake/save-connection - saved id=%s", new_connection.id)
        return new_connection
    except IntegrityError:
        db.rollback()
        logger.exception("/snowflake/save-connection - failed to save connection")
        raise HTTPException(status_code=400, detail="Failed to save connection")


@router.get("/connections", response_model=List[ConnectionResponse])
@version(1, 0)
def list_connections(current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    """List all Snowflake connections for the organization"""
    connections = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).all()
    logger.debug("/snowflake/connections - list count=%d", len(connections))
    return connections


@router.delete("/connections/{connection_id}")
@version(1, 0)
def deactivate_connection(
    connection_id: str,
    current_user: User = Depends(require_connector_access()),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Deactivate a Snowflake connection (soft delete)"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/connections - delete: invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/connections - delete: connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    connection.is_active = False
    
    # Also deactivate the associated job if it exists
    job = db.query(SnowflakeJob).filter(SnowflakeJob.connection_id == conn_uuid).first()
    if job:
        job.is_active = False
        logger.debug("/snowflake/connections - delete: deactivated job for connection %s", conn_uuid)
    
    try:
        db.commit()
        logger.info("/snowflake/connections - delete: deactivated id=%s", connection.id)
        return {"message": "Snowflake connection deactivated successfully"}
    except IntegrityError:
        db.rollback()
        logger.exception("/snowflake/connections - delete: failed to deactivate")
        raise HTTPException(status_code=400, detail="Failed to deactivate connection")


@router.delete(
    "/dev/connections/{connection_id}/purge",
    tags=["Snowflake (Dev)"],
    summary="DEV ONLY: Hard delete Snowflake connection and all related data",
)
@version(1, 0)
def dev_purge_connection_data(
    connection_id: str,
    current_user: User = Depends(require_connector_access()),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """
    Dev-only: permanently delete a Snowflake connection and all associated data.
    """
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/dev/purge - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    connection = db.query(SnowflakeConnection).filter(SnowflakeConnection.id == conn_uuid).first()
    if not connection:
        logger.warning("/snowflake/dev/purge - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    # Ensure caller can access the org that owns this connection (PRODUCT_SUPPORT_ADMIN can cross-org)
    check_organization_access(current_user, connection.org_id)

    try:
        deleted_counts = {}

        # Delete lineage artifacts
        deleted_counts["column_level_lineage"] = (
            db.query(ColumnLevelLineage)
            .filter(ColumnLevelLineage.connection_id == conn_uuid)
            .delete(synchronize_session=False)
        )
        # deleted_counts["filter_clause_lineage"] = (
        #     db.query(FilterClauseColumnLineage)
        #     .filter(FilterClauseColumnLineage.connection_id == conn_uuid)
        #     .delete(synchronize_session=False)
        # )
        deleted_counts["lineage_watermarks"] = (
            db.query(LineageLoadWatermark)
            .filter(LineageLoadWatermark.connection_id == conn_uuid)
            .delete(synchronize_session=False)
        )

        # Delete crawl history and query metadata
        deleted_counts["crawl_audit"] = (
            db.query(SnowflakeCrawlAudit)
            .filter(SnowflakeCrawlAudit.connection_id == conn_uuid)
            .delete(synchronize_session=False)
        )
        deleted_counts["query_history"] = (
            db.query(SnowflakeQueryRecord)
            .filter(SnowflakeQueryRecord.connection_id == conn_uuid)
            .delete(synchronize_session=False)
        )
        deleted_counts["information_schema_columns"] = (
            db.query(InformationSchemacolumns)
            .filter(InformationSchemacolumns.connection_id == conn_uuid)
            .delete(synchronize_session=False)
        )

        # Delete schemas and databases tied to this connection
        db_ids_subquery = (
            db.query(SnowflakeDatabase.id)
            .filter(SnowflakeDatabase.connection_id == conn_uuid)
            .subquery()
        )
        deleted_counts["schemas"] = (
            db.query(SnowflakeSchema)
            .filter(SnowflakeSchema.database_id.in_(db_ids_subquery))
            .delete(synchronize_session=False)
        )
        deleted_counts["databases"] = (
            db.query(SnowflakeDatabase)
            .filter(SnowflakeDatabase.connection_id == conn_uuid)
            .delete(synchronize_session=False)
        )

        # Delete scheduled job (if any)
        deleted_counts["jobs"] = (
            db.query(SnowflakeJob)
            .filter(SnowflakeJob.connection_id == conn_uuid)
            .delete(synchronize_session=False)
        )

        # Finally delete the connection itself (hard delete)
        deleted_counts["connections"] = (
            db.query(SnowflakeConnection)
            .filter(SnowflakeConnection.id == conn_uuid)
            .delete(synchronize_session=False)
        )

        db.commit()
        logger.warning(
            "/snowflake/dev/purge - hard-deleted connection %s with counts %s",
            conn_uuid,
            deleted_counts,
        )
        return {
            "message": "Snowflake connection and all related data permanently deleted (dev-only).",
            "connection_id": str(conn_uuid),
            "deleted_counts": deleted_counts,
        }
    except Exception as exc:
        db.rollback()
        logger.exception("/snowflake/dev/purge - failed to delete data for %s", conn_uuid)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purge Snowflake connection data: {str(exc)}",
        )


@router.get("/fetch-databases/{connection_id}")
@version(1, 0)
def fetch_databases(connection_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    """Fetch all databases from Snowflake connection"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/fetch-databases - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Get connection details
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/fetch-databases - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    try:
        # Connect to Snowflake and fetch databases
        snowflake_conn = snowflake.connector.connect(
            user=connection.username,
            password=connection.password,
            account=connection.account,
            warehouse=connection.warehouse,
            role=connection.role
        )
        cur = snowflake_conn.cursor()
        cur.execute("SHOW DATABASES")
        databases = [row[1] for row in cur.fetchall()]
        cur.close()
        snowflake_conn.close()
        
        # Store databases in our database
        for db_name in databases:
            existing_db = db.query(SnowflakeDatabase).filter(
                SnowflakeDatabase.connection_id == conn_uuid,
                SnowflakeDatabase.database_name == db_name
            ).first()
            
            if not existing_db:
                new_db = SnowflakeDatabase(
                    connection_id=conn_uuid,
                    database_name=db_name
                )
                db.add(new_db)
        
        db.commit()
        logger.info("/snowflake/fetch-databases - fetched %d databases", len(databases))
        return {"databases": databases}
        
    except Exception as e:
        logger.exception("/snowflake/fetch-databases - error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/fetch-schemas/{connection_id}/{database_name}")
@version(1, 0)
def fetch_schemas(connection_id: str, database_name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    """Fetch schemas for a specific database"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/fetch-schemas - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Get connection details
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/fetch-schemas - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    try:
        # Connect to Snowflake and fetch schemas
        snowflake_conn = snowflake.connector.connect(
            user=connection.username,
            password=connection.password,
            account=connection.account,
            warehouse=connection.warehouse,
            database=database_name,
            role=connection.role
        )
        cur = snowflake_conn.cursor()
        cur.execute("SHOW SCHEMAS")
        schemas = [row[1] for row in cur.fetchall()]
        cur.close()
        snowflake_conn.close()
        
        # Get or create database record
        database = db.query(SnowflakeDatabase).filter(
            SnowflakeDatabase.connection_id == conn_uuid,
            SnowflakeDatabase.database_name == database_name
        ).first()
        
        if not database:
            database = SnowflakeDatabase(
                connection_id=conn_uuid,
                database_name=database_name
            )
            db.add(database)
            db.flush()  # Get the ID
        
        # Store schemas in our database
        for schema_name in schemas:
            existing_schema = db.query(SnowflakeSchema).filter(
                SnowflakeSchema.database_id == database.id,
                SnowflakeSchema.schema_name == schema_name
            ).first()
            
            if not existing_schema:
                new_schema = SnowflakeSchema(
                    database_id=database.id,
                    schema_name=schema_name
                )
                db.add(new_schema)
        
        db.commit()
        logger.info("/snowflake/fetch-schemas - fetched %d schemas for %s", len(schemas), database_name)
        return {"schemas": schemas}
        
    except Exception as e:
        logger.exception("/snowflake/fetch-schemas - error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/save-database-selection")
@version(1, 0)
def save_database_selection(selection: DatabaseSelection, connection_id: str, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db), request: Request = None):
    """Save database selections for a connection"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/save-database-selection - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Verify connection belongs to user's org
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/save-database-selection - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    # Update database selections
    database_rows = db.query(SnowflakeDatabase).filter(
        SnowflakeDatabase.connection_id == conn_uuid
    ).all()
    
    for database_row in database_rows:
        database_row.is_selected = database_row.database_name in selection.database_names
    
    db.commit()
    logger.info("/snowflake/save-database-selection - saved selections count=%d", len(selection.database_names))
    return {"message": "Database selections saved"}


@router.post("/save-schema-selection")
@version(1, 0)
def save_schema_selection(selection: SchemaSelection, connection_id: str, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db), request: Request = None):
    """Save schema selections for a specific database"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/save-schema-selection - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Verify connection belongs to user's org
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/save-schema-selection - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    # Get database
    database = db.query(SnowflakeDatabase).filter(
        SnowflakeDatabase.connection_id == conn_uuid,
        SnowflakeDatabase.database_name == selection.database_name
    ).first()
    
    if not database:
        logger.warning("/snowflake/save-schema-selection - database not found %s", selection.database_name)
        raise HTTPException(status_code=404, detail="Database not found")

    # Update schema selections
    schemas = db.query(SnowflakeSchema).filter(
        SnowflakeSchema.database_id == database.id
    ).all()
    
    for schema in schemas:
        schema.is_selected = schema.schema_name in selection.schema_names
    
    db.commit()
    logger.info("/snowflake/save-schema-selection - saved %d schemas for %s", len(selection.schema_names), selection.database_name)
    return {"message": "Schema selections saved"}


@router.get("/selected-databases/{connection_id}", response_model=List[DatabaseResponse])
@version(1, 0)
def get_selected_databases(connection_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    """Get selected databases for a connection"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/selected-databases - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Verify connection belongs to user's org
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/selected-databases - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    selected_databases = db.query(SnowflakeDatabase).filter(
        SnowflakeDatabase.connection_id == conn_uuid,
        SnowflakeDatabase.is_selected == True
    ).all()
    
    logger.debug("/snowflake/selected-databases - count=%d", len(selected_databases))
    return selected_databases


@router.get("/selected-schemas/{connection_id}/{database_name}", response_model=List[SchemaResponse])
@version(1, 0)
def get_selected_schemas(connection_id: str, database_name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    """Get selected schemas for a specific database"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/selected-schemas - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Verify connection belongs to user's org
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/selected-schemas - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    # Get database
    database = db.query(SnowflakeDatabase).filter(
        SnowflakeDatabase.connection_id == conn_uuid,
        SnowflakeDatabase.database_name == database_name
    ).first()
    
    if not database:
        logger.warning("/snowflake/selected-schemas - database not found %s", database_name)
        raise HTTPException(status_code=404, detail="Database not found")

    selected_schemas = db.query(SnowflakeSchema).filter(
        SnowflakeSchema.database_id == database.id,
        SnowflakeSchema.is_selected == True
    ).all()
    
    logger.debug("/snowflake/selected-schemas - count=%d", len(selected_schemas))
    return selected_schemas


@router.get("/database-schema-structure/{connection_id}", response_model=List[DatabaseNode])
@version(1, 0)
def fetch_database_schema_structure(connection_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), request: Request = None):
    """Fetch all databases with their schemas in hierarchical format"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/database-schema-structure - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Get connection details
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/database-schema-structure - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    try:
        # Connect to Snowflake and fetch databases
        snowflake_conn = snowflake.connector.connect(
            user=connection.username,
            password=connection.password,
            account=connection.account,
            warehouse=connection.warehouse,
            role=connection.role
        )
        cur = snowflake_conn.cursor()
        cur.execute("SHOW DATABASES")
        databases = [row[1] for row in cur.fetchall()]
        cur.close()
        snowflake_conn.close()
        
        # Build tree structure
        database_nodes = []
        
        for db_name in databases:
            # Get or create database record
            database = db.query(SnowflakeDatabase).filter(
                SnowflakeDatabase.connection_id == conn_uuid,
                SnowflakeDatabase.database_name == db_name
            ).first()
            
            if not database:
                database = SnowflakeDatabase(
                    connection_id=conn_uuid,
                    database_name=db_name
                )
                db.add(database)
                db.flush()  # Get the ID
            
            # Fetch schemas for this database
            try:
                snowflake_conn = snowflake.connector.connect(
                    user=connection.username,
                    password=connection.password,
                    account=connection.account,
                    warehouse=connection.warehouse,
                    database=db_name,
                    role=connection.role
                )
                cur = snowflake_conn.cursor()
                cur.execute("SHOW SCHEMAS")
                schemas = [row[1] for row in cur.fetchall()]
                cur.close()
                snowflake_conn.close()
                
                # Store schemas in our database and build schema nodes
                schema_nodes = []
                for schema_name in schemas:
                    existing_schema = db.query(SnowflakeSchema).filter(
                        SnowflakeSchema.database_id == database.id,
                        SnowflakeSchema.schema_name == schema_name
                    ).first()
                    
                    if not existing_schema:
                        new_schema = SnowflakeSchema(
                            database_id=database.id,
                            schema_name=schema_name
                        )
                        db.add(new_schema)
                        db.flush()  # Get the ID
                        schema_nodes.append(SchemaNode(
                            id=new_schema.id,
                            schema_name=new_schema.schema_name,
                            is_selected=new_schema.is_selected,
                            created_at=new_schema.created_at
                        ))
                    else:
                        schema_nodes.append(SchemaNode(
                            id=existing_schema.id,
                            schema_name=existing_schema.schema_name,
                            is_selected=existing_schema.is_selected,
                            created_at=existing_schema.created_at
                        ))
                
                # Create database node with schemas
                database_nodes.append(DatabaseNode(
                    id=database.id,
                    database_name=database.database_name,
                    is_selected=database.is_selected,
                    created_at=database.created_at,
                    schemas=schema_nodes
                ))
                
            except Exception as e:
                logger.warning("/snowflake/database-schema-structure - error fetching schemas for %s: %s", db_name, str(e))
                # Still add database node but with empty schemas
                database_nodes.append(DatabaseNode(
                    id=database.id,
                    database_name=database.database_name,
                    is_selected=database.is_selected,
                    created_at=database.created_at,
                    schemas=[]
                ))
        
        db.commit()
        logger.info("/snowflake/database-schema-structure - fetched %d databases with schemas", len(database_nodes))
        return database_nodes
        
    except Exception as e:
        logger.exception("/snowflake/database-schema-structure - error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/save-database-schema-selections")
@version(1, 0)
def save_database_schema_selections(selection: DatabaseSchemaSelectionRequest, connection_id: str, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db), request: Request = None):
    """Save database and schema selections from UI structure with flexible selection options"""
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/save-database-schema-selections - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    # Verify connection belongs to user's org
    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()
    
    if not connection:
        logger.warning("/snowflake/save-database-schema-selections - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    try:
        # Process each database selection
        for db_selection in selection.databases:
            # Get or create database record
            database = db.query(SnowflakeDatabase).filter(
                SnowflakeDatabase.connection_id == conn_uuid,
                SnowflakeDatabase.database_name == db_selection.database_name
            ).first()
            
            if not database:
                logger.warning("/snowflake/save-database-schema-selections - database not found %s", db_selection.database_name)
                continue  # Skip if database doesn't exist
            
            # Handle schema selections - only process if schemas are provided
            if db_selection.schemas is not None:
                # Get all schemas for this database
                schemas = db.query(SnowflakeSchema).filter(
                    SnowflakeSchema.database_id == database.id
                ).all()
                
                # Create a map of schema names to selection status for efficient lookup
                schema_selection_map = {schema_item.schema_name: schema_item.is_selected 
                                      for schema_item in db_selection.schemas}
                
                # Update schema selections based on the provided data
                any_schema_selected = False
                for schema in schemas:
                    if schema.schema_name in schema_selection_map:
                        schema.is_selected = schema_selection_map[schema.schema_name]
                        if schema.is_selected:
                            any_schema_selected = True
                    else:
                        # If schema is not mentioned in the request, set to not selected
                        schema.is_selected = False
                
                # Database is only selected if at least one schema is selected
                database.is_selected = any_schema_selected
            else:
                # If no schemas provided in request, don't change schema selections
                # But ensure database selection follows the rule: database selected only if schemas are selected
                schemas = db.query(SnowflakeSchema).filter(
                    SnowflakeSchema.database_id == database.id
                ).all()
                
                # Check if any existing schema is selected
                any_schema_selected = any(schema.is_selected for schema in schemas)
                
                # Database selection must follow schema selection rule
                if db_selection.is_selected and not any_schema_selected:
                    # User tried to select database but no schemas are selected - invalid state
                    logger.warning("/snowflake/save-database-schema-selections - cannot select database %s without selecting schemas", db_selection.database_name)
                    database.is_selected = False
                else:
                    database.is_selected = any_schema_selected
        
        db.commit()
        logger.info("/snowflake/save-database-schema-selections - saved selections for %d databases", len(selection.databases))
        return {"message": "Database and schema selections saved successfully"}
        
    except Exception as e:
        db.rollback()
        logger.exception("/snowflake/save-database-schema-selections - error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


# --- Demo/Automation Endpoint ---
@router.post("/demo-auto-setup/{connection_id}")
@version(1, 0)
def demo_auto_setup(connection_id: str, current_user: User = Depends(require_connector_access()), db: Session = Depends(get_db), request: Request = None):
    """Demo helper: given a connection_id, automatically
    - fetches and stores all databases
    - fetches and stores schemas for each database
    - marks all databases and schemas as selected

    This is intended to speed up demos by skipping manual multi-step setup.
    """
    # Validate connection id and ownership
    try:
        conn_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.warning("/snowflake/demo-auto-setup - invalid id %s", connection_id)
        raise HTTPException(status_code=400, detail="Invalid connection ID format")

    connection = db.query(SnowflakeConnection).filter(
        SnowflakeConnection.id == conn_uuid,
        SnowflakeConnection.org_id == current_user.org_id,
        SnowflakeConnection.is_active == True
    ).first()

    if not connection:
        logger.warning("/snowflake/demo-auto-setup - connection not found %s", conn_uuid)
        raise HTTPException(status_code=404, detail="Snowflake connection not found")

    # Step 1: Fetch and persist databases (reuse logic inline to avoid HTTP hop)
    try:
        sf_conn = snowflake.connector.connect(
            user=connection.username,
            password=connection.password,
            account=connection.account,
            warehouse=connection.warehouse,
            role=connection.role
        )
        cur = sf_conn.cursor()
        cur.execute("SHOW DATABASES")
        database_names = [row[1] for row in cur.fetchall()]
        cur.close()
        sf_conn.close()
    except Exception as e:
        logger.exception("/snowflake/demo-auto-setup - fetch databases error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

    # Persist databases
    for db_name in database_names:
        existing_db = db.query(SnowflakeDatabase).filter(
            SnowflakeDatabase.connection_id == conn_uuid,
            SnowflakeDatabase.database_name == db_name
        ).first()
        if not existing_db:
            existing_db = SnowflakeDatabase(
                connection_id=conn_uuid,
                database_name=db_name,
                is_selected=True
            )
            db.add(existing_db)
        else:
            existing_db.is_selected = True

    db.commit()

    # Step 2: For each database, fetch schemas and persist them, mark selected
    databases_rows = db.query(SnowflakeDatabase).filter(
        SnowflakeDatabase.connection_id == conn_uuid
    ).all()

    for database_row in databases_rows:
        try:
            sf_conn = snowflake.connector.connect(
                user=connection.username,
                password=connection.password,
                account=connection.account,
                warehouse=connection.warehouse,
                database=database_row.database_name,
                role=connection.role
            )
            cur = sf_conn.cursor()
            cur.execute("SHOW SCHEMAS")
            schema_names = [row[1] for row in cur.fetchall()]
            cur.close()
            sf_conn.close()
        except Exception as e:
            logger.exception("/snowflake/demo-auto-setup - fetch schemas error for %s: %s", database_row.database_name, str(e))
            raise HTTPException(status_code=400, detail=f"Failed fetching schemas for {database_row.database_name}: {str(e)}")

        for schema_name in schema_names:
            existing_schema = db.query(SnowflakeSchema).filter(
                SnowflakeSchema.database_id == database_row.id,
                SnowflakeSchema.schema_name == schema_name
            ).first()
            if not existing_schema:
                existing_schema = SnowflakeSchema(
                    database_id=database_row.id,
                    schema_name=schema_name,
                    is_selected=True
                )
                db.add(existing_schema)
            else:
                existing_schema.is_selected = True

        db.commit()

    logger.info("/snowflake/demo-auto-setup - completed for connection %s: %d databases", str(conn_uuid), len(database_names))
    return {
        "message": "Demo auto-setup completed",
        "connection_id": str(conn_uuid),
        "databases_count": len(database_names)
    }