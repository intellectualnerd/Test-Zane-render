"""
Data Catalog API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID

from app.database import get_db
from app.utils.auth_deps import get_current_user
from app.utils.models import User
from app.data_catalog.service import (
    search_tables,
    get_table_detail,
    get_table_lineage,
    create_or_update_table_metadata,
    delete_table_metadata,
    build_table_id,
    parse_table_id,
    get_connections_for_org,
    get_databases_for_connection,
    get_schemas_for_connection_database,
    get_owners_for_org,
    get_tags_for_org
)
from app.data_catalog.models import (
    TableSearchResponse,
    TableDetailResponse,
    LineageGraphResponse,
    TableMetadataCreate
)

router = APIRouter(prefix="/data-catalog", tags=["Data Catalog"])


@router.get("/search", response_model=TableSearchResponse)
def search_tables_endpoint(
    q: Optional[str] = Query(None, description="Search query (searches table name, schema, database)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    connection_id: Optional[UUID] = Query(None, description="Filter by Snowflake connection ID"),
    database: Optional[str] = Query(None, description="Filter by database name"),
    schema: Optional[str] = Query(None, description="Filter by schema name"),
    owner: Optional[str] = Query(None, description="Filter by table owner"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    has_metadata: Optional[bool] = Query(None, description="Filter tables with/without metadata"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search for tables in the data catalog.
    Searches through all tables present in the lineage data.
    
    Supports advanced filtering:
    - Filter by connection_id, then database, then schema (hierarchical)
    - Filter by owner and tags (from metadata)
    - Filter by has_metadata (True/False)
    """
    try:
        # Parse tags from comma-separated string
        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        results, total = search_tables(
            db=db,
            org_id=current_user.org_id,
            search_query=q,
            limit=limit,
            offset=offset,
            connection_id=connection_id,
            database=database,
            schema=schema,
            owner=owner,
            tags=tags_list,
            has_metadata=has_metadata
        )
        
        return TableSearchResponse(
            results=results,
            total=total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching tables: {str(e)}")


# IMPORTANT: More specific routes must come BEFORE less specific ones
# Otherwise FastAPI will match /lineage or /metadata as part of table_id

@router.get("/tables/{table_id:path}/lineage", response_model=LineageGraphResponse)
def get_table_lineage_endpoint(
    table_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get lineage graph for a specific table.
    Returns upstream and downstream lineage data similar to the analysis endpoint format.
    
    Note: table_id can contain slashes (e.g., "database/schema/table_name")
    """
    try:
        # URL decode the table_id in case it was double-encoded
        from urllib.parse import unquote
        table_id = unquote(table_id)
        
        # First verify table exists
        table_detail = get_table_detail(
            db=db,
            org_id=current_user.org_id,
            table_id=table_id
        )
        
        if not table_detail:
            raise HTTPException(status_code=404, detail=f"Table not found: {table_id}")
        
        lineage = get_table_lineage(
            db=db,
            org_id=current_user.org_id,
            table_id=table_id
        )
        
        return lineage
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching lineage: {str(e)}")


@router.get("/tables/{table_id:path}/metadata", response_model=TableDetailResponse)
def get_table_metadata_endpoint(
    table_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific table including metadata, columns, and basic information.
    Returns the same response as the table detail endpoint.
    
    Note: table_id can contain slashes (e.g., "database/schema/table_name")
    """
    try:
        # URL decode the table_id in case it was double-encoded
        from urllib.parse import unquote
        table_id = unquote(table_id)
        
        table_detail = get_table_detail(
            db=db,
            org_id=current_user.org_id,
            table_id=table_id
        )
        
        if not table_detail:
            raise HTTPException(status_code=404, detail=f"Table not found: {table_id}")
        
        return table_detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching table details: {str(e)}")


@router.put("/tables/{table_id:path}/metadata", response_model=TableDetailResponse)
def create_or_update_table_metadata_endpoint(
    table_id: str,
    metadata: TableMetadataCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create or update table metadata.
    This endpoint allows users to add/edit table descriptions, column descriptions, owners, etc.
    """
    try:
        # URL decode the table_id in case it was double-encoded
        from urllib.parse import unquote
        table_id = unquote(table_id)
        
        # Verify table exists by trying to get its detail
        table_detail = get_table_detail(
            db=db,
            org_id=current_user.org_id,
            table_id=table_id
        )
        
        if not table_detail:
            raise HTTPException(status_code=404, detail=f"Table not found: {table_id}")
        
        # Parse table_id to get components
        database, schema, table_name = parse_table_id(table_id)
        
        # Create or update metadata
        metadata_record = create_or_update_table_metadata(
            db=db,
            org_id=current_user.org_id,
            table_id=table_id,
            database=database,
            schema=schema,
            table_name=table_name,
            description=metadata.description,
            owner=metadata.owner,
            column_descriptions=metadata.column_descriptions,
            user_id=current_user.id
        )
        
        # Return updated table detail
        updated_detail = get_table_detail(
            db=db,
            org_id=current_user.org_id,
            table_id=table_id
        )
        
        return updated_detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving metadata: {str(e)}")


@router.delete("/tables/{table_id:path}/metadata")
def delete_table_metadata_endpoint(
    table_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete table metadata.
    """
    try:
        # URL decode the table_id in case it was double-encoded
        from urllib.parse import unquote
        table_id = unquote(table_id)
        
        deleted = delete_table_metadata(
            db=db,
            org_id=current_user.org_id,
            table_id=table_id
        )
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Metadata not found for table: {table_id}")
        
        return {"message": "Metadata deleted successfully", "table_id": table_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting metadata: {str(e)}")


# Filter helper endpoints for hierarchical filtering

@router.get("/filters/connections")
def get_connections_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of connections (with ID and name) that have tables in the lineage data.
    Used for hierarchical filtering: connection → database → schema
    
    Returns:
        {
            "connections": [
                {"id": "uuid", "name": "connection_name"},
                ...
            ]
        }
    """
    try:
        connections = get_connections_for_org(db=db, org_id=current_user.org_id)
        return {"connections": connections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching connections: {str(e)}")


@router.get("/filters/connections/{connection_id}/databases")
def get_databases_endpoint(
    connection_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of database names for a given connection.
    Used for hierarchical filtering after selecting a connection.
    """
    try:
        databases = get_databases_for_connection(
            db=db,
            org_id=current_user.org_id,
            connection_id=connection_id
        )
        return {"databases": databases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching databases: {str(e)}")


@router.get("/filters/connections/{connection_id}/databases/{database}/schemas")
def get_schemas_endpoint(
    connection_id: UUID,
    database: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of schema names for a given connection and database.
    Used for hierarchical filtering after selecting a connection and database.
    """
    try:
        # URL decode the database name in case it was encoded
        from urllib.parse import unquote
        database = unquote(database)
        
        schemas = get_schemas_for_connection_database(
            db=db,
            org_id=current_user.org_id,
            connection_id=connection_id,
            database=database
        )
        return {"schemas": schemas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching schemas: {str(e)}")


@router.get("/filters/owners")
def get_owners_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of unique owners from table metadata.
    Used for filtering tables by owner.
    """
    try:
        owners = get_owners_for_org(db=db, org_id=current_user.org_id)
        return {"owners": owners}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching owners: {str(e)}")


@router.get("/filters/tags")
def get_tags_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of unique tags from table metadata.
    Used for filtering tables by tags.
    """
    try:
        tags = get_tags_for_org(db=db, org_id=current_user.org_id)
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tags: {str(e)}")

