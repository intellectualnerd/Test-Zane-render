"""
Service layer for Data Catalog business logic
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, or_, and_
from typing import List, Dict, Optional, Set, Tuple, Any
from uuid import UUID
from app.utils.models import ColumnLevelLineage, TableMetadata, SnowflakeConnection
from app.data_catalog.models import (
    TableSearchResult,
    TableDetailResponse,
    ColumnInfo,
    LineageGraphResponse,
    LineageNode,
    LineageEdge,
    TableIdentifier
)


def build_table_id(database: Optional[str], schema: Optional[str], table_name: str) -> str:
    """
    Build unique table identifier in format: database/schema/table_name
    Falls back to schema/table_name or table_name if database/schema is None
    """
    parts = []
    if database:
        parts.append(database)
    if schema:
        parts.append(schema)
    parts.append(table_name)
    return "/".join(parts)


def parse_table_id(table_id: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parse table identifier back to database, schema, table_name
    Returns: (database, schema, table_name)
    """
    parts = table_id.split("/")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return None, parts[0], parts[1]
    else:
        return None, None, parts[0]


def search_tables(
    db: Session,
    org_id: UUID,
    search_query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    connection_id: Optional[UUID] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    owner: Optional[str] = None,
    tags: Optional[List[str]] = None,
    has_metadata: Optional[bool] = None
) -> Tuple[List[TableSearchResult], int]:
    """
    Search for tables in the lineage data.
    Searches both source and target tables from ColumnLevelLineage.
    
    Args:
        db: Database session
        org_id: Organization ID
        search_query: Optional search query (fuzzy match on table name, schema, database)
        limit: Maximum number of results
        offset: Offset for pagination
        connection_id: Filter by Snowflake connection ID
        database: Filter by database name
        schema: Filter by schema name
        owner: Filter by table owner (from metadata)
        tags: Filter by tags (list of tags - table must have at least one)
        has_metadata: Filter tables with/without metadata (True/False)
        
    Returns:
        Tuple of (list of TableSearchResult, total count)
    """
    # Query to get distinct tables from both source and target
    # We'll use a UNION approach to get unique tables
    
    # Base query for source tables
    source_query = db.query(
        ColumnLevelLineage.source_database,
        ColumnLevelLineage.source_schema,
        ColumnLevelLineage.source_table,
        ColumnLevelLineage.connection_id
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        ColumnLevelLineage.source_table.isnot(None),
        ColumnLevelLineage.source_table != ""
    )
    
    # Base query for target tables
    target_query = db.query(
        ColumnLevelLineage.target_database,
        ColumnLevelLineage.target_schema,
        ColumnLevelLineage.target_table,
        ColumnLevelLineage.connection_id
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        ColumnLevelLineage.target_table.isnot(None),
        ColumnLevelLineage.target_table != ""
    )
    
    # Apply connection_id filter
    if connection_id:
        source_query = source_query.filter(ColumnLevelLineage.connection_id == connection_id)
        target_query = target_query.filter(ColumnLevelLineage.connection_id == connection_id)
    
    # Apply database filter
    if database:
        source_query = source_query.filter(
            func.lower(ColumnLevelLineage.source_database) == func.lower(database)
        )
        target_query = target_query.filter(
            func.lower(ColumnLevelLineage.target_database) == func.lower(database)
        )
    
    # Apply schema filter
    if schema:
        source_query = source_query.filter(
            func.lower(ColumnLevelLineage.source_schema) == func.lower(schema)
        )
        target_query = target_query.filter(
            func.lower(ColumnLevelLineage.target_schema) == func.lower(schema)
        )
    
    # Apply search filter if provided
    if search_query:
        search_pattern = f"%{search_query.lower()}%"
        source_query = source_query.filter(
            or_(
                func.lower(ColumnLevelLineage.source_table).like(search_pattern),
                func.lower(ColumnLevelLineage.source_schema).like(search_pattern),
                func.lower(ColumnLevelLineage.source_database).like(search_pattern)
            )
        )
        target_query = target_query.filter(
            or_(
                func.lower(ColumnLevelLineage.target_table).like(search_pattern),
                func.lower(ColumnLevelLineage.target_schema).like(search_pattern),
                func.lower(ColumnLevelLineage.target_database).like(search_pattern)
            )
        )
    
    # Get all source tables
    source_tables = source_query.distinct().all()
    target_tables = target_query.distinct().all()
    
    # Combine and deduplicate by table identifier
    table_map: Dict[str, TableSearchResult] = {}
    
    for row in source_tables:
        table_id = build_table_id(row.source_database, row.source_schema, row.source_table)
        if table_id not in table_map:
            table_map[table_id] = TableSearchResult(
                id=table_id,
                database=row.source_database,
                schema=row.source_schema,
                table_name=row.source_table,
                connection_id=row.connection_id
            )
    
    for row in target_tables:
        table_id = build_table_id(row.target_database, row.target_schema, row.target_table)
        if table_id not in table_map:
            table_map[table_id] = TableSearchResult(
                id=table_id,
                database=row.target_database,
                schema=row.target_schema,
                table_name=row.target_table,
                connection_id=row.connection_id
            )
    
    # Get column counts for each table
    for table_id, table_result in table_map.items():
        db_name, schema_name, table_name = parse_table_id(table_id)
        column_count = _get_column_count(db, org_id, db_name, schema_name, table_name)
        table_result.column_count = column_count
    
    # Fetch metadata for all tables
    table_ids = list(table_map.keys())
    metadata_map = _fetch_metadata_batch(db, org_id, table_ids)
    
    # Merge metadata into results and apply metadata-based filters
    filtered_table_map: Dict[str, TableSearchResult] = {}
    
    for table_id, table_result in table_map.items():
        metadata = metadata_map.get(table_id)
        
        # Apply has_metadata filter
        if has_metadata is not None:
            has_meta = metadata is not None
            if has_metadata != has_meta:
                continue
        
        # Merge metadata into result
        if metadata:
            table_result.description = metadata.get('description')
            table_result.owner = metadata.get('owner')
            table_result.tags = metadata.get('tags')
        
        # Apply owner filter
        if owner:
            if not table_result.owner or table_result.owner.lower() != owner.lower():
                continue
        
        # Apply tags filter
        if tags:
            if not metadata or not metadata.get('tags'):
                continue
            table_tags = metadata.get('tags', [])
            if not isinstance(table_tags, list):
                continue
            # Check if table has at least one of the requested tags
            table_tags_lower = [str(tag).lower() for tag in table_tags]
            requested_tags_lower = [str(tag).lower() for tag in tags]
            if not any(tag in table_tags_lower for tag in requested_tags_lower):
                continue
        
        filtered_table_map[table_id] = table_result
    
    # Convert to list and sort by table_name
    results = sorted(filtered_table_map.values(), key=lambda x: x.table_name.lower())
    
    # Apply pagination
    total = len(results)
    paginated_results = results[offset:offset + limit]
    
    return paginated_results, total


def _get_column_count(
    db: Session,
    org_id: UUID,
    database: Optional[str],
    schema: Optional[str],
    table_name: str
) -> int:
    """Get count of unique columns for a table from lineage data"""
    # Count distinct columns from both source and target
    source_cols = db.query(
        ColumnLevelLineage.source_column
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.source_table) == func.lower(table_name),
        ColumnLevelLineage.source_column.isnot(None),
        ColumnLevelLineage.source_column != ""
    )
    
    if database:
        source_cols = source_cols.filter(
            func.lower(ColumnLevelLineage.source_database) == func.lower(database)
        )
    if schema:
        source_cols = source_cols.filter(
            func.lower(ColumnLevelLineage.source_schema) == func.lower(schema)
        )
    
    target_cols = db.query(
        ColumnLevelLineage.target_column
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.target_table) == func.lower(table_name),
        ColumnLevelLineage.target_column.isnot(None),
        ColumnLevelLineage.target_column != ""
    )
    
    if database:
        target_cols = target_cols.filter(
            func.lower(ColumnLevelLineage.target_database) == func.lower(database)
        )
    if schema:
        target_cols = target_cols.filter(
            func.lower(ColumnLevelLineage.target_schema) == func.lower(schema)
        )
    
    # Get unique columns
    source_column_set = {col[0].lower() for col in source_cols.distinct().all() if col[0]}
    target_column_set = {col[0].lower() for col in target_cols.distinct().all() if col[0]}
    
    # Union of both sets
    all_columns = source_column_set.union(target_column_set)
    
    return len(all_columns)


def _fetch_metadata_batch(
    db: Session,
    org_id: UUID,
    table_ids: List[str]
) -> Dict[str, Dict[str, any]]:
    """
    Fetch metadata for multiple tables in batch.
    Returns a dictionary mapping table_id to metadata dict.
    """
    if not table_ids:
        return {}
    
    metadata_records = db.query(TableMetadata).filter(
        TableMetadata.org_id == org_id,
        TableMetadata.table_id.in_(table_ids)
    ).all()
    
    metadata_map = {}
    for record in metadata_records:
        metadata_map[record.table_id] = {
            'description': record.description,
            'owner': record.owner,
            'tags': record.tags,
            'column_descriptions': record.column_descriptions,
            'created_at': record.created_at.isoformat() if record.created_at else None,
            'updated_at': record.updated_at.isoformat() if record.updated_at else None
        }
    
    return metadata_map


def _fetch_metadata(
    db: Session,
    org_id: UUID,
    table_id: str
) -> Optional[Dict[str, any]]:
    """
    Fetch metadata for a single table.
    Returns metadata dict or None if not found.
    """
    metadata = db.query(TableMetadata).filter(
        TableMetadata.org_id == org_id,
        TableMetadata.table_id == table_id
    ).first()
    
    if not metadata:
        return None
    
    return {
        'description': metadata.description,
        'owner': metadata.owner,
        'tags': metadata.tags,
        'column_descriptions': metadata.column_descriptions,
        'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
        'updated_at': metadata.updated_at.isoformat() if metadata.updated_at else None
    }


def get_table_detail(
    db: Session,
    org_id: UUID,
    table_id: str
) -> Optional[TableDetailResponse]:
    """
    Get detailed information about a specific table.
    
    Args:
        db: Database session
        org_id: Organization ID
        table_id: Table identifier (database/schema/table_name)
        
    Returns:
        TableDetailResponse or None if table not found
    """
    database, schema, table_name = parse_table_id(table_id)
    
    # First, verify the table exists in lineage data (either as source or target)
    # Use more flexible matching - try exact match first, then fallback to table name only
    source_table_exists = db.query(ColumnLevelLineage).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.source_table) == func.lower(table_name)
    )
    
    target_table_exists = db.query(ColumnLevelLineage).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.target_table) == func.lower(table_name)
    )
    
    # Apply database filter if provided
    if database:
        source_table_exists = source_table_exists.filter(
            or_(
                func.lower(ColumnLevelLineage.source_database) == func.lower(database),
                ColumnLevelLineage.source_database.is_(None)  # Allow None if database not in lineage
            )
        )
        target_table_exists = target_table_exists.filter(
            or_(
                func.lower(ColumnLevelLineage.target_database) == func.lower(database),
                ColumnLevelLineage.target_database.is_(None)  # Allow None if database not in lineage
            )
        )
    
    # Apply schema filter if provided
    if schema:
        source_table_exists = source_table_exists.filter(
            or_(
                func.lower(ColumnLevelLineage.source_schema) == func.lower(schema),
                ColumnLevelLineage.source_schema.is_(None)  # Allow None if schema not in lineage
            )
        )
        target_table_exists = target_table_exists.filter(
            or_(
                func.lower(ColumnLevelLineage.target_schema) == func.lower(schema),
                ColumnLevelLineage.target_schema.is_(None)  # Allow None if schema not in lineage
            )
        )
    
    # Check if table exists at all
    source_exists = source_table_exists.first()
    target_exists = target_table_exists.first()
    
    if not source_exists and not target_exists:
        # Try without database/schema filters as fallback (table name only)
        fallback_source = db.query(ColumnLevelLineage).filter(
            ColumnLevelLineage.org_id == org_id,
            ColumnLevelLineage.is_active == 1,
            func.lower(ColumnLevelLineage.source_table) == func.lower(table_name)
        ).first()
        
        fallback_target = db.query(ColumnLevelLineage).filter(
            ColumnLevelLineage.org_id == org_id,
            ColumnLevelLineage.is_active == 1,
            func.lower(ColumnLevelLineage.target_table) == func.lower(table_name)
        ).first()
        
        if not fallback_source and not fallback_target:
            return None
        # If found in fallback, use table name only (ignore database/schema filters)
        # Reset database and schema to None so queries below don't filter by them
        database = None
        schema = None
    
    # Query to get all columns for this table
    source_cols_query = db.query(
        ColumnLevelLineage.source_column
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.source_table) == func.lower(table_name),
        ColumnLevelLineage.source_column.isnot(None),
        ColumnLevelLineage.source_column != ""
    )
    
    target_cols_query = db.query(
        ColumnLevelLineage.target_column
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.target_table) == func.lower(table_name),
        ColumnLevelLineage.target_column.isnot(None),
        ColumnLevelLineage.target_column != ""
    )
    
    if database:
        source_cols_query = source_cols_query.filter(
            func.lower(ColumnLevelLineage.source_database) == func.lower(database)
        )
        target_cols_query = target_cols_query.filter(
            func.lower(ColumnLevelLineage.target_database) == func.lower(database)
        )
    
    if schema:
        source_cols_query = source_cols_query.filter(
            func.lower(ColumnLevelLineage.source_schema) == func.lower(schema)
        )
        target_cols_query = target_cols_query.filter(
            func.lower(ColumnLevelLineage.target_schema) == func.lower(schema)
        )
    
    # Get unique columns
    source_columns = {col[0] for col in source_cols_query.distinct().all() if col[0]}
    target_columns = {col[0] for col in target_cols_query.distinct().all() if col[0]}
    all_columns = sorted(source_columns.union(target_columns))
    
    # Get connection_id from first record (try source first, then target)
    sample_query = db.query(ColumnLevelLineage.connection_id).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        func.lower(ColumnLevelLineage.source_table) == func.lower(table_name)
    )
    if database:
        sample_query = sample_query.filter(
            func.lower(ColumnLevelLineage.source_database) == func.lower(database)
        )
    if schema:
        sample_query = sample_query.filter(
            func.lower(ColumnLevelLineage.source_schema) == func.lower(schema)
        )
    
    connection_id = sample_query.first()
    if not connection_id:
        # Try target table
        sample_query = db.query(ColumnLevelLineage.connection_id).filter(
            ColumnLevelLineage.org_id == org_id,
            ColumnLevelLineage.is_active == 1,
            func.lower(ColumnLevelLineage.target_table) == func.lower(table_name)
        )
        if database:
            sample_query = sample_query.filter(
                func.lower(ColumnLevelLineage.target_database) == func.lower(database)
            )
        if schema:
            sample_query = sample_query.filter(
                func.lower(ColumnLevelLineage.target_schema) == func.lower(schema)
            )
        connection_id = sample_query.first()
    
    connection_id = connection_id[0] if connection_id else None
    
    # Fetch metadata
    metadata = _fetch_metadata(db, org_id, table_id)
    column_descriptions = metadata.get('column_descriptions', {}) if metadata else {}
    
    # Build column info list with descriptions from metadata
    columns = []
    for col in all_columns:
        col_info = ColumnInfo(
            column_name=col,
            description=column_descriptions.get(col) if isinstance(column_descriptions, dict) else None
        )
        columns.append(col_info)
    
    return TableDetailResponse(
        id=table_id,
        database=database,
        schema=schema,
        table_name=table_name,
        connection_id=connection_id,
        description=metadata.get('description') if metadata else None,
        owner=metadata.get('owner') if metadata else None,
        columns=columns,
        created_at=metadata.get('created_at') if metadata else None,
        updated_at=metadata.get('updated_at') if metadata else None
    )


def get_table_lineage(
    db: Session,
    org_id: UUID,
    table_id: str
) -> LineageGraphResponse:
    """
    Get lineage graph for a specific table.
    Similar to the analysis endpoint format.
    
    Args:
        db: Database session
        org_id: Organization ID
        table_id: Table identifier (database/schema/table_name)
        
    Returns:
        LineageGraphResponse with upstream and downstream lineage
    """
    from app.api.github import get_upstream_lineage, get_recursive_downstream_lineage
    
    database, schema, table_name = parse_table_id(table_id)
    
    # Build center node
    center_node = LineageNode(
        database=database,
        schema=schema,
        table=table_name,
        id=table_id
    )
    
    # Get upstream lineage (what feeds into this table)
    upstream_lineage = get_upstream_lineage(
        db=db,
        org_id=org_id,
        target_database=database,
        target_schema=schema,
        target_table=table_name,
        target_column=None  # Get all columns
    )
    
    # Get downstream lineage (what this table feeds into)
    downstream_lineage = get_recursive_downstream_lineage(
        db=db,
        org_id=org_id,
        source_database=database,
        source_schema=schema,
        source_table=table_name,
        source_column=None  # Get all columns
    )
    
    return LineageGraphResponse(
        center_node=center_node,
        upstream=upstream_lineage,
        downstream=downstream_lineage
    )


def create_or_update_table_metadata(
    db: Session,
    org_id: UUID,
    table_id: str,
    database: Optional[str],
    schema: Optional[str],
    table_name: str,
    description: Optional[str] = None,
    owner: Optional[str] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    tags: Optional[List[str]] = None,
    user_id: Optional[UUID] = None
) -> TableMetadata:
    """
    Create or update table metadata.
    If metadata exists, updates it; otherwise creates new record.
    
    Returns:
        TableMetadata object
    """
    # Check if metadata already exists
    existing_metadata = db.query(TableMetadata).filter(
        TableMetadata.org_id == org_id,
        TableMetadata.table_id == table_id
    ).first()
    
    if existing_metadata:
        # Update existing
        if description is not None:
            existing_metadata.description = description
        if owner is not None:
            existing_metadata.owner = owner
        if column_descriptions is not None:
            existing_metadata.column_descriptions = column_descriptions
        if tags is not None:
            existing_metadata.tags = tags
        if user_id:
            existing_metadata.updated_by = user_id
        
        db.commit()
        db.refresh(existing_metadata)
        return existing_metadata
    else:
        # Create new
        new_metadata = TableMetadata(
            org_id=org_id,
            table_id=table_id,
            database=database,
            schema=schema,
            table_name=table_name,
            description=description,
            owner=owner,
            column_descriptions=column_descriptions,
            tags=tags,
            created_by=user_id,
            updated_by=user_id
        )
        db.add(new_metadata)
        db.commit()
        db.refresh(new_metadata)
        return new_metadata


def get_table_metadata(
    db: Session,
    org_id: UUID,
    table_id: str
) -> Optional[TableMetadata]:
    """
    Get table metadata record.
    
    Returns:
        TableMetadata object or None if not found
    """
    return db.query(TableMetadata).filter(
        TableMetadata.org_id == org_id,
        TableMetadata.table_id == table_id
    ).first()


def delete_table_metadata(
    db: Session,
    org_id: UUID,
    table_id: str
) -> bool:
    """
    Delete table metadata.
    
    Returns:
        True if deleted, False if not found
    """
    metadata = db.query(TableMetadata).filter(
        TableMetadata.org_id == org_id,
        TableMetadata.table_id == table_id
    ).first()
    
    if not metadata:
        return False
    
    db.delete(metadata)
    db.commit()
    return True


def get_connections_for_org(
    db: Session,
    org_id: UUID
) -> List[Dict[str, Any]]:
    """
    Get list of unique connections that have tables in the lineage data.
    Returns both connection ID and name.
    
    Returns:
        List of dicts with 'id' and 'name' keys
    """
    # Get distinct connection IDs from lineage data
    connection_ids = db.query(
        distinct(ColumnLevelLineage.connection_id)
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1
    ).all()
    
    connection_ids_list = [conn[0] for conn in connection_ids if conn[0]]
    
    if not connection_ids_list:
        return []
    
    # Fetch connection names from SnowflakeConnection table
    connections = db.query(
        SnowflakeConnection.id,
        SnowflakeConnection.connection_name
    ).filter(
        SnowflakeConnection.id.in_(connection_ids_list),
        SnowflakeConnection.org_id == org_id,
        SnowflakeConnection.is_active == True
    ).all()
    
    # Return list of dicts with id and name
    return [
        {
            'id': str(conn.id),
            'name': conn.connection_name
        }
        for conn in connections
    ]


def get_databases_for_connection(
    db: Session,
    org_id: UUID,
    connection_id: UUID
) -> List[str]:
    """
    Get list of unique database names for a given connection.
    Searches both source and target tables.
    
    Returns:
        List of database names (sorted)
    """
    # Get databases from source tables
    source_dbs = db.query(
        distinct(ColumnLevelLineage.source_database)
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        ColumnLevelLineage.connection_id == connection_id,
        ColumnLevelLineage.source_database.isnot(None),
        ColumnLevelLineage.source_database != ""
    ).all()
    
    # Get databases from target tables
    target_dbs = db.query(
        distinct(ColumnLevelLineage.target_database)
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        ColumnLevelLineage.connection_id == connection_id,
        ColumnLevelLineage.target_database.isnot(None),
        ColumnLevelLineage.target_database != ""
    ).all()
    
    # Combine and deduplicate
    db_set = set()
    for db_row in source_dbs:
        if db_row[0]:
            db_set.add(db_row[0])
    for db_row in target_dbs:
        if db_row[0]:
            db_set.add(db_row[0])
    
    return sorted(list(db_set))


def get_schemas_for_connection_database(
    db: Session,
    org_id: UUID,
    connection_id: UUID,
    database: str
) -> List[str]:
    """
    Get list of unique schema names for a given connection and database.
    Searches both source and target tables.
    
    Returns:
        List of schema names (sorted)
    """
    # Get schemas from source tables
    source_schemas = db.query(
        distinct(ColumnLevelLineage.source_schema)
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        ColumnLevelLineage.connection_id == connection_id,
        func.lower(ColumnLevelLineage.source_database) == func.lower(database),
        ColumnLevelLineage.source_schema.isnot(None),
        ColumnLevelLineage.source_schema != ""
    ).all()
    
    # Get schemas from target tables
    target_schemas = db.query(
        distinct(ColumnLevelLineage.target_schema)
    ).filter(
        ColumnLevelLineage.org_id == org_id,
        ColumnLevelLineage.is_active == 1,
        ColumnLevelLineage.connection_id == connection_id,
        func.lower(ColumnLevelLineage.target_database) == func.lower(database),
        ColumnLevelLineage.target_schema.isnot(None),
        ColumnLevelLineage.target_schema != ""
    ).all()
    
    # Combine and deduplicate
    schema_set = set()
    for schema_row in source_schemas:
        if schema_row[0]:
            schema_set.add(schema_row[0])
    for schema_row in target_schemas:
        if schema_row[0]:
            schema_set.add(schema_row[0])
    
    return sorted(list(schema_set))


def get_owners_for_org(
    db: Session,
    org_id: UUID
) -> List[str]:
    """
    Get list of unique owners from table metadata.
    
    Returns:
        List of owner names (sorted)
    """
    owners = db.query(
        distinct(TableMetadata.owner)
    ).filter(
        TableMetadata.org_id == org_id,
        TableMetadata.owner.isnot(None),
        TableMetadata.owner != ""
    ).all()
    
    return sorted([owner[0] for owner in owners if owner[0]])


def get_tags_for_org(
    db: Session,
    org_id: UUID
) -> List[str]:
    """
    Get list of unique tags from table metadata.
    Tags are stored as JSONB arrays.
    
    Returns:
        List of unique tag names (sorted)
    """
    # Query all metadata records with tags
    metadata_records = db.query(TableMetadata.tags).filter(
        TableMetadata.org_id == org_id,
        TableMetadata.tags.isnot(None)
    ).all()
    
    # Collect all tags
    tag_set = set()
    for record in metadata_records:
        if record.tags and isinstance(record.tags, list):
            for tag in record.tags:
                if tag:
                    tag_set.add(str(tag))
    
    return sorted(list(tag_set))

