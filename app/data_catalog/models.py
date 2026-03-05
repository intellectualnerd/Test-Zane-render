"""
Pydantic models for Data Catalog API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID


class TableIdentifier(BaseModel):
    """Unique identifier for a table"""
    database: Optional[str] = None
    schema: Optional[str] = None
    table_name: str
    id: str  # Format: database/schema/table_name or schema/table_name or table_name


class TableSearchResult(BaseModel):
    """Single table search result"""
    id: str  # Unique identifier: database/schema/table_name
    database: Optional[str] = None
    schema: Optional[str] = None
    table_name: str
    connection_id: Optional[UUID] = None
    # Metadata (will be populated from metadata table if available)
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[List[str]] = None  # Tags from metadata
    column_count: Optional[int] = None  # Number of columns found in lineage


class TableSearchResponse(BaseModel):
    """Response for table search"""
    results: List[TableSearchResult]
    total: int


class ColumnInfo(BaseModel):
    """Column information"""
    column_name: str
    description: Optional[str] = None
    data_type: Optional[str] = None


class TableDetailResponse(BaseModel):
    """Detailed table information"""
    id: str  # Unique identifier: database/schema/table_name
    database: Optional[str] = None
    schema: Optional[str] = None
    table_name: str
    connection_id: Optional[UUID] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    columns: List[ColumnInfo] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class LineageNode(BaseModel):
    """Node in the lineage graph"""
    database: Optional[str] = None
    schema: Optional[str] = None
    table: str
    column: Optional[str] = None
    id: str  # Unique identifier: database/schema/table or database/schema/table.column


class LineageEdge(BaseModel):
    """Edge in the lineage graph"""
    source_id: str
    target_id: str
    query_id: Optional[List[str]] = None
    query_type: Optional[str] = None
    dbt_model_file_path: Optional[str] = None


class LineageGraphResponse(BaseModel):
    """Lineage graph response"""
    center_node: Optional[LineageNode] = None
    upstream: List[Dict[str, Any]] = []  # Upstream lineage data
    downstream: List[Dict[str, Any]] = []  # Downstream lineage data
    nodes: List[LineageNode] = []
    edges: List[LineageEdge] = []


class TableMetadataCreate(BaseModel):
    """Request model for creating/updating table metadata"""
    database: Optional[str] = None
    schema: Optional[str] = None
    table_name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    column_descriptions: Optional[Dict[str, str]] = None  # column_name -> description

