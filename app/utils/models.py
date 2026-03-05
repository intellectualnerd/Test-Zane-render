from sqlalchemy import Column, String, DateTime, Text, Boolean, ForeignKey, Integer, BigInteger, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
import uuid


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    is_connection_setup = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    users = relationship("User", back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    role = Column(String(50), nullable=False, default="MEMBER", index=True)  # PRODUCT_SUPPORT_ADMIN, SYSTEM_ADMIN, ORGANIZATION_ADMIN, MEMBER
    password_reset_token = Column(String(255), nullable=True)
    password_reset_token_expires = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    organization = relationship("Organization", back_populates="users")


class UserToken(Base):
    __tablename__ = "user_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    token = Column(Text, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SnowflakeConnection(Base):
    __tablename__ = "snowflake_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    connection_name = Column(String(100), nullable=False)
    account = Column(String(100), nullable=False)
    username = Column(String(100), nullable=False)
    password = Column(String(255), nullable=False)
    warehouse = Column(String(100), nullable=True)
    role = Column(String(100), nullable=True)
    cron_expression = Column(String(100), nullable=True)  # Miner config
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", backref="snowflake_connections")
    databases = relationship("SnowflakeDatabase", back_populates="connection", cascade="all, delete-orphan")
    job = relationship("SnowflakeJob", back_populates="connection", uselist=False, cascade="all, delete-orphan")


class SnowflakeDatabase(Base):
    __tablename__ = "snowflake_databases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), nullable=False, index=True)
    database_name = Column(String(100), nullable=False)
    is_selected = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    connection = relationship("SnowflakeConnection", back_populates="databases")
    schemas = relationship("SnowflakeSchema", back_populates="database", cascade="all, delete-orphan")


class SnowflakeSchema(Base):
    __tablename__ = "snowflake_schemas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    database_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_databases.id"), nullable=False, index=True)
    schema_name = Column(String(100), nullable=False)
    is_selected = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    database = relationship("SnowflakeDatabase", back_populates="schemas")


class GitHubInstallation(Base):
    __tablename__ = "github_installations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    installation_id = Column(String(50), unique=True, nullable=False, index=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    account_type = Column(String(20), nullable=False)  # 'User' or 'Organization'
    account_login = Column(String(100), nullable=False)
    repository_selection = Column(String(20), nullable=False)  # 'all' or 'selected'
    permissions = Column(Text, nullable=True)  # JSON string of permissions
    events = Column(Text, nullable=True)  # JSON string of events
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", backref="github_installations")
    repositories = relationship("GitHubRepository", back_populates="installation", cascade="all, delete-orphan")


class GitHubRepository(Base):
    __tablename__ = "github_repositories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    installation_id = Column(UUID(as_uuid=True), ForeignKey("github_installations.id"), nullable=False, index=True)
    repo_id = Column(String(50), nullable=False, index=True)
    repo_name = Column(String(200), nullable=False)
    full_name = Column(String(200), nullable=False)
    private = Column(Boolean, default=False)
    description = Column(Text, nullable=True)
    default_branch = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    installation = relationship("GitHubInstallation", back_populates="repositories")


# New code

class JiraConnection(Base):
    __tablename__ = "jira_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    connection_name = Column(String(100), nullable=False)
    server_url = Column(String(255), nullable=False)  # e.g., https://company.atlassian.net
    username = Column(String(100), nullable=False)  # Email for Atlassian Cloud
    api_token = Column(String(255), nullable=False)  # API token or password
    project_key = Column(String(20), nullable=True)  # Default project key for tickets
    issue_type = Column(String(50), nullable=True, default="Task")  # Default issue type
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", backref="jira_connections")


class JiraTicket(Base):
    __tablename__ = "jira_tickets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("jira_connections.id"), nullable=False, index=True)
    ticket_key = Column(String(50), nullable=False, index=True)  # e.g., PROJ-123
    ticket_url = Column(String(500), nullable=False)
    summary = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    issue_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    priority = Column(String(50), nullable=True)
    assignee = Column(String(100), nullable=True)
    pr_url = Column(String(500), nullable=True)  # Related PR URL
    analysis_report_url = Column(String(500), nullable=True)  # Analysis report URL
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    connection = relationship("JiraConnection", backref="tickets")
    creator = relationship("User", backref="created_jira_tickets")


# dbt Cloud connection model

class DbtCloudConnection(Base):
    __tablename__ = "dbt_cloud_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    connection_name = Column(String(100), nullable=False)
    api_key = Column(String(255), nullable=False)
    account_id = Column(String(100), nullable=False)
    base_url = Column(String(255), nullable=False)  # e.g., https://api.getdbt.com
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", backref="dbt_cloud_connections")


# dbt Cloud metadata storage models

class DbtProject(Base):
    __tablename__ = "dbt_projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("dbt_cloud_connections.id"), nullable=False, index=True)
    project_id = Column(String(100), nullable=False, index=True)
    account_id = Column(String(100), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    state = Column(String(50), nullable=True)
    type = Column(String(50), nullable=True)
    dbt_project_subdirectory = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)

    organization = relationship("Organization", backref="dbt_projects")


class DbtRun(Base):
    __tablename__ = "dbt_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("dbt_cloud_connections.id"), nullable=False, index=True)
    run_id = Column(String(100), nullable=False, index=True)
    job_id = Column(String(100), nullable=True)
    account_id = Column(String(100), nullable=True)
    project_id = Column(String(100), nullable=True)
    status = Column(String(50), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    duration = Column(String(50), nullable=True)
    trigger = Column(String(100), nullable=True)


class DbtManifestNode(Base):
    __tablename__ = "dbt_manifest_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("dbt_cloud_connections.id"), nullable=False, index=True)
    run_id = Column(String(100), nullable=False, index=True)
    unique_id = Column(String(500), nullable=False, index=True)
    database = Column(String(200), nullable=True)
    schema = Column(String(200), nullable=True)
    name = Column(String(255), nullable=True)
    package_name = Column(String(255), nullable=True)
    path = Column(String(500), nullable=True)
    original_file_path = Column(String(500), nullable=True)
    resource_type = Column(String(100), nullable=True)
    raw_code = Column(Text, nullable=True)
    compiled_code = Column(Text, nullable=True)
    downstream_models = Column(JSONB, nullable=True)
    last_successful_run_at = Column(DateTime(timezone=True), nullable=True)
    synced_at = Column(DateTime(timezone=True), nullable=True)


class DbtCrawlAudit(Base):
    __tablename__ = "dbt_crawl_audit"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("dbt_jobs.id"), nullable=True, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("dbt_cloud_connections.id"), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="running")  # running|success|failed
    nodes_inserted = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)


class DbtJob(Base):
    __tablename__ = "dbt_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("dbt_cloud_connections.id"), unique=True, nullable=False, index=True)
    cron_expression = Column(String(100), nullable=False)
    last_run_time = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Snowflake crawler job and audit models

class SnowflakeJob(Base):
    __tablename__ = "snowflake_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), unique=True, nullable=False, index=True)
    cron_expression = Column(String(100), nullable=False)
    last_run_time = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    connection = relationship("SnowflakeConnection", back_populates="job")


class SnowflakeCrawlAudit(Base):
    __tablename__ = "snowflake_crawl_audit"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    batch_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), nullable=False, index=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="running")  # running|success|failed
    query_history_rows_fetched = Column(Integer, nullable=False, default=0)
    information_schema_columns_rows_fetched = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)

    organization = relationship("Organization", foreign_keys=[org_id], backref="snowflake_crawl_audits_org_id")
    connection = relationship("SnowflakeConnection", foreign_keys=[connection_id], backref="snowflake_crawl_audits_conn_id")


class SnowflakeQueryRecord(Base):
    __tablename__ = "snowflake_query_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    batch_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), nullable=False, index=True)
    query_id = Column(String, nullable=False, index=True)
    query_text = Column(Text, nullable=True)
    database_name = Column(String, nullable=True)
    database_id = Column(Integer, nullable=True)
    schema_name = Column(String, nullable=True)
    schema_id = Column(Integer, nullable=True)
    query_type = Column(String, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    session_id = Column(BigInteger, nullable=True)
    base_objects_accessed = Column(JSONB, nullable=True)
    objects_modified = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    organization = relationship("Organization", foreign_keys=[org_id], backref="snowflake_query_record_ord_id")
    connection = relationship("SnowflakeConnection", foreign_keys=[connection_id], backref="snowflake_query_record_conn_id")


class InformationSchemacolumns(Base):
    __tablename__ = "information_schema_columns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), nullable=False, index=True)
    table_catalog = Column(String, nullable=True)
    table_schema = Column(String, nullable=True)
    table_name = Column(String, nullable=True)
    column_name = Column(String, nullable=True)
    data_type = Column(String, nullable=True)
    ordinal_position = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    organization = relationship("Organization", foreign_keys=[org_id], backref="information_schema_columns_org_id")
    connection = relationship("SnowflakeConnection", foreign_keys=[connection_id], backref="information_schema_columns_conn_id")

class ChatThread(Base):
    """Chat conversation thread - represents a chat session/conversation"""
    __tablename__ = "chat_threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)  # Auto-generated from first message or user-provided
    is_active = Column(Boolean, default=True)  # Soft delete flag
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True), nullable=True)  # Timestamp of last message

    # Relationships
    organization = relationship("Organization", backref="chat_threads")
    user = relationship("User", backref="chat_threads")
    messages = relationship("ChatMessage", back_populates="thread", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    """Individual chat message within a thread"""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("chat_threads.id"), nullable=False, index=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)  # Message content
    message_metadata = Column(JSONB, nullable=True)  # Store additional data like impacted_queries, pr_repo_data, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    thread = relationship("ChatThread", back_populates="messages")
    organization = relationship("Organization", backref="chat_messages")
    user = relationship("User", backref="chat_messages")


class ColumnLevelLineage(Base):
    __tablename__ = "column_level_lineage"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    batch_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), nullable=False, index=True)
    source_database = Column(String, nullable=True)
    source_schema = Column(String, nullable=True)
    source_table = Column(String, nullable=True)
    source_column = Column(String, nullable=True)
    target_database = Column(String, nullable=True)
    target_schema = Column(String, nullable=True)
    target_table = Column(String, nullable=True)
    target_column = Column(String, nullable=True)
    query_id = Column(JSONB, nullable=True)  # store list of query IDs
    query_type = Column(String, nullable=True)
    session_id = Column(BigInteger, nullable=True)
    dependency_score = Column(Integer, nullable=True)
    dbt_model_file_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Integer, nullable=False, server_default="1")
    is_filter_clause = Column(Integer, nullable=False, server_default="0")
    
    # Relationships
    organization = relationship("Organization", foreign_keys=[org_id], backref="column_level_lineage_org_id")
    connection = relationship("SnowflakeConnection", foreign_keys=[connection_id], backref="column_level_lineage_conn_id")

# class FilterClauseColumnLineage(Base):
#     __tablename__ = "filter_clause_column_lineage"

#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
#     batch_id = Column(UUID(as_uuid=True), nullable=False, index=True)
#     connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), nullable=False, index=True)
#     source_database = Column(String, nullable=True)
#     source_schema = Column(String, nullable=True)
#     source_table = Column(String, nullable=True)
#     source_column = Column(String, nullable=True)
#     target_database = Column(String, nullable=True)
#     target_schema = Column(String, nullable=True)
#     target_table = Column(String, nullable=True)
#     target_column = Column(String, nullable=True)
#     query_id = Column(JSONB, nullable=True)  # store list of query IDs
#     query_type = Column(String, nullable=True)
#     session_id = Column(BigInteger, nullable=True)
#     dependency_score = Column(Integer, nullable=True)
#     dbt_model_file_path = Column(String, nullable=True)
#     created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
#     updated_at = Column(DateTime(timezone=True), onupdate=func.now())
#     is_active = Column(Integer, nullable=False, server_default="1")
    
#     # Relationships
#     organization = relationship("Organization", foreign_keys=[org_id], backref="filter_clause_column_org_id")
#     connection = relationship("SnowflakeConnection", foreign_keys=[connection_id], backref="filter_clause_column_conn_id")


class TableMetadata(Base):
    """
    Metadata for tables in the data catalog.
    Stores table descriptions, column descriptions, owners, and other metadata.
    """
    __tablename__ = "table_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    
    # Table identifier - stored as: database/schema/table_name or schema/table_name or table_name
    table_id = Column(String(500), nullable=False, index=True)
    
    # Individual components for easier querying
    database = Column(String(200), nullable=True, index=True)
    schema = Column(String(200), nullable=True, index=True)
    table_name = Column(String(200), nullable=False, index=True)
    
    # Metadata fields
    description = Column(Text, nullable=True)  # Table description
    owner = Column(String(200), nullable=True)  # Table owner
    tags = Column(JSONB, nullable=True)  # Array of tags for categorization
    column_descriptions = Column(JSONB, nullable=True)  # Dictionary: {column_name: description}
    
    # Audit fields
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", foreign_keys=[org_id], backref="table_metadata_org_id")
    creator = relationship("User", foreign_keys=[created_by], backref="table_metadata_created")
    updater = relationship("User", foreign_keys=[updated_by], backref="table_metadata_updated")
    
    # Unique constraint: one metadata record per table per org
    __table_args__ = (
        UniqueConstraint('org_id', 'table_id', name='uq_table_metadata_org_table'),
    )


class LineageLoadWatermark(Base):
    __tablename__ = "lineage_load_watermarks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    batch_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("snowflake_connections.id"), nullable=False, index=True)
    last_processed_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    organization = relationship("Organization", foreign_keys=[org_id], backref="lineage_load_watermark_org_id")
    connection = relationship("SnowflakeConnection", foreign_keys=[connection_id], backref="lineage_watermarks_conn_id")
      



class GitHubPullRequestAnalysis(Base):
    __tablename__ = "github_pr_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    installation_id = Column(UUID(as_uuid=True), ForeignKey("github_installations.id"), nullable=False, index=True)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("github_repositories.id"), nullable=True, index=True)
    repo_full_name = Column(String(200), nullable=False, index=True)
    pr_number = Column(Integer, nullable=False, index=True)
    pr_title = Column(String(500), nullable=True)
    pr_description = Column(Text, nullable=True)
    branch_name = Column(String(200), nullable=True)
    author_name = Column(String(100), nullable=True)
    total_impacted_queries = Column(Integer, nullable=True)
    pr_url = Column(String(500), nullable=True)
    analysis_data = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Optional relationships
    # organization = relationship("Organization", backref="pr_analyses")
    # installation = relationship("GitHubInstallation", backref="pr_analyses")
    # repository = relationship("GitHubRepository", backref="pr_analyses")


# Pydantic models for WebSocket messages and chat
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

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
    """Model for AI response data"""
    response: str
    sources: List[Dict[str, Any]] = []
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

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
