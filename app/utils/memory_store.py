"""
LangGraph PostgresStore-based long-term memory management for the chatbot.

This module provides functions to store and retrieve long-term memories using
LangGraph's PostgresStore, which persists memories across sessions and threads.

References:
- https://docs.langchain.com/oss/python/langgraph/memory#long-term-memory
- https://docs.langchain.com/oss/python/langgraph/add-memory#add-long-term-memory
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# Import PostgresStore from langgraph
# Try multiple import paths as LangGraph structure may vary by version
POSTGRES_STORE_AVAILABLE = False
PostgresStore = None
BaseStore = None

try:
    from langgraph.store.postgres import PostgresStore
    from langgraph.store.base import BaseStore
    POSTGRES_STORE_AVAILABLE = True
    logger.info("PostgresStore imported successfully from langgraph.store.postgres")
except ImportError as e1:
    try:
        # Alternative import path (some versions)
        from langgraph.store import PostgresStore
        from langgraph.store import BaseStore
        POSTGRES_STORE_AVAILABLE = True
        logger.info("PostgresStore imported successfully from langgraph.store")
    except ImportError as e2:
        POSTGRES_STORE_AVAILABLE = False
        logger.warning(
            f"PostgresStore not available. Import errors: {e1}, {e2}. "
            f"Install langgraph: pip install langgraph"
        )

# Global store instance (initialized on first use)
_store: Optional[Any] = None


def get_memory_store() -> Any:
    """
    Get or create the PostgresStore instance for long-term memory.
    Uses the DATABASE_URL environment variable for connection.
    
    Returns:
        PostgresStore instance configured with the database connection
        
    Raises:
        ValueError: If DATABASE_URL is not set or PostgresStore is not available
        Exception: If store initialization fails
    """
    global _store
    
    if not POSTGRES_STORE_AVAILABLE:
        raise ValueError(
            "PostgresStore is not available. "
            "Please install: pip install langgraph"
        )
    
    if _store is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required for memory store")
        
        # Convert SQLAlchemy-style URL to standard PostgreSQL URL if needed
        # PostgresStore expects postgresql:// format
        if database_url.startswith("postgresql+psycopg://"):
            # Remove the +psycopg part for PostgresStore
            database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        elif database_url.startswith("postgresql+psycopg2://"):
            database_url = database_url.replace("postgresql+psycopg2://", "postgresql://", 1)
        
        try:
            # PostgresStore requires a connection pool, not just a connection string
            # Create a connection pool using psycopg
            try:
                import psycopg
                from psycopg import pool
                # Create connection pool
                conn_pool = pool.ConnectionPool(
                    database_url,
                    min_size=1,
                    max_size=10,
                    open=False
                )
                conn_pool.open()
            except ImportError:
                # Fallback to psycopg2 if psycopg3 is not available
                import psycopg2.pool
                conn_pool = psycopg2.pool.SimpleConnectionPool(
                    1, 10, database_url
                )
            
            # Initialize PostgresStore with the connection pool
            _store = PostgresStore(conn=conn_pool)
            
            # Set up the store (creates necessary tables)
            _store.setup()
            
            logger.info("PostgresStore initialized successfully for long-term memory")
        except Exception as e:
            logger.error(f"Failed to initialize PostgresStore: {str(e)}")
            raise
    
    return _store


def get_user_namespace(user_id: str, org_id: str) -> tuple:
    """
    Create a namespace tuple for organizing memories by user and organization.
    
    Args:
        user_id: User ID
        org_id: Organization ID
        
    Returns:
        Tuple representing the namespace (org_id, user_id)
    """
    return (org_id, user_id)


def get_thread_namespace(thread_id: str, user_id: str, org_id: str) -> tuple:
    """
    Create a namespace tuple for thread-specific memories.
    
    Args:
        thread_id: Thread ID
        user_id: User ID
        org_id: Organization ID
        
    Returns:
        Tuple representing the namespace (org_id, user_id, thread_id)
    """
    return (org_id, user_id, "threads", thread_id)


def store_semantic_memory(
    user_id: str,
    org_id: str,
    key: str,
    memory_data: Dict[str, Any],
    namespace: Optional[tuple] = None
) -> None:
    """
    Store a semantic memory (fact or knowledge) for a user.
    
    Semantic memories are facts about the user, their preferences, or domain knowledge
    that should be remembered across all conversations.
    
    Args:
        user_id: User ID
        org_id: Organization ID
        key: Unique key for this memory
        memory_data: Dictionary containing the memory data
        namespace: Optional custom namespace (defaults to user namespace)
    """
    try:
        store = get_memory_store()
        ns = namespace or get_user_namespace(user_id, org_id)
        
        # Store the memory
        store.put(ns, key, memory_data)
        logger.info(f"Stored semantic memory: {key} for user {user_id} in org {org_id}")
    except Exception as e:
        logger.error(f"Error storing semantic memory: {str(e)}")
        raise


def get_semantic_memory(
    user_id: str,
    org_id: str,
    key: str,
    namespace: Optional[tuple] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific semantic memory by key.
    
    Args:
        user_id: User ID
        org_id: Organization ID
        key: Memory key
        namespace: Optional custom namespace
        
    Returns:
        Memory data dictionary or None if not found
    """
    if not POSTGRES_STORE_AVAILABLE:
        logger.debug("PostgresStore not available, skipping semantic memory retrieval")
        return None
    try:
        store = get_memory_store()
        ns = namespace or get_user_namespace(user_id, org_id)
        
        items = store.get(ns, key)
        if items:
            return items[0].value if hasattr(items[0], 'value') else items[0]
        return None
    except Exception as e:
        logger.error(f"Error retrieving semantic memory: {str(e)}")
        return None


def search_semantic_memories(
    user_id: str,
    org_id: str,
    query: Optional[str] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    namespace: Optional[tuple] = None
) -> List[Dict[str, Any]]:
    """
    Search semantic memories using semantic search or filtering.
    
    Args:
        user_id: User ID
        org_id: Organization ID
        query: Optional semantic search query (uses embeddings if available)
        filter_dict: Optional filter dictionary for exact matches
        limit: Maximum number of results
        namespace: Optional custom namespace
        
    Returns:
        List of memory dictionaries matching the search
    """
    if not POSTGRES_STORE_AVAILABLE:
        logger.debug("PostgresStore not available, skipping semantic memory search")
        return []
    try:
        store = get_memory_store()
        ns = namespace or get_user_namespace(user_id, org_id)
        
        # Search memories
        items = store.search(ns, filter=filter_dict, query=query, limit=limit)
        
        # Convert to list of dicts
        memories = []
        for item in items:
            if hasattr(item, 'value'):
                memories.append(item.value)
            else:
                memories.append(item)
        
        logger.info(f"Found {len(memories)} semantic memories for user {user_id}")
        return memories
    except Exception as e:
        logger.error(f"Error searching semantic memories: {str(e)}")
        return []


def store_episodic_memory(
    thread_id: str,
    user_id: str,
    org_id: str,
    memory_key: str,
    memory_data: Dict[str, Any]
) -> None:
    """
    Store an episodic memory (experience or event) for a specific thread.
    
    Episodic memories are experiences or events that happened in a conversation.
    These can be used for few-shot learning or remembering how tasks were accomplished.
    
    Args:
        thread_id: Thread ID
        user_id: User ID
        org_id: Organization ID
        memory_key: Unique key for this memory
        memory_data: Dictionary containing the memory data
    """
    if not POSTGRES_STORE_AVAILABLE:
        logger.debug("PostgresStore not available, skipping episodic memory storage")
        return
    try:
        store = get_memory_store()
        ns = get_thread_namespace(thread_id, user_id, org_id)
        
        # Add metadata
        memory_data["thread_id"] = thread_id
        memory_data["user_id"] = user_id
        memory_data["org_id"] = org_id
        
        store.put(ns, memory_key, memory_data)
        logger.info(f"Stored episodic memory: {memory_key} for thread {thread_id}")
    except Exception as e:
        logger.error(f"Error storing episodic memory: {str(e)}")
        raise


def get_episodic_memories(
    thread_id: str,
    user_id: str,
    org_id: str,
    query: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve episodic memories for a thread, optionally filtered by query.
    
    Args:
        thread_id: Thread ID
        user_id: User ID
        org_id: Organization ID
        query: Optional semantic search query
        limit: Maximum number of results
        
    Returns:
        List of episodic memory dictionaries
    """
    if not POSTGRES_STORE_AVAILABLE:
        logger.debug("PostgresStore not available, skipping episodic memory retrieval")
        return []
    try:
        store = get_memory_store()
        ns = get_thread_namespace(thread_id, user_id, org_id)
        
        items = store.search(ns, query=query, limit=limit)
        
        memories = []
        for item in items:
            if hasattr(item, 'value'):
                memories.append(item.value)
            else:
                memories.append(item)
        
        return memories
    except Exception as e:
        logger.error(f"Error retrieving episodic memories: {str(e)}")
        return []


def store_user_profile(
    user_id: str,
    org_id: str,
    profile_data: Dict[str, Any]
) -> None:
    """
    Store or update a user profile (semantic memory as a single document).
    
    A profile is a continuously updated document containing well-scoped information
    about a user, their preferences, and other facts.
    
    Args:
        user_id: User ID
        org_id: Organization ID
        profile_data: Dictionary containing profile information
    """
    try:
        store = get_memory_store()
        ns = get_user_namespace(user_id, org_id)
        
        # Get existing profile if it exists
        existing = store.get(ns, "user_profile")
        if existing:
            # Merge with existing profile (update)
            existing_data = existing[0].value if hasattr(existing[0], 'value') else existing[0]
            # Merge dictionaries (new data takes precedence)
            merged_profile = {**existing_data, **profile_data}
            store.put(ns, "user_profile", merged_profile)
            logger.info(f"Updated user profile for user {user_id}")
        else:
            # Create new profile
            store.put(ns, "user_profile", profile_data)
            logger.info(f"Created user profile for user {user_id}")
    except Exception as e:
        logger.error(f"Error storing user profile: {str(e)}")
        raise


def get_user_profile(
    user_id: str,
    org_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the user profile.
    
    Args:
        user_id: User ID
        org_id: Organization ID
        
    Returns:
        User profile dictionary or None if not found
    """
    if not POSTGRES_STORE_AVAILABLE:
        logger.debug("PostgresStore not available, skipping user profile retrieval")
        return None
    try:
        store = get_memory_store()
        ns = get_user_namespace(user_id, org_id)
        
        items = store.get(ns, "user_profile")
        if items:
            return items[0].value if hasattr(items[0], 'value') else items[0]
        return None
    except Exception as e:
        logger.error(f"Error retrieving user profile: {str(e)}")
        return None


def delete_memory(
    user_id: str,
    org_id: str,
    key: str,
    namespace: Optional[tuple] = None
) -> bool:
    """
    Delete a memory by key.
    
    Args:
        user_id: User ID
        org_id: Organization ID
        key: Memory key to delete
        namespace: Optional custom namespace
        
    Returns:
        True if deleted, False otherwise
    """
    try:
        store = get_memory_store()
        ns = namespace or get_user_namespace(user_id, org_id)
        
        # PostgresStore delete method
        store.delete(ns, [key])
        logger.info(f"Deleted memory: {key} for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        return False

