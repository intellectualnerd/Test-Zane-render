from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.vector_db import get_qa_chain
from app.utils.websocket_manager import websocket_manager, WebSocketMessage
from app.utils.websocket_models import (
    ChatMessageRequest, 
    ChatMessageResponse, 
    WebSocketMessageData, 
    MessageType,
    AIResponseData,
    SystemMessageData,
    UserStatus,
    ChatSessionInfo,
    ChatStatsResponse
)
import json
from app.api.auth import get_current_user
from app.utils.models import User, ChatThread, ChatMessage
from app.database import get_db, SessionLocal
from app.tools import build_org_lineage_tool, build_org_query_history_tool, build_org_pr_repo_tool, build_org_code_suggestion_tool, build_org_jira_tool
from app.tools.pr_repo import fetch_pr_analyses_for_org
from app.vector_db import CHAT_LLM
from app.services.impact_analysis import fetch_queries
# PostgresStore long-term memory imports (commented out - not currently in use)
# from app.utils.memory_store import (
#     get_user_profile,
#     search_semantic_memories,
#     store_semantic_memory,
#     store_episodic_memory,
#     get_episodic_memories
# )
import logging
import uuid
import asyncio
from datetime import datetime
from langchain.agents import initialize_agent, AgentType
from sqlalchemy.orm import Session
from sqlalchemy import desc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Configuration for conversation history limits
# These can be adjusted based on LLM context window and requirements
# Note: Most modern LLMs have context windows of 8K-128K tokens
# Adjust these values based on your LLM's context window and desired behavior
MAX_CONTEXT_MESSAGES = 20  # Maximum number of messages to include in context (default: 20)
MAX_CONTEXT_TOKENS = 8000  # Approximate max tokens for context (rough estimate: 1 token ≈ 4 chars)
CONTEXT_MESSAGE_ESTIMATE = 200  # Estimated tokens per message (rough estimate, not used but kept for reference)

def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count (1 token ≈ 4 characters).
    This is a simple heuristic; for accurate counts, use tiktoken or similar.
    """
    return len(text) // 4

def load_conversation_history(
    thread_id: str,
    user_id: str,
    org_id: str,
    db: Session,
    max_messages: Optional[int] = None,
    max_tokens: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Load conversation history from database for a given thread.
    Returns a list of messages in format [{"role": "user|assistant", "content": "..."}]
    
    Args:
        thread_id: Thread ID to load history for
        user_id: User ID (for security validation)
        org_id: Organization ID (for security validation)
        db: Database session
        max_messages: Maximum number of messages to return (default: MAX_CONTEXT_MESSAGES)
        max_tokens: Maximum approximate tokens to include (default: MAX_CONTEXT_TOKENS)
        
    Returns:
        List of message dicts with role and content, ordered chronologically
    """
    if not thread_id:
        return []
    
    try:
        thread_uuid = uuid.UUID(thread_id)
    except ValueError:
        logger.warning(f"Invalid thread_id format: {thread_id}")
        return []
    
    # Validate thread belongs to user and org
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_uuid,
        ChatThread.user_id == uuid.UUID(user_id),
        ChatThread.org_id == uuid.UUID(org_id),
        ChatThread.is_active == True
    ).first()
    
    if not thread:
        logger.warning(f"Thread {thread_id} not found or access denied for user {user_id}")
        return []
    
    # Load messages ordered by creation time
    messages = db.query(ChatMessage).filter(
        ChatMessage.thread_id == thread_uuid
    ).order_by(ChatMessage.created_at).all()
    
    if not messages:
        return []
    
    # Convert to list of dicts
    history = []
    for msg in messages:
        history.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Apply limits
    max_msgs = max_messages or MAX_CONTEXT_MESSAGES
    max_toks = max_tokens or MAX_CONTEXT_TOKENS
    
    # First, limit by message count (take most recent N messages)
    if len(history) > max_msgs:
        history = history[-max_msgs:]
    
    # Then, limit by token count (remove oldest messages if needed)
    total_tokens = 0
    if max_toks:
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in history)
        while total_tokens > max_toks and len(history) > 1:
            # Remove oldest message
            removed = history.pop(0)
            total_tokens -= estimate_tokens(removed["content"])
    else:
        # Calculate tokens even if not limiting by them (for logging)
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in history)
    
    logger.info(f"Loaded {len(history)} messages from thread {thread_id} (estimated {total_tokens} tokens)")
    return history

def format_conversation_context(history: List[Dict[str, str]]) -> str:
    """
    Format conversation history into a context string for the LLM.
    Uses a format compatible with most chat models (User/Assistant format).
    
    Args:
        history: List of message dicts with role and content
        
    Returns:
        Formatted context string ready to be included in LLM prompt
    """
    if not history:
        return ""
    
    context_parts = []
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        # Format: "User: message content" or "Assistant: message content"
        context_parts.append(f"{role_label}: {msg['content']}")
    
    return "\n".join(context_parts)

# Helper functions for chat history
def get_or_create_thread(thread_id: Optional[str], user_id: str, org_id: str, db: Session) -> ChatThread:
    """
    Get existing thread or create a new one if thread_id is None.
    """
    if thread_id:
        try:
            thread_uuid = uuid.UUID(thread_id)
            thread = db.query(ChatThread).filter(
                ChatThread.id == thread_uuid,
                ChatThread.user_id == uuid.UUID(user_id),
                ChatThread.org_id == uuid.UUID(org_id),
                ChatThread.is_active == True
            ).first()
            if thread:
                return thread
        except ValueError:
            pass
    
    # Create new thread
    new_thread = ChatThread(
        org_id=uuid.UUID(org_id),
        user_id=uuid.UUID(user_id),
        title="New Chat"
    )
    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)
    return new_thread

def save_user_message(thread_id: str, user_id: str, org_id: str, content: str, db: Session) -> ChatMessage:
    """Save a user message to the database."""
    message = ChatMessage(
        thread_id=uuid.UUID(thread_id),
        org_id=uuid.UUID(org_id),
        user_id=uuid.UUID(user_id),
        role="user",
        content=content
    )
    db.add(message)
    
    # Update thread's last_message_at
    thread = db.query(ChatThread).filter(ChatThread.id == uuid.UUID(thread_id)).first()
    if thread:
        thread.last_message_at = datetime.utcnow()
        # Auto-generate title from first message if title is still "New Chat"
        if thread.title == "New Chat" and content:
            # Generate title from first 50 chars of message
            title = content[:50].strip()
            if len(content) > 50:
                title += "..."
            thread.title = title
    
    db.commit()
    db.refresh(message)
    return message

def save_assistant_message(
    thread_id: str, 
    user_id: str, 
    org_id: str, 
    content: str, 
    metadata: Optional[Dict[str, Any]], 
    db: Session
) -> ChatMessage:
    """Save an assistant message to the database."""
    message = ChatMessage(
        thread_id=uuid.UUID(thread_id),
        org_id=uuid.UUID(org_id),
        user_id=uuid.UUID(user_id),
        role="assistant",
        content=content,
        message_metadata=metadata
    )
    db.add(message)
    
    # Update thread's last_message_at
    thread = db.query(ChatThread).filter(ChatThread.id == uuid.UUID(thread_id)).first()
    if thread:
        thread.last_message_at = datetime.utcnow()
    
    db.commit()
    db.refresh(message)
    return message

class ChatMessageData(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    k: Optional[int] = 5  # Number of documents to retrieve for context
    conversation_history: Optional[List[ChatMessageData]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    conversation_id: Optional[str] = None
    impacted_query_ids: Optional[List[str]] = []
    impacted_queries: Optional[List[Dict[str, Any]]] = []
    pr_repo_data: Optional[Dict[str, Any]] = None
    code_suggestions: Optional[Dict[str, Any]] = None
    jira_ticket: Optional[Dict[str, Any]] = None

class ChatConversation(BaseModel):
    conversation_id: str
    org_id: str
    messages: List[ChatMessageData]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@router.post("/query", response_model=ChatResponse)
async def chat_with_llm(
    request: ChatRequest, 
    current_user: User = Depends(get_current_user),
    thread_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Chat with LLM using vector database context for a specific organization.
    Optionally saves chat history if thread_id is provided.
    
    Args:
        request: ChatRequest containing message and optional parameters
        thread_id: Optional thread ID for saving chat history (query parameter)
        current_user: Authenticated user (from token)
        db: Database session
        
    Returns:
        ChatResponse with LLM response and source documents
    """
    try:
        # Resolve organization strictly from authenticated user
        resolved_org_id = str(current_user.org_id)

        logger.info(f"Chat request for org_id: {resolved_org_id}, message: {request.message[:100]}...")
        
        # Get or create thread if thread_id provided
        db_thread = None
        conversation_history = []
        if thread_id:
            try:
                db_thread = get_or_create_thread(thread_id, str(current_user.id), str(current_user.org_id), db)
                thread_id = str(db_thread.id)
                # Save user message
                save_user_message(thread_id, str(current_user.id), str(current_user.org_id), request.message, db)
                
                # Load conversation history from database (excluding the message we just saved)
                # This provides long-term memory across sessions
                conversation_history = load_conversation_history(
                    thread_id=thread_id,
                    user_id=str(current_user.id),
                    org_id=str(current_user.org_id),
                    db=db
                )
                # Remove the last message (the one we just saved) from history for context
                # We'll add it back in the query, but we want previous context
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history = conversation_history[:-1]
            except Exception as e:
                logger.error(f"Error saving user message or loading history: {str(e)}")
        
        # Use provided conversation_history if no database history, otherwise use database history
        # Database history takes precedence as it's the source of truth
        if not conversation_history and request.conversation_history:
            conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        
        # Prepare the query with conversation context
        query = request.message
        if conversation_history:
            context = format_conversation_context(conversation_history)
            if context:
                query = f"Previous conversation context:\n{context}\n\nCurrent question: {request.message}"
                logger.info(f"Added conversation context: {len(conversation_history)} messages")
        
        # LLM classification: decide whether to use tools (lineage/impact) or respond conversationally (other)
        classify_prompt = (
            "You are a classifier. Decide if the user's message requires using specialized tools for: "
            "data lineage (extract_lineage), query impact analysis (query_history_search), "
            "PR/Repo analysis (pr_repo_analysis), code suggestions (code_suggestion), or Jira ticket creation (create_jira_ticket).\n"
            "Respond with exactly one word: lineage, impact, pr, code, jira, or other.\n\n"
            f"Message: {request.message}"
        )
        if not CHAT_LLM:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured for chatbot")
        
        # Log which LLM is being used
        actual_model = "unknown"
        if CHAT_LLM:
            if hasattr(CHAT_LLM, 'model_name'):
                actual_model = CHAT_LLM.model_name
            elif hasattr(CHAT_LLM, 'model'):
                actual_model = CHAT_LLM.model
            elif hasattr(CHAT_LLM, '_model_name'):
                actual_model = CHAT_LLM._model_name
            llm_class = type(CHAT_LLM).__name__
            logger.info(f"Using CHAT_LLM ({llm_class}) with model: {actual_model} for message classification")
        
        classification = CHAT_LLM.invoke(classify_prompt)
        classification_label = (getattr(classification, "content", str(classification)) or "other").strip().lower()

        if classification_label not in {"lineage", "impact", "pr", "code", "jira"}:
            # Conversational reply without tools
            persona_prompt = (
                "SYSTEM: You are Zane AI, a helpful assistant for data lineage and change-impact analysis.\n"
                "- Be concise.\n"
                "- Do NOT invent lineage or impacts without analysis.\n"
                "- If the user hasn't asked for lineage/impact, introduce capabilities briefly and ask a clarifying question.\n\n"
                f"USER: {request.message}\n"
                "ASSISTANT:"
            )
            llm_reply = CHAT_LLM.invoke(persona_prompt)
            reply_text = getattr(llm_reply, "content", str(llm_reply))
            
            # Save assistant message if thread_id provided
            if thread_id:
                try:
                    save_assistant_message(
                        thread_id,
                        str(current_user.id),
                        str(current_user.org_id),
                        reply_text,
                        {"processing_time": 0},
                        db
                    )
                except Exception as e:
                    logger.error(f"Error saving assistant message: {str(e)}")
            
            return ChatResponse(
                response=reply_text,
                sources=[],
                conversation_id=thread_id,  # Return thread_id for compatibility
                impacted_query_ids=[],
                impacted_queries=[],
                pr_repo_data=None,
                code_suggestions=None,
                jira_ticket=None,
            )

        # Build org-aware tools and delegate tool selection to the LLM agent
        lineage_tool = build_org_lineage_tool(org_id=resolved_org_id, k=request.k or 5)
        query_history_tool = build_org_query_history_tool(org_id=resolved_org_id, max_iters=5)
        pr_repo_tool = build_org_pr_repo_tool(org_id=resolved_org_id, default_limit=10)
        code_suggestion_tool = build_org_code_suggestion_tool(org_id=resolved_org_id)
        jira_tool = build_org_jira_tool(org_id=resolved_org_id, user_id=str(current_user.id))

        # Log which LLM is being used for the agent
        actual_model = "unknown"
        if CHAT_LLM:
            if hasattr(CHAT_LLM, 'model_name'):
                actual_model = CHAT_LLM.model_name
            elif hasattr(CHAT_LLM, 'model'):
                actual_model = CHAT_LLM.model
            elif hasattr(CHAT_LLM, '_model_name'):
                actual_model = CHAT_LLM._model_name
            llm_class = type(CHAT_LLM).__name__
            logger.info(f"Initializing agent with CHAT_LLM ({llm_class}) using model: {actual_model}")
        
        agent = initialize_agent(
            tools=[lineage_tool, query_history_tool, pr_repo_tool, code_suggestion_tool, jira_tool],
            llm=CHAT_LLM,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,  # Limit iterations to prevent infinite loops
            max_execution_time=120,  # 2 minutes max execution time
        )

        # Strong guidance to the agent on tool selection and output format
        # Include conversation awareness if we have history
        conversation_note = ""
        if conversation_history:
            conversation_note = (
                "\nCONVERSATION CONTEXT:\n"
                "- You are continuing an ongoing conversation. Use the previous conversation context to understand references, "
                "follow-up questions, and maintain continuity.\n"
                "- If the user refers to something mentioned earlier (e.g., 'that table', 'the previous query'), "
                "use the conversation context to understand what they mean.\n"
            )
        
        guidance = (
            "SYSTEM ROLE: You are Zane AI, an assistant that helps analyze data lineage and change impacts.\n"
            "BEHAVIOR:\n"
            "- Be concise and helpful.\n"
            f"{conversation_note}"
            "- CRITICAL: DO NOT call the same tool multiple times with the same input. If a tool returns results, use those results and move forward. DO NOT loop.\n"
            "- CRITICAL: After a tool returns results, you MUST include the COMPLETE tool output in your final answer. DO NOT summarize or say 'listed above' - show the actual results.\n"
            "- CRITICAL: When query_history_search tool returns impacted queries, you MUST copy and display ALL the queries from the tool output in your response. Include the Query IDs and SQL previews exactly as shown in the tool output.\n"
            "- CRITICAL: DO NOT say 'queries are listed above' or 'see above' - the user cannot see above. Always include the full tool output in your response.\n"
            "- WORKFLOW FOR CODE SUGGESTIONS: If the user asks to 'suggest code changes', 'suggest fixes', or 'code changes needed' for a PR, you MUST use the code_suggestion tool directly with the repo and PR number. The code_suggestion tool will automatically fetch PR analysis if needed. DO NOT call pr_repo_analysis first - it's not necessary.\n"
            "- If the user greets you (e.g., 'hi', 'hello'), respond with a short intro of who you are and how you can help (lineage Q&A, query impact analysis, PR analysis, code suggestions, and Jira ticket creation).\n"
            "- If the question is about schema/column changes or 'impacted queries', you MUST use the query_history_search tool ONCE, then copy the complete tool output into your final answer.\n"
            "- When reporting impacted queries, include the complete numbered list with Query IDs and SQL previews from the tool output.\n"
            "- If it's a pure lineage question, use the extract_lineage tool ONCE, then include the complete tool output in your answer.\n"
            "- If the question is ONLY about viewing PR analysis or repository information (not asking for code suggestions), use the pr_repo_analysis tool.\n"
            "- If the question asks for code suggestions, fixes, or changes needed for a PR, use the code_suggestion tool directly (it handles PR analysis internally).\n"
            "- If the question asks to create a Jira ticket, use the create_jira_ticket tool.\n"
            "- ALWAYS honor prior conversation context (including repo/PR selections, confirmations, and disambiguations) when invoking any tool. Do not ask the user to repeat details that were already confirmed.\n"
            "- IMPORTANT: Follow the ReAct format strictly. After each tool call, provide your final answer using 'Final Answer:' and include the COMPLETE tool output, not a summary."
        )
        # Nudge the agent to preferred tool if classification is specific
        preferred_hint = (
            "\nPREFERRED_TOOL: query_history_search\n" if classification_label == "impact" else (
                "\nPREFERRED_TOOL: extract_lineage\n" if classification_label == "lineage" else (
                    "\nPREFERRED_TOOL: pr_repo_analysis\n" if classification_label == "pr" else (
                        "\nPREFERRED_TOOL: code_suggestion\n" if classification_label == "code" else (
                            "\nPREFERRED_TOOL: create_jira_ticket\n" if classification_label == "jira" else ""
                        )
                    )
                )
            )
        )
        agent_query = f"{guidance}{preferred_hint}\nUser question: {query}"

        try:
            agent_result = agent.invoke(agent_query)
            # LangChain agents often return dicts with `output`; fallback to str
            if isinstance(agent_result, dict) and "output" in agent_result:
                response_text = agent_result.get("output", "")
            else:
                response_text = str(agent_result)
        except ValueError as e:
            # Handle parsing errors - extract the actual response from the error if possible
            error_msg = str(e)
            if "Could not parse LLM output" in error_msg:
                # Try to extract the response from the error message
                # The error format is: "Could not parse LLM output: `...response...`"
                import re
                # Find text after "Could not parse LLM output: `" and extract until the last backtick before "For troubleshooting"
                match = re.search(r"Could not parse LLM output: `(.*?)(?:`\s*For troubleshooting|$)", error_msg, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
                    # Clean up common prefixes that the agent might add
                    response_text = re.sub(r"^I now know the final answer\.\s*", "", response_text, flags=re.IGNORECASE)
                    logger.warning(f"Agent parsing error handled, extracted response: {response_text[:100]}...")
                else:
                    response_text = "I encountered an issue formatting my response. Please try rephrasing your question."
                    logger.error(f"Agent parsing error: {error_msg}")
            else:
                raise
        except Exception as e:
            # Handle iteration/time limit errors and other agent errors
            error_msg = str(e)
            logger.error(f"Agent execution error: {error_msg}")
            
            # Check for iteration/time limit errors
            if "iteration limit" in error_msg.lower() or "time limit" in error_msg.lower() or "stopped due to" in error_msg.lower():
                # Try to extract any partial response from the error
                import re
                # Look for any response content in the error
                response_match = re.search(r'(?:output|response|answer)[:\s]+(.*?)(?:\n|$)', error_msg, re.DOTALL | re.IGNORECASE)
                if response_match:
                    partial_response = response_match.group(1).strip()
                    if len(partial_response) > 50:  # Only use if substantial
                        response_text = f"{partial_response}\n\n[Note: Response was truncated due to processing limits. The information above should address your query.]"
                    else:
                        response_text = "I've reached the processing limit while analyzing your query. The tool has retrieved the impacted queries, but I wasn't able to format the complete response. Here's what was found:\n\nPlease try rephrasing your question or breaking it into smaller parts."
                else:
                    # Check if we can extract query IDs from the error message
                    query_ids = re.findall(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", error_msg, re.I)
                    if query_ids:
                        response_text = f"I found {len(query_ids)} impacted query(s) but encountered a processing limit. Query IDs: {', '.join(query_ids[:5])}"
                        if len(query_ids) > 5:
                            response_text += f" (and {len(query_ids) - 5} more)"
                    else:
                        response_text = "I've reached the processing limit while analyzing your query. Please try rephrasing your question or breaking it into smaller parts."
            else:
                # For other errors, provide a generic message
                response_text = "I encountered an error while processing your request. Please try rephrasing your question."
                logger.error(f"Unexpected agent error: {error_msg}")

        # Best-effort: extract query IDs from the response text and fetch full queries
        impacted_query_ids: List[str] = []
        impacted_queries: List[Dict[str, Any]] = []
        try:
            import re as _re
            # Match UUID-like ids commonly used in results
            impacted_query_ids = list(dict.fromkeys(_re.findall(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", response_text, flags=_re.I)))
            if impacted_query_ids:
                impacted_queries = fetch_queries(impacted_query_ids) or []
        except Exception:
            impacted_query_ids = []
            impacted_queries = []

        # Best-effort: extract tool payloads if present (DATA:\n{...})
        pr_repo_data: Optional[Dict[str, Any]] = None
        code_suggestions: Optional[Dict[str, Any]] = None
        jira_ticket: Optional[Dict[str, Any]] = None
        try:
            import re as _re2, json as _json2
            m = _re2.search(r"DATA:\n(\{[\s\S]*\})", response_text)
            if m:
                data = _json2.loads(m.group(1))
                # Check what type of data it is
                if "suggestions_by_file" in data:
                    code_suggestions = data
                elif "ticket" in data or "jira_issue" in data:
                    jira_ticket = data
                else:
                    pr_repo_data = data
        except Exception:
            pass
        
        # Save assistant message if thread_id provided
        if thread_id:
            try:
                message_metadata = {
                    "impacted_query_ids": impacted_query_ids,
                    "impacted_queries": impacted_queries,
                    "pr_repo_data": pr_repo_data,
                    "code_suggestions": code_suggestions,
                    "jira_ticket": jira_ticket,
                }
                save_assistant_message(
                    thread_id,
                    str(current_user.id),
                    str(current_user.org_id),
                    response_text,
                    message_metadata,
                    db
                )
                
                # Episodic memory storage (PostgresStore) - commented out for now
                # Can be enabled when PostgresStore is properly configured
                # try:
                #     if thread_id:
                #         episodic_memory = {
                #             "user_message": request.message,
                #             "assistant_response": response_text[:500],
                #             "timestamp": datetime.utcnow().isoformat(),
                #             "classification": classification_label,
                #             "has_tool_results": bool(impacted_queries or pr_repo_data or code_suggestions or jira_ticket)
                #         }
                #         memory_key = f"episode_{datetime.utcnow().timestamp()}"
                #         store_episodic_memory(
                #             thread_id=thread_id,
                #             user_id=str(current_user.id),
                #             org_id=resolved_org_id,
                #             memory_key=memory_key,
                #             memory_data=episodic_memory
                #         )
                # except Exception as e:
                #     logger.warning(f"Error storing episodic memory: {str(e)} (non-critical)")
            except Exception as e:
                logger.error(f"Error saving assistant message: {str(e)}")
        
        return ChatResponse(
            response=response_text,
            sources=[],  # Tool outputs include their own context; no structured source docs here
            conversation_id=thread_id,  # Return thread_id for compatibility
            impacted_query_ids=impacted_query_ids,
            impacted_queries=impacted_queries,
            pr_repo_data=pr_repo_data,
            code_suggestions=code_suggestions,
            jira_ticket=jira_ticket,
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# Chat Thread Management Models
class ChatThreadCreate(BaseModel):
    title: Optional[str] = None

class ChatThreadResponse(BaseModel):
    id: str
    org_id: str
    user_id: str
    title: Optional[str]
    is_active: bool
    created_at: str
    updated_at: Optional[str]
    last_message_at: Optional[str]
    message_count: int = 0

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: str

    class Config:
        from_attributes = True

class ChatThreadWithMessages(BaseModel):
    id: str
    org_id: str
    user_id: str
    title: Optional[str]
    is_active: bool
    created_at: str
    updated_at: Optional[str]
    last_message_at: Optional[str]
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

# Chat Thread Management Endpoints
@router.post("/threads", response_model=ChatThreadResponse, status_code=201)
async def create_chat_thread(
    thread_data: ChatThreadCreate = ChatThreadCreate(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new chat thread for the authenticated user.
    
    Returns:
        ChatThreadResponse with thread details
    """
    try:
        new_thread = ChatThread(
            org_id=current_user.org_id,
            user_id=current_user.id,
            title=thread_data.title or "New Chat"
        )
        db.add(new_thread)
        db.commit()
        db.refresh(new_thread)
        
        logger.info(f"Created chat thread {new_thread.id} for user {current_user.id}")
        return ChatThreadResponse(
            id=str(new_thread.id),
            org_id=str(new_thread.org_id),
            user_id=str(new_thread.user_id),
            title=new_thread.title,
            is_active=new_thread.is_active,
            created_at=new_thread.created_at.isoformat(),
            updated_at=new_thread.updated_at.isoformat() if new_thread.updated_at else None,
            last_message_at=new_thread.last_message_at.isoformat() if new_thread.last_message_at else None,
            message_count=0
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating chat thread: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating chat thread: {str(e)}")

@router.get("/threads", response_model=List[ChatThreadResponse])
async def get_user_chat_threads(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all chat threads for the authenticated user.
    Returns threads ordered by last_message_at (most recent first).
    """
    try:
        threads = db.query(ChatThread).filter(
            ChatThread.user_id == current_user.id,
            ChatThread.org_id == current_user.org_id,
            ChatThread.is_active == True
        ).order_by(desc(ChatThread.last_message_at), desc(ChatThread.created_at)).all()
        
        result = []
        for thread in threads:
            message_count = db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread.id
            ).count()
            
            result.append(ChatThreadResponse(
                id=str(thread.id),
                org_id=str(thread.org_id),
                user_id=str(thread.user_id),
                title=thread.title,
                is_active=thread.is_active,
                created_at=thread.created_at.isoformat(),
                updated_at=thread.updated_at.isoformat() if thread.updated_at else None,
                last_message_at=thread.last_message_at.isoformat() if thread.last_message_at else None,
                message_count=message_count
            ))
        
        return result
    except Exception as e:
        logger.error(f"Error getting chat threads: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting chat threads: {str(e)}")

@router.get("/threads/{thread_id}", response_model=ChatThreadWithMessages)
async def get_chat_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific chat thread with all its messages.
    """
    try:
        thread_uuid = uuid.UUID(thread_id)
        thread = db.query(ChatThread).filter(
            ChatThread.id == thread_uuid,
            ChatThread.user_id == current_user.id,
            ChatThread.org_id == current_user.org_id,
            ChatThread.is_active == True
        ).first()
        
        if not thread:
            raise HTTPException(status_code=404, detail="Chat thread not found")
        
        messages = db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread.id
        ).order_by(ChatMessage.created_at).all()
        
        return ChatThreadWithMessages(
            id=str(thread.id),
            org_id=str(thread.org_id),
            user_id=str(thread.user_id),
            title=thread.title,
            is_active=thread.is_active,
            created_at=thread.created_at.isoformat(),
            updated_at=thread.updated_at.isoformat() if thread.updated_at else None,
            last_message_at=thread.last_message_at.isoformat() if thread.last_message_at else None,
            messages=[
                ChatMessageResponse(
                    id=str(msg.id),
                    thread_id=str(msg.thread_id),
                    role=msg.role,
                    content=msg.content,
                    metadata=msg.message_metadata,
                    created_at=msg.created_at.isoformat()
                )
                for msg in messages
            ]
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid thread ID format")
    except Exception as e:
        logger.error(f"Error getting chat thread: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting chat thread: {str(e)}")

@router.put("/threads/{thread_id}/title")
async def update_thread_title(
    thread_id: str,
    title: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update the title of a chat thread.
    """
    try:
        thread_uuid = uuid.UUID(thread_id)
        thread = db.query(ChatThread).filter(
            ChatThread.id == thread_uuid,
            ChatThread.user_id == current_user.id,
            ChatThread.org_id == current_user.org_id,
            ChatThread.is_active == True
        ).first()
        
        if not thread:
            raise HTTPException(status_code=404, detail="Chat thread not found")
        
        thread.title = title
        db.commit()
        db.refresh(thread)
        
        return {"message": "Thread title updated", "title": thread.title}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid thread ID format")
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating thread title: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating thread title: {str(e)}")

@router.delete("/threads/{thread_id}")
async def delete_chat_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Soft delete a chat thread (sets is_active to False).
    """
    try:
        thread_uuid = uuid.UUID(thread_id)
        thread = db.query(ChatThread).filter(
            ChatThread.id == thread_uuid,
            ChatThread.user_id == current_user.id,
            ChatThread.org_id == current_user.org_id,
            ChatThread.is_active == True
        ).first()
        
        if not thread:
            raise HTTPException(status_code=404, detail="Chat thread not found")
        
        thread.is_active = False
        db.commit()
        
        return {"message": "Chat thread deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid thread ID format")
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting chat thread: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting chat thread: {str(e)}")

@router.post("/conversation", response_model=ChatConversation)
async def create_conversation(org_id: str):
    """
    Create a new conversation for an organization.
    DEPRECATED: Use /threads endpoint instead.
    
    Args:
        org_id: Organization ID
        
    Returns:
        ChatConversation object
    """
    try:
        if not org_id:
            raise HTTPException(status_code=400, detail="org_id is required")
        
        # Generate a simple conversation ID (in production, use proper UUID)
        import uuid
        conversation_id = str(uuid.uuid4())
        
        conversation = ChatConversation(
            conversation_id=conversation_id,
            org_id=org_id,
            messages=[],
            created_at=None,  # Could add timestamp
            updated_at=None
        )
        
        logger.info(f"Created conversation {conversation_id} for org {org_id}")
        return conversation
        
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")

@router.get("/health")
async def chat_health():
    """
    Health check for chat service.
    
    Returns:
        Status of chat service
    """
    return {"status": "healthy", "service": "chat"}

@router.get("/orgs/{org_id}/test")
async def test_org_vector_store(org_id: str):
    """
    Test if vector store is available for an organization.
    
    Args:
        org_id: Organization ID
        
    Returns:
        Test result
    """
    try:
        from app.vector_db import get_org_vector_store
        
        # Try to get the vector store
        vector_store = get_org_vector_store(org_id)
        
        # Try a simple query to test
        test_query = "test query"
        results = vector_store.similarity_search(test_query, k=1)
        
        return {
            "org_id": org_id,
            "status": "available",
            "test_results_count": len(results),
            "message": "Vector store is accessible"
        }
        
    except Exception as e:
        logger.error(f"Error testing vector store for org {org_id}: {str(e)}")
        return {
            "org_id": org_id,
            "status": "error",
            "message": f"Vector store error: {str(e)}"
        }


# WebSocket endpoint for real-time chat
@router.websocket("/ws/{org_id}/{user_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket, 
    org_id: str, 
    user_id: str,
    session_id: Optional[str] = Query(None),
    user_name: Optional[str] = Query(None),
    token: Optional[str] = Query(None),  # Authentication token
    thread_id: Optional[str] = Query(None)  # Chat thread ID for history
):
    """
    WebSocket endpoint for real-time chat functionality.
    
    Args:
        websocket: WebSocket connection
        org_id: Organization ID (can be overridden by authenticated user's org_id)
        user_id: User ID (should match authenticated user)
        session_id: Optional session ID (will generate if not provided)
        user_name: Optional user display name
        token: JWT authentication token (required for authenticated access and chat history)
        thread_id: Optional chat thread ID - if provided, messages will be saved to this thread.
                  If not provided and user is authenticated, a new thread will be created.
    
    Note: If token is provided, user will be authenticated and org_id will be resolved from user.
          Chat history is only saved when token is provided.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logger.info(f"WebSocket connection attempt for org {org_id}, user {user_id}, session {session_id}")
    
    # Authenticate user if token is provided
    current_user = None
    resolved_org_id = org_id
    
    if token:
        try:
            from app.database import SessionLocal
            from app.utils.auth_deps import get_user_from_token
            db = SessionLocal()
            try:
                current_user = get_user_from_token(token, db)
                resolved_org_id = str(current_user.org_id)
                user_id = str(current_user.id)  # Use authenticated user's ID
                user_name = user_name or current_user.username
                logger.info(f"WebSocket authenticated: user_id={user_id}, org_id={resolved_org_id}")
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {str(e)}")
            await websocket.close(code=4001, reason="Authentication failed")
            return
    
    try:
        # Connect to WebSocket manager (use resolved org_id)
        session = await websocket_manager.connect(websocket, session_id, resolved_org_id, user_id)
        
        # Send welcome message
        welcome_message = WebSocketMessage(
            type="system_message",
            data={
                "message": f"Welcome to QueryGuard chat! Session {session_id} established.",
                "session_id": session_id,
                "user_name": user_name,
                "status": "connected"
            },
            sender_id="system",
            room_id=resolved_org_id
        )
        await websocket_manager.send_message(session_id, welcome_message)
        
        # Notify other users in the organization
        user_join_message = WebSocketMessage(
            type="user_status",
            data={
                "message": f"User {user_name or user_id} joined the chat",
                "user_id": user_id,
                "user_name": user_name,
                "status": "online",
                "action": "joined"
            },
            sender_id=user_id,
            room_id=resolved_org_id
        )
        await websocket_manager.broadcast_to_org(resolved_org_id, user_join_message, exclude_session=session_id)
        
        # Main message loop
        while True:
            try:
                # Receive message from client
                raw_message = await websocket.receive_text()
                message_data = json.loads(raw_message)
                
                logger.info(f"Received message from {session_id}: {message_data.get('type', 'unknown')}")
                
                # Process different message types
                message_type = message_data.get("type", MessageType.CHAT_MESSAGE)
                
                if message_type == MessageType.CHAT_MESSAGE:
                    await handle_chat_message(session_id, resolved_org_id, user_id, user_name, message_data, current_user, thread_id)
                elif message_type == MessageType.TYPING_INDICATOR:
                    await handle_typing_indicator(session_id, resolved_org_id, user_id, user_name, message_data)
                elif message_type == "ping":
                    await handle_ping(session_id)
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from session {session_id}")
                error_message = WebSocketMessage(
                    type="error",
                    data={
                        "error_code": "INVALID_JSON",
                        "error_message": "Invalid message format. Please send valid JSON."
                    },
                    sender_id="system"
                )
                await websocket_manager.send_message(session_id, error_message)
            except Exception as e:
                logger.error(f"Error processing message from session {session_id}: {str(e)}")
                # Only try to send error message if session is still connected
                # Don't log error if connection is already closed (this is expected)
                if session_id in websocket_manager.active_connections:
                    try:
                        error_message = WebSocketMessage(
                            type="error",
                            data={
                                "error_code": "PROCESSING_ERROR",
                                "error_message": f"Error processing message: {str(e)}"
                            },
                            sender_id="system"
                        )
                        await websocket_manager.send_message(session_id, error_message)
                    except Exception as send_error:
                        # If sending error message fails, connection is likely closed
                        logger.debug(f"Could not send error message to session {session_id} (connection may be closed): {str(send_error)}")
                else:
                    logger.debug(f"Session {session_id} not in active connections, skipping error message")
                
    except Exception as e:
        logger.error(f"WebSocket connection error for session {session_id}: {str(e)}")
    finally:
        # Clean up connection
        await websocket_manager.disconnect(session_id)
        
        # Notify other users that this user left
        user_leave_message = WebSocketMessage(
            type="user_status",
            data={
                "message": f"User {user_name or user_id} left the chat",
                "user_id": user_id,
                "user_name": user_name,
                "status": "offline",
                "action": "left"
            },
            sender_id=user_id,
            room_id=resolved_org_id
        )
        await websocket_manager.broadcast_to_org(resolved_org_id, user_leave_message)
        
        logger.info(f"WebSocket cleanup completed for session {session_id}")

async def handle_chat_message(
    session_id: str, 
    org_id: str, 
    user_id: str, 
    user_name: Optional[str], 
    message_data: dict,
    current_user: Optional[User] = None,
    thread_id: Optional[str] = None
):
    """
    Handle incoming chat messages and generate AI responses using the full chat logic.
    This integrates the sophisticated chat endpoint logic (LLM agents, tools, classification) 
    with WebSocket real-time communication.
    
    Args:
        session_id: WebSocket session ID
        org_id: Organization ID (resolved from authenticated user if available)
        user_id: User ID
        user_name: User display name
        message_data: Message data from client
        current_user: Authenticated user object (if available)
        thread_id: Chat thread ID for saving history (if available)
    """
    try:
        content = message_data.get("content", "").strip()
        if not content:
            return
        
        # Get thread_id from message or use provided one
        message_thread_id = message_data.get("thread_id") or thread_id
        k = message_data.get("k", 5)  # Number of documents to retrieve
        
        # Get or create thread if user is authenticated
        db_thread = None
        conversation_history = []
        if current_user and message_thread_id:
            db = SessionLocal()
            try:
                db_thread = get_or_create_thread(message_thread_id, str(current_user.id), str(current_user.org_id), db)
                message_thread_id = str(db_thread.id)
                # Save user message
                save_user_message(message_thread_id, str(current_user.id), str(current_user.org_id), content, db)
                
                # Load conversation history from database (excluding the message we just saved)
                # This provides long-term memory across sessions
                conversation_history = load_conversation_history(
                    thread_id=message_thread_id,
                    user_id=str(current_user.id),
                    org_id=str(current_user.org_id),
                    db=db
                )
                # Remove the last message (the one we just saved) from history for context
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history = conversation_history[:-1]
            except Exception as e:
                logger.error(f"Error saving user message or loading history: {str(e)}")
            finally:
                db.close()
        elif current_user and not message_thread_id:
            # Create new thread for authenticated user
            db = SessionLocal()
            try:
                db_thread = get_or_create_thread(None, str(current_user.id), str(current_user.org_id), db)
                message_thread_id = str(db_thread.id)
                # Save user message
                save_user_message(message_thread_id, str(current_user.id), str(current_user.org_id), content, db)
                # Notify client about new thread_id
                thread_notification = WebSocketMessage(
                    type="system_message",
                    data={
                        "message": f"New chat thread created: {message_thread_id}",
                        "thread_id": message_thread_id,
                        "status": "thread_created"
                    },
                    sender_id="system",
                    room_id=org_id
                )
                await websocket_manager.send_message(session_id, thread_notification)
            except Exception as e:
                logger.error(f"Error creating thread: {str(e)}")
            finally:
                db.close()
        
        # Echo the user message to all users in the organization
        user_message = WebSocketMessage(
            type="chat_message",
            data={
                "content": content,
                "sender_id": user_id,
                "sender_name": user_name,
                "message_type": "user",
                "thread_id": message_thread_id
            },
            sender_id=user_id,
            room_id=org_id
        )
        await websocket_manager.broadcast_to_org(org_id, user_message)
        
        # Send typing indicator for AI
        ai_typing = WebSocketMessage(
            type="typing",
            data={
                "is_typing": True,
                "sender_id": "ai_assistant",
                "sender_name": "QueryGuard AI"
            },
            sender_id="ai_assistant",
            room_id=org_id
        )
        await websocket_manager.broadcast_to_org(org_id, ai_typing)
        
        # Process with AI using the full chat logic
        start_time = datetime.utcnow()
        try:
            # Check if we have authenticated user (required for full functionality)
            if not current_user:
                # Fallback to simple QA chain if no authentication
                logger.warning(f"No authenticated user for WebSocket chat, using simple QA chain")
                
                # Use client-provided conversation history if available (no database access without auth)
                client_history = message_data.get("conversation_history", [])
                query_with_context = content
                if client_history:
                    history_list = [
                        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                        for msg in client_history
                    ]
                    context = format_conversation_context(history_list)
                    if context:
                        query_with_context = f"Previous conversation context:\n{context}\n\nCurrent question: {content}"
                
                # Log which LLM is being used
                actual_model = "unknown"
                if CHAT_LLM:
                    if hasattr(CHAT_LLM, 'model_name'):
                        actual_model = CHAT_LLM.model_name
                    elif hasattr(CHAT_LLM, 'model'):
                        actual_model = CHAT_LLM.model
                    elif hasattr(CHAT_LLM, '_model_name'):
                        actual_model = CHAT_LLM._model_name
                    llm_class = type(CHAT_LLM).__name__
                    logger.info(f"Using CHAT_LLM ({llm_class}) with model: {actual_model} for WebSocket QA chain")
                
                qa_chain = get_qa_chain(org_id, k=k, llm=CHAT_LLM)  # Use CHAT_LLM for chatbot
                result = qa_chain.invoke({"query": query_with_context})
                
                response_text = result.get("result", "I'm sorry, I couldn't generate a response.")
                source_documents = result.get("source_documents", [])
                
                # Format source documents
                sources = []
                for doc in source_documents:
                    source_info = {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Save assistant message if user is authenticated (fallback mode)
                if current_user and message_thread_id:
                    db = SessionLocal()
                    try:
                        save_assistant_message(
                            message_thread_id,
                            str(current_user.id),
                            str(current_user.org_id),
                            response_text,
                            {"sources": sources, "processing_time": processing_time},
                            db
                        )
                    except Exception as e:
                        logger.error(f"Error saving assistant message: {str(e)}")
                    finally:
                        db.close()
                
                # Send AI response
                ai_response = WebSocketMessage(
                    type="ai_response",
                    data={
                        "response": response_text,
                        "sources": sources,
                        "processing_time": processing_time,
                        "thread_id": message_thread_id,
                        "sender_id": "ai_assistant",
                        "sender_name": "QueryGuard AI",
                        "message_type": "assistant"
                    },
                    sender_id="ai_assistant",
                    room_id=org_id
                )
                await websocket_manager.broadcast_to_org(org_id, ai_response)
            else:
                # Use full chat logic with agents and tools
                # Resolve organization strictly from authenticated user
                resolved_org_id = str(current_user.org_id)
                
                # Prepare the query with conversation context
                # Use database-loaded history if available, otherwise fall back to client-provided history
                if not conversation_history:
                    client_history = message_data.get("conversation_history", [])
                    if client_history:
                        conversation_history = [
                            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                            for msg in client_history
                        ]
                
                # Prepare the query with conversation context
                query = content
                if conversation_history:
                    context = format_conversation_context(conversation_history)
                    if context:
                        query = f"Previous conversation context:\n{context}\n\nCurrent question: {content}"
                        logger.info(f"Added conversation context: {len(conversation_history)} messages")
                
                classification_label = None
                if not classification_label:
                    # LLM classification: decide whether to use tools (lineage/impact) or respond conversationally (other)
                    classifier_message = content
                    classifier_context = ""
                    if conversation_history:
                        ctx = format_conversation_context(conversation_history)
                        if ctx:
                            classifier_context = f"\nConversation context:\n{ctx}\n"
                            classifier_message = f"{classifier_context}\nCurrent message: {content}"
                    classify_prompt = (
                        "You are a classifier. Decide if the user's message requires using specialized tools for: "
                        "data lineage (extract_lineage), query impact analysis (query_history_search), "
                        "PR/Repo analysis (pr_repo_analysis), code suggestions (code_suggestion), or Jira ticket creation (create_jira_ticket).\n"
                        "If the user is confirming a repo/PR choice (e.g., 'yes that repo', 'use the listed repo') after a prior disambiguation, treat this as Jira ticket creation.\n"
                        "Respond with exactly one word: lineage, impact, pr, code, jira, or other.\n\n"
                        f"Message: {classifier_message}"
                    )
                    if not CHAT_LLM:
                        raise Exception("OpenAI API key not configured for chatbot")
                    
                    # Log which LLM is being used
                    actual_model = "unknown"
                    if CHAT_LLM:
                        if hasattr(CHAT_LLM, 'model_name'):
                            actual_model = CHAT_LLM.model_name
                        elif hasattr(CHAT_LLM, 'model'):
                            actual_model = CHAT_LLM.model
                        elif hasattr(CHAT_LLM, '_model_name'):
                            actual_model = CHAT_LLM._model_name
                        llm_class = type(CHAT_LLM).__name__
                        logger.info(f"Using CHAT_LLM ({llm_class}) with model: {actual_model} for WebSocket message classification")
                        
                    classification = CHAT_LLM.invoke(classify_prompt)
                    classification_label = (getattr(classification, "content", str(classification)) or "other").strip().lower()

                if classification_label not in {"lineage", "impact", "pr", "code", "jira"}:
                    # Conversational reply without tools
                    persona_prompt = (
                        "SYSTEM: You are Zane AI, a helpful assistant for data lineage and change-impact analysis.\n"
                        "- Be concise.\n"
                        "- Do NOT invent lineage or impacts without analysis.\n"
                        "- If the user hasn't asked for lineage/impact, introduce capabilities briefly and ask a clarifying question.\n\n"
                        f"USER: {content}\n"
                        "ASSISTANT:"
                    )
                    llm_reply = CHAT_LLM.invoke(persona_prompt)
                    reply_text = getattr(llm_reply, "content", str(llm_reply))
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Save assistant message if user is authenticated
                    if current_user and message_thread_id:
                        db = SessionLocal()
                        try:
                            save_assistant_message(
                                message_thread_id,
                                str(current_user.id),
                                str(current_user.org_id),
                                reply_text,
                                {"processing_time": processing_time},
                                db
                            )
                        except Exception as e:
                            logger.error(f"Error saving assistant message: {str(e)}")
                        finally:
                            db.close()
                    
                    ai_response = WebSocketMessage(
                        type="ai_response",
                        data={
                            "response": reply_text,
                            "sources": [],
                            "processing_time": processing_time,
                            "thread_id": message_thread_id,
                            "impacted_query_ids": [],
                            "impacted_queries": [],
                            "pr_repo_data": None,
                            "code_suggestions": None,
                            "jira_ticket": None,
                            "sender_id": "ai_assistant",
                            "sender_name": "QueryGuard AI",
                            "message_type": "assistant"
                        },
                        sender_id="ai_assistant",
                        room_id=org_id
                    )
                    await websocket_manager.broadcast_to_org(org_id, ai_response)
                else:
                    # Note: Removed fast path for Jira tickets - the tool now handles everything including
                    # project and issue type selection interactively. The tool has all the necessary logic
                    # for PR analysis and will ask for project/issue type if not provided.

                    # Build org-aware tools and delegate tool selection to the LLM agent
                    lineage_tool = build_org_lineage_tool(org_id=resolved_org_id, k=k)
                    query_history_tool = build_org_query_history_tool(org_id=resolved_org_id, max_iters=5)
                    pr_repo_tool = build_org_pr_repo_tool(org_id=resolved_org_id, default_limit=10)
                    code_suggestion_tool = build_org_code_suggestion_tool(org_id=resolved_org_id)
                    jira_tool = build_org_jira_tool(org_id=resolved_org_id, user_id=str(current_user.id))

                    # Log which LLM is being used for the agent
                    actual_model = "unknown"
                    if CHAT_LLM:
                        if hasattr(CHAT_LLM, 'model_name'):
                            actual_model = CHAT_LLM.model_name
                        elif hasattr(CHAT_LLM, 'model'):
                            actual_model = CHAT_LLM.model
                        elif hasattr(CHAT_LLM, '_model_name'):
                            actual_model = CHAT_LLM._model_name
                        llm_class = type(CHAT_LLM).__name__
                        logger.info(f"Initializing WebSocket agent with CHAT_LLM ({llm_class}) using model: {actual_model}")
                    
                    agent = initialize_agent(
                        tools=[lineage_tool, query_history_tool, pr_repo_tool, code_suggestion_tool, jira_tool],
                        llm=CHAT_LLM,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=10,  # Limit iterations to prevent infinite loops
                        max_execution_time=120,  # 2 minutes max execution time
                    )

                    # Strong guidance to the agent on tool selection and output format
                    # Include conversation awareness if we have history
                    conversation_note = ""
                    if conversation_history:
                        conversation_note = (
                            "\nCONVERSATION CONTEXT:\n"
                            "- You are continuing an ongoing conversation. Use the previous conversation context to understand references, "
                            "follow-up questions, and maintain continuity.\n"
                            "- If the user refers to something mentioned earlier (e.g., 'that table', 'the previous query'), "
                            "use the conversation context to understand what they mean.\n"
                        )
                    
                    guidance = (
                        "SYSTEM ROLE: You are Zane AI, an assistant that helps analyze data lineage and change impacts.\n"
                        "BEHAVIOR:\n"
                        "- Be concise and helpful.\n"
                        f"{conversation_note}"
                        "- CRITICAL: DO NOT call the same tool multiple times with the same input. If a tool returns results, use those results and move forward. DO NOT loop.\n"
                        "- CRITICAL: After a tool returns results, you MUST include the COMPLETE tool output in your final answer. DO NOT summarize or say 'listed above' - show the actual results.\n"
                        "- CRITICAL: When query_history_search tool returns impacted queries, you MUST copy and display ALL the queries from the tool output in your response. Include the Query IDs and SQL previews exactly as shown in the tool output.\n"
                        "- CRITICAL: DO NOT say 'queries are listed above' or 'see above' - the user cannot see above. Always include the full tool output in your response.\n"
                        "- WORKFLOW FOR CODE SUGGESTIONS: If the user asks to 'suggest code changes', 'suggest fixes', or 'code changes needed' for a PR, you MUST use the code_suggestion tool directly with the repo and PR number. The code_suggestion tool will automatically fetch PR analysis if needed. DO NOT call pr_repo_analysis first - it's not necessary.\n"
                        "- CRITICAL: If a tool returns 'NO DATA FOUND' or indicates no data is available, you MUST tell the user that no data was found. DO NOT make up or assume information. For PR/Jira flows, immediately ask the user for the repository (owner/repo) and PR number so you can retry instead of ending the conversation.\n"
                        "- CRITICAL: If the create_jira_ticket tool returns 'JIRA CONNECTION NOT CONFIGURED' or 'JIRA ERROR', you MUST inform the user that Jira is not set up. DO NOT try other tools - this is a configuration issue, not a data issue.\n"
                        "- DO NOT hallucinate lineage relationships, impacted queries, or any data that isn't explicitly returned by the tools.\n"
                        "- DO NOT try alternative tools when a tool fails due to configuration issues (like missing Jira connection).\n"
                        "- If the user greets you (e.g., 'hi', 'hello'), respond with a short intro of who you are and how you can help (lineage Q&A, query impact analysis, PR analysis, code suggestions, and Jira ticket creation).\n"
                        "- If the question is about schema/column changes or 'impacted queries', you MUST use the query_history_search tool ONCE, then copy the complete tool output into your final answer.\n"
                        "- When reporting impacted queries, include the complete numbered list with Query IDs and SQL previews from the tool output.\n"
                        "- If it's a pure lineage question, use the extract_lineage tool ONCE, then include the complete tool output in your answer.\n"
                        "- If you already suggested a repo/PR and the user replies affirmatively (e.g., 'yes', 'that repo'), assume that repo/PR and proceed with create_jira_ticket instead of asking again.\n"
                        "- If you do NOT have repo/PR details, ask for them once (owner/repo and PR number) and wait; do not loop or re-ask in the same turn.\n"
                        "- If the question is ONLY about viewing PR analysis or repository information (not asking for code suggestions), use the pr_repo_analysis tool.\n"
                        "- If the question asks for code suggestions, fixes, or changes needed for a PR, use the code_suggestion tool directly (it handles PR analysis internally).\n"
                        "- If the question asks to create a Jira ticket, use the create_jira_ticket tool.\n"
                        "- When tools return 'NO DATA FOUND', acknowledge this clearly to the user without making assumptions.\n"
                        "- When tools return configuration errors (like 'JIRA CONNECTION NOT CONFIGURED'), inform the user about the configuration issue and do NOT try other tools.\n"
                        "- IMPORTANT: Follow the ReAct format strictly. After each tool call, provide your final answer using 'Final Answer:' and include the COMPLETE tool output, not a summary."
                    )
                    # Nudge the agent to preferred tool if classification is specific
                    preferred_hint = (
                        "\nPREFERRED_TOOL: query_history_search\n" if classification_label == "impact" else (
                            "\nPREFERRED_TOOL: extract_lineage\n" if classification_label == "lineage" else (
                                "\nPREFERRED_TOOL: pr_repo_analysis\n" if classification_label == "pr" else (
                                    "\nPREFERRED_TOOL: code_suggestion\n" if classification_label == "code" else (
                                        "\nPREFERRED_TOOL: create_jira_ticket\n" if classification_label == "jira" else ""
                                    )
                                )
                            )
                        )
                    )
                    agent_query = f"{guidance}{preferred_hint}\nUser question: {query}"

                    try:
                        agent_result = agent.invoke(agent_query)
                        # LangChain agents often return dicts with `output`; fallback to str
                        if isinstance(agent_result, dict) and "output" in agent_result:
                            response_text = agent_result.get("output", "")
                        else:
                            response_text = str(agent_result)
                    except ValueError as e:
                        # Handle parsing errors - extract the actual response from the error if possible
                        error_msg = str(e)
                        if "Could not parse LLM output" in error_msg:
                            # Try to extract the response from the error message
                            # The error format is: "Could not parse LLM output: `...response...`"
                            import re
                            # Find text after "Could not parse LLM output: `" and extract until the last backtick before "For troubleshooting"
                            match = re.search(r"Could not parse LLM output: `(.*?)(?:`\s*For troubleshooting|$)", error_msg, re.DOTALL)
                            if match:
                                response_text = match.group(1).strip()
                                # Clean up common prefixes that the agent might add
                                response_text = re.sub(r"^I now know the final answer\.\s*", "", response_text, flags=re.IGNORECASE)
                                logger.warning(f"Agent parsing error handled, extracted response: {response_text[:100]}...")
                            else:
                                response_text = "I encountered an issue formatting my response. Please try rephrasing your question."
                                logger.error(f"Agent parsing error: {error_msg}")
                        else:
                            raise
                    except Exception as e:
                        # Handle iteration/time limit errors and other agent errors
                        error_msg = str(e)
                        logger.error(f"Agent execution error: {error_msg}")
                        
                        # Check for iteration/time limit errors
                        if "iteration limit" in error_msg.lower() or "time limit" in error_msg.lower() or "stopped due to" in error_msg.lower():
                            # Try to extract any partial response from the error
                            import re
                            # Look for any response content in the error
                            response_match = re.search(r'(?:output|response|answer)[:\s]+(.*?)(?:\n|$)', error_msg, re.DOTALL | re.IGNORECASE)
                            if response_match:
                                partial_response = response_match.group(1).strip()
                                if len(partial_response) > 50:  # Only use if substantial
                                    response_text = f"{partial_response}\n\n[Note: Response was truncated due to processing limits. The information above should address your query.]"
                                else:
                                    response_text = "I've reached the processing limit while analyzing your query. The tool has retrieved the impacted queries, but I wasn't able to format the complete response. Here's what was found:\n\nPlease try rephrasing your question or breaking it into smaller parts."
                            else:
                                # Check if we can extract query IDs from the error message
                                query_ids = re.findall(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", error_msg, re.I)
                                if query_ids:
                                    response_text = f"I found {len(query_ids)} impacted query(s) but encountered a processing limit. Query IDs: {', '.join(query_ids[:5])}"
                                    if len(query_ids) > 5:
                                        response_text += f" (and {len(query_ids) - 5} more)"
                                else:
                                    response_text = "I've reached the processing limit while analyzing your query. Please try rephrasing your question or breaking it into smaller parts."
                        else:
                            # For other errors, provide a generic message
                            response_text = "I encountered an error while processing your request. Please try rephrasing your question."
                            logger.error(f"Unexpected agent error: {error_msg}")

                    # Best-effort: extract query IDs from the response text and fetch full queries
                    impacted_query_ids: List[str] = []
                    impacted_queries: List[Dict[str, Any]] = []
                    try:
                        import re as _re
                        # Match UUID-like ids commonly used in results
                        impacted_query_ids = list(dict.fromkeys(_re.findall(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", response_text, flags=_re.I)))
                        if impacted_query_ids:
                            impacted_queries = fetch_queries(impacted_query_ids) or []
                    except Exception:
                        impacted_query_ids = []
                        impacted_queries = []

                    # Best-effort: extract tool payloads if present (DATA:\n{...})
                    pr_repo_data: Optional[Dict[str, Any]] = None
                    code_suggestions: Optional[Dict[str, Any]] = None
                    jira_ticket: Optional[Dict[str, Any]] = None
                    try:
                        import re as _re2, json as _json2
                        m = _re2.search(r"DATA:\n(\{[\s\S]*\})", response_text)
                        if m:
                            data = _json2.loads(m.group(1))
                            # Check what type of data it is
                            if "suggestions_by_file" in data:
                                code_suggestions = data
                            elif "ticket" in data or "jira_issue" in data:
                                jira_ticket = data
                            else:
                                pr_repo_data = data
                    except Exception:
                        pass
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Prepare metadata for saving
                    message_metadata = {
                        "impacted_query_ids": impacted_query_ids,
                        "impacted_queries": impacted_queries,
                        "pr_repo_data": pr_repo_data,
                        "code_suggestions": code_suggestions,
                        "jira_ticket": jira_ticket,
                        "processing_time": processing_time,
                        "sources": []
                    }
                    
                    # Save assistant message if user is authenticated
                    if current_user and message_thread_id:
                        db = SessionLocal()
                        try:
                            save_assistant_message(
                                message_thread_id,
                                str(current_user.id),
                                str(current_user.org_id),
                                response_text,
                                message_metadata,
                                db
                            )
                            
                            # Episodic memory storage (PostgresStore) - commented out for now
                            # Can be enabled when PostgresStore is properly configured
                            # try:
                            #     episodic_memory = {
                            #         "user_message": content,
                            #         "assistant_response": response_text[:500],
                            #         "timestamp": datetime.utcnow().isoformat(),
                            #         "classification": classification_label,
                            #         "has_tool_results": bool(impacted_queries or pr_repo_data or code_suggestions or jira_ticket)
                            #     }
                            #     memory_key = f"episode_{datetime.utcnow().timestamp()}"
                            #     store_episodic_memory(
                            #         thread_id=message_thread_id,
                            #         user_id=str(current_user.id),
                            #         org_id=resolved_org_id,
                            #         memory_key=memory_key,
                            #         memory_data=episodic_memory
                            #     )
                            # except Exception as e:
                            #     logger.warning(f"Error storing episodic memory: {str(e)} (non-critical)")
                        except Exception as e:
                            logger.error(f"Error saving assistant message: {str(e)}")
                        finally:
                            db.close()
                    
                    # Send AI response with all data
                    ai_response = WebSocketMessage(
                        type="ai_response",
                        data={
                            "response": response_text,
                            "sources": [],  # Tool outputs include their own context
                            "processing_time": processing_time,
                            "thread_id": message_thread_id,
                            "impacted_query_ids": impacted_query_ids,
                            "impacted_queries": impacted_queries,
                            "pr_repo_data": pr_repo_data,
                            "code_suggestions": code_suggestions,
                            "jira_ticket": jira_ticket,
                            "sender_id": "ai_assistant",
                            "sender_name": "QueryGuard AI",
                            "message_type": "assistant"
                        },
                        sender_id="ai_assistant",
                        room_id=org_id
                    )
                    await websocket_manager.broadcast_to_org(org_id, ai_response)
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            error_response = WebSocketMessage(
                type="ai_response",
                data={
                    "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                    "sources": [],
                    "error": str(e),
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "thread_id": message_thread_id,
                    "impacted_query_ids": [],
                    "impacted_queries": [],
                    "pr_repo_data": None,
                    "code_suggestions": None,
                    "jira_ticket": None,
                    "sender_id": "ai_assistant",
                    "sender_name": "QueryGuard AI",
                    "message_type": "assistant"
                },
                sender_id="ai_assistant",
                room_id=org_id
            )
            await websocket_manager.broadcast_to_org(org_id, error_response)
        finally:
            # Stop typing indicator
            ai_stop_typing = WebSocketMessage(
                type="typing",
                data={
                    "is_typing": False,
                    "sender_id": "ai_assistant",
                    "sender_name": "QueryGuard AI"
                },
                sender_id="ai_assistant",
                room_id=org_id
            )
            await websocket_manager.broadcast_to_org(org_id, ai_stop_typing)
            
    except Exception as e:
        logger.error(f"Error handling chat message: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

async def handle_typing_indicator(session_id: str, org_id: str, user_id: str, user_name: Optional[str], message_data: dict):
    """Handle typing indicator messages"""
    try:
        is_typing = message_data.get("data", {}).get("is_typing", False)
        
        typing_message = WebSocketMessage(
            type="typing",
            data={
                "is_typing": is_typing,
                "sender_id": user_id,
                "sender_name": user_name
            },
            sender_id=user_id,
            room_id=org_id
        )
        
        # Broadcast to everyone except the sender
        await websocket_manager.broadcast_to_org(org_id, typing_message, exclude_session=session_id)
        
    except Exception as e:
        logger.error(f"Error handling typing indicator: {str(e)}")

async def handle_ping(session_id: str):
    """Handle ping messages to keep connection alive"""
    try:
        pong_message = WebSocketMessage(
            type="pong",
            data={"message": "pong"},
            sender_id="system"
        )
        await websocket_manager.send_message(session_id, pong_message)
        
    except Exception as e:
        logger.error(f"Error handling ping: {str(e)}")

# REST API endpoints for chat management
@router.get("/sessions/{org_id}", response_model=List[ChatSessionInfo])
async def get_active_chat_sessions(org_id: str):
    """
    Get all active chat sessions for an organization.
    
    Args:
        org_id: Organization ID
        
    Returns:
        List of active chat sessions
    """
    try:
        sessions = await websocket_manager.get_active_sessions(org_id=org_id)
        
        session_infos = []
        for session in sessions:
            session_info = ChatSessionInfo(
                session_id=session.session_id,
                org_id=session.org_id,
                user_id=session.user_id,
                status=UserStatus.ONLINE,
                created_at=session.created_at,
                last_activity=session.last_activity,
                connection_count=1
            )
            session_infos.append(session_info)
        
        return session_infos
        
    except Exception as e:
        logger.error(f"Error getting active sessions for org {org_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting active sessions: {str(e)}")

@router.get("/stats", response_model=ChatStatsResponse)
async def get_chat_statistics():
    """
    Get WebSocket connection statistics.
    
    Returns:
        Chat statistics
    """
    try:
        stats = websocket_manager.get_stats()
        return ChatStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting chat statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@router.post("/cleanup")
async def cleanup_inactive_sessions(timeout_minutes: int = 30):
    """
    Manually trigger cleanup of inactive sessions.
    
    Args:
        timeout_minutes: Minutes of inactivity before cleanup
        
    Returns:
        Cleanup result
    """
    try:
        await websocket_manager.cleanup_inactive_sessions(timeout_minutes)
        return {"message": f"Cleaned up inactive sessions (timeout: {timeout_minutes} minutes)"}
        
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up sessions: {str(e)}")

@router.get("/test-page")
async def get_websocket_test_page():
    """
    Serve a simple test page for WebSocket functionality.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QueryGuard Chat WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; display: flex; gap: 20px; }
            .left-panel { width: 300px; }
            .right-panel { flex: 1; }
            .thread-list { border: 1px solid #ccc; padding: 10px; margin: 10px 0; max-height: 300px; overflow-y: scroll; }
            .thread-item { padding: 8px; margin: 5px 0; border: 1px solid #ddd; cursor: pointer; border-radius: 4px; }
            .thread-item:hover { background-color: #f0f0f0; }
            .thread-item.active { background-color: #e3f2fd; border-color: #2196F3; }
            .messages { height: 400px; border: 1px solid #ccc; overflow-y: scroll; padding: 10px; margin: 10px 0; background: #fafafa; }
            .message { margin: 5px 0; padding: 8px; border-radius: 5px; }
            .user-message { background-color: #e3f2fd; margin-left: 20%; }
            .ai-message { background-color: #f3e5f5; margin-right: 20%; }
            .system-message { background-color: #fff3e0; font-style: italic; text-align: center; }
            .input-group { display: flex; gap: 10px; margin: 10px 0; flex-wrap: wrap; }
            input, button, select { padding: 8px; }
            input[type="text"] { flex: 1; min-width: 200px; }
            .status { margin: 10px 0; font-weight: bold; padding: 8px; background: #f5f5f5; border-radius: 4px; }
            .thread-info { margin: 10px 0; padding: 8px; background: #e8f5e9; border-radius: 4px; font-size: 12px; }
            button { cursor: pointer; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="left-panel">
                <h2>Chat Threads</h2>
                <div class="input-group">
                    <button onclick="loadThreads()" id="loadThreadsBtn">Load Threads</button>
                    <button onclick="createNewThread()" id="createThreadBtn">New Chat</button>
                </div>
                <div class="thread-list" id="threadList">
                    <p style="text-align: center; color: #999;">Click "Load Threads" to see your chats</p>
                </div>
            </div>
            
            <div class="right-panel">
                <h1>QueryGuard Chat WebSocket Test</h1>
                
                <div class="input-group">
                    <input type="text" id="orgId" placeholder="Organization ID" value="76d33fb3-6062-456b-a211-4aec9971f8be">
                    <input type="text" id="userId" placeholder="User ID" value="test-user">
                    <input type="text" id="userName" placeholder="User Name" value="Test User">
                    <input type="text" id="token" placeholder="JWT Token (required for history)" style="width: 300px;">
                </div>
                <div class="input-group">
                    <select id="threadSelect" style="flex: 1;">
                        <option value="">-- Select Thread or Create New --</option>
                    </select>
                    <button onclick="connect()" id="connectBtn">Connect</button>
                    <button onclick="disconnect()" id="disconnectBtn">Disconnect</button>
                </div>
                <div style="margin: 10px 0; font-size: 12px; color: #666;">
                    <strong>Note:</strong> Token is required for chat history. Without token = Simple QA mode (no history).
                    <br>Get token from: <code>POST /auth/login</code>
                </div>
                
                <div class="thread-info" id="threadInfo" style="display: none;">
                    <strong>Current Thread:</strong> <span id="currentThreadId">None</span>
                </div>
                
                <div class="status" id="status">Disconnected</div>
                
                <div class="messages" id="messages"></div>
                
                <div class="input-group">
                    <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)" disabled>
                    <button onclick="sendMessage()" id="sendBtn" disabled>Send</button>
                    <button onclick="sendTyping(true)">Start Typing</button>
                    <button onclick="sendTyping(false)">Stop Typing</button>
                </div>
            </div>
        </div>

        <script>
            let ws = null;
            let sessionId = null;
            let currentThreadId = null;
            let token = null;
            const baseUrl = window.location.origin.replace('http', 'ws');

            // Load user's chat threads
            async function loadThreads() {
                token = document.getElementById('token').value.trim();
                if (!token) {
                    alert('Please enter JWT token to load threads');
                    return;
                }
                
                try {
                    const response = await fetch('/chat/threads', {
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to load threads');
                    }
                    
                    const threads = await response.json();
                    displayThreads(threads);
                    populateThreadSelect(threads);
                } catch (error) {
                    alert('Error loading threads: ' + error.message);
                    console.error(error);
                }
            }

            function displayThreads(threads) {
                const threadList = document.getElementById('threadList');
                threadList.innerHTML = '';
                
                if (threads.length === 0) {
                    threadList.innerHTML = '<p style="text-align: center; color: #999;">No threads found. Create a new chat!</p>';
                    return;
                }
                
                threads.forEach(thread => {
                    const div = document.createElement('div');
                    div.className = 'thread-item' + (thread.id === currentThreadId ? ' active' : '');
                    div.innerHTML = `
                        <div style="font-weight: bold;">${thread.title || 'Untitled'}</div>
                        <div style="font-size: 11px; color: #666;">
                            ${thread.message_count} messages • ${new Date(thread.last_message_at || thread.created_at).toLocaleString()}
                        </div>
                    `;
                    div.onclick = () => selectThread(thread.id);
                    threadList.appendChild(div);
                });
            }

            function populateThreadSelect(threads) {
                const select = document.getElementById('threadSelect');
                select.innerHTML = '<option value="">-- Select Thread or Create New --</option>';
                threads.forEach(thread => {
                    const option = document.createElement('option');
                    option.value = thread.id;
                    option.textContent = `${thread.title || 'Untitled'} (${thread.message_count} msgs)`;
                    select.appendChild(option);
                });
            }

            async function createNewThread() {
                token = document.getElementById('token').value.trim();
                if (!token) {
                    alert('Please enter JWT token to create thread');
                    return;
                }
                
                try {
                    const response = await fetch('/chat/threads', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ title: 'New Chat' })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to create thread');
                    }
                    
                    const thread = await response.json();
                    currentThreadId = thread.id;
                    document.getElementById('threadSelect').value = thread.id;
                    updateThreadInfo(thread.id);
                    await loadThreads(); // Reload list
                    alert('New thread created! Thread ID: ' + thread.id);
                } catch (error) {
                    alert('Error creating thread: ' + error.message);
                    console.error(error);
                }
            }

            async function selectThread(threadId) {
                token = document.getElementById('token').value.trim();
                if (!token) {
                    alert('Please enter JWT token');
                    return;
                }
                
                try {
                    const response = await fetch(`/chat/threads/${threadId}`, {
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to load thread');
                    }
                    
                    const thread = await response.json();
                    currentThreadId = thread.id;
                    document.getElementById('threadSelect').value = thread.id;
                    updateThreadInfo(thread.id);
                    
                    // Clear and load messages
                    document.getElementById('messages').innerHTML = '';
                    thread.messages.forEach(msg => {
                        addMessage(
                            msg.role === 'user' ? 'You' : 'QueryGuard AI',
                            msg.content,
                            msg.role === 'user' ? 'user' : 'ai'
                        );
                    });
                    
                    await loadThreads(); // Reload to highlight active
                } catch (error) {
                    alert('Error loading thread: ' + error.message);
                    console.error(error);
                }
            }

            function updateThreadInfo(threadId) {
                const info = document.getElementById('threadInfo');
                const idSpan = document.getElementById('currentThreadId');
                if (threadId) {
                    info.style.display = 'block';
                    idSpan.textContent = threadId;
                } else {
                    info.style.display = 'none';
                }
            }

            function connect() {
                const orgId = document.getElementById('orgId').value;
                const userId = document.getElementById('userId').value;
                const userName = document.getElementById('userName').value;
                token = document.getElementById('token').value.trim();
                const selectedThreadId = document.getElementById('threadSelect').value;
                
                if (!orgId || !userId) {
                    alert('Please enter Organization ID and User ID');
                    return;
                }
                
                sessionId = 'test-session-' + Math.random().toString(36).substr(2, 9);
                let wsUrl = `${baseUrl}/chat/ws/${orgId}/${userId}?session_id=${sessionId}&user_name=${encodeURIComponent(userName)}`;
                
                // Add token if provided (for full features and history)
                if (token) {
                    wsUrl += `&token=${encodeURIComponent(token)}`;
                }
                
                // Add thread_id if selected
                if (selectedThreadId) {
                    wsUrl += `&thread_id=${encodeURIComponent(selectedThreadId)}`;
                    currentThreadId = selectedThreadId;
                    updateThreadInfo(selectedThreadId);
                }
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    document.getElementById('status').textContent = 'Connected';
                    document.getElementById('status').style.background = '#c8e6c9';
                    document.getElementById('messageInput').disabled = false;
                    document.getElementById('sendBtn').disabled = false;
                    addMessage('System', 'Connected to WebSocket', 'system');
                    if (currentThreadId) {
                        addMessage('System', `Using thread: ${currentThreadId}`, 'system');
                    }
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function(event) {
                    document.getElementById('status').textContent = 'Disconnected';
                    document.getElementById('status').style.background = '#ffcdd2';
                    document.getElementById('messageInput').disabled = true;
                    document.getElementById('sendBtn').disabled = true;
                    addMessage('System', 'WebSocket connection closed', 'system');
                };
                
                ws.onerror = function(error) {
                    addMessage('System', 'WebSocket error: ' + error, 'system');
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }

            function sendMessage() {
                const input = document.getElementById('messageInput');
                const content = input.value.trim();
                
                if (!content || !ws) return;
                
                const message = {
                    type: 'chat_message',
                    content: content,
                    thread_id: currentThreadId  // Include thread_id in message
                };
                
                ws.send(JSON.stringify(message));
                input.value = '';
            }

            function sendTyping(isTyping) {
                if (!ws) return;
                
                const message = {
                    type: 'typing',
                    data: { is_typing: isTyping }
                };
                
                ws.send(JSON.stringify(message));
            }

            function handleMessage(data) {
                console.log('Received:', data);
                
                switch(data.type) {
                    case 'chat_message':
                        const senderName = data.data.sender_name || data.sender_id;
                        addMessage(senderName, data.data.content, 'user');
                        break;
                    case 'ai_response':
                        let aiText = data.data.response;
                        // Update thread_id if received
                        if (data.data.thread_id && !currentThreadId) {
                            currentThreadId = data.data.thread_id;
                            updateThreadInfo(data.data.thread_id);
                            document.getElementById('threadSelect').value = data.data.thread_id;
                        }
                        // Show additional data if available
                        if (data.data.impacted_queries && data.data.impacted_queries.length > 0) {
                            aiText += '\\n\\n[Impacted Queries: ' + data.data.impacted_queries.length + ' found]';
                        }
                        if (data.data.pr_repo_data) {
                            aiText += '\\n[PR/Repo data available]';
                        }
                        if (data.data.code_suggestions) {
                            aiText += '\\n[Code suggestions available]';
                        }
                        if (data.data.jira_ticket) {
                            aiText += '\\n[Jira ticket created]';
                        }
                        if (data.data.processing_time) {
                            aiText += '\\n(Processing time: ' + data.data.processing_time.toFixed(2) + 's)';
                        }
                        if (data.data.thread_id) {
                            aiText += '\\n[Thread ID: ' + data.data.thread_id + ']';
                        }
                        addMessage('QueryGuard AI', aiText, 'ai');
                        // Log full response to console for debugging
                        console.log('Full AI Response:', data.data);
                        break;
                    case 'system_message':
                        addMessage('System', data.data.message, 'system');
                        // Handle thread creation notification
                        if (data.data.thread_id && data.data.status === 'thread_created') {
                            currentThreadId = data.data.thread_id;
                            updateThreadInfo(data.data.thread_id);
                            document.getElementById('threadSelect').value = data.data.thread_id;
                            loadThreads(); // Reload thread list
                        }
                        break;
                    case 'typing':
                        const typingSender = data.data.sender_name || data.sender_id;
                        if (data.data.is_typing) {
                            addMessage(typingSender, 'is typing...', 'system');
                        }
                        break;
                    case 'user_status':
                        addMessage('System', data.data.message, 'system');
                        break;
                }
            }

            function addMessage(sender, content, type) {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                // Format content with line breaks
                const formattedContent = content.replace(/\\n/g, '<br>');
                messageDiv.innerHTML = `<strong>${sender}:</strong><br>${formattedContent}`;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            // Thread select change handler
            document.getElementById('threadSelect').addEventListener('change', function(e) {
                if (e.target.value) {
                    selectThread(e.target.value);
                } else {
                    currentThreadId = null;
                    updateThreadInfo(null);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
