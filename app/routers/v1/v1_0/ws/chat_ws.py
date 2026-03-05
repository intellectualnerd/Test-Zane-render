from fastapi import APIRouter
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import List, Optional, Dict, Any
from app.vector_db import get_qa_chain
from app.utils.websocket_manager import websocket_manager, WebSocketMessage
from app.utils.websocket_models import (
    MessageType,
)
import json
from app.utils.models import User
from app.database import SessionLocal
from app.tools import build_org_lineage_tool, build_org_query_history_tool, build_org_pr_repo_tool, build_org_code_suggestion_tool, build_org_jira_tool
from app.vector_db import CHAT_LLM
from app.services.impact_analysis import fetch_queries
import logging
import uuid
from datetime import datetime
from langchain.agents import initialize_agent, AgentType
logger = logging.getLogger(__name__)

# Configuration for conversation history limits
# These can be adjusted based on LLM context window and requirements
# Note: Most modern LLMs have context windows of 8K-128K tokens
# Adjust these values based on your LLM's context window and desired behavior
MAX_CONTEXT_MESSAGES = 20  # Maximum number of messages to include in context (default: 20)
MAX_CONTEXT_TOKENS = 8000  # Approximate max tokens for context (rough estimate: 1 token ≈ 4 chars)
CONTEXT_MESSAGE_ESTIMATE = 200  # Estimated tokens per message (rough estimate, not used but kept for reference)


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




# WebSocket endpoint for real-time chat

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
    print("🔥 ENDPOINT WS HIT")
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


