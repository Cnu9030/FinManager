import logging
from typing import List, Dict, Any, Optional
from google import genai
from app.config import settings
from app.core.db import supabase

logger = logging.getLogger("memory_agent")

def generate_embedding(text: str) -> List[float]:
    """
    Generates a 768-dimensional vector embedding for the input text
    using Gemini's text-embedding-004 model.
    """
    if not text:
        return []
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        if response.embeddings:
            return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}", exc_info=True)
        raise e
    return []

def store_memory(context_summary: str) -> str:
    """
    Converts a takeaway or milestone into an embedding and inserts it
    into the financial_memories table. Returns the created memory ID.
    """
    logger.info(f"Storing semantic memory: '{context_summary[:50]}...'")
    embedding = generate_embedding(context_summary)
    if not embedding:
        raise ValueError("Could not generate embedding for memory.")
        
    try:
        response = supabase.table("financial_memories").insert({
            "context_summary": context_summary,
            "embedding": embedding
        }).execute()
        
        if response.data:
            memory_id = response.data[0]["id"]
            logger.info(f"Memory stored successfully with ID: {memory_id}")
            return memory_id
    except Exception as e:
        logger.error(f"Failed to store memory in database: {e}", exc_info=True)
        raise e
    raise RuntimeError("Failed to insert memory record.")

def retrieve_memories(query: str, threshold: float = 0.3, count: int = 3) -> List[Dict[str, Any]]:
    """
    Searches the financial_memories table for records semantically similar
    to the query using Cosine Similarity.
    """
    logger.info(f"Retrieving memories for query: '{query}'")
    embedding = generate_embedding(query)
    if not embedding:
        return []
        
    try:
        response = supabase.rpc(
            "match_financial_memories",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": count
            }
        ).execute()
        return response.data or []
    except Exception as e:
        logger.error(f"Database error during memory retrieval: {e}", exc_info=True)
    return []

async def retrieve_memory_node(state: Any) -> Dict[str, Any]:
    """
    LangGraph initial node serving as the RAG hook.
    Fetches semantically relevant financial records and prepends them
    directly into the messages context.
    """
    logger.info("Running retrieve_memory_node")
    messages = state.messages
    if not messages:
        return {}
        
    # Get the last user message to extract RAG query
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
            
    if not last_user_msg:
        return {}
        
    # Query semantic memory
    matched_memories = retrieve_memories(last_user_msg)
    
    if not matched_memories:
        return {}
        
    # Format matched records into a readable system instruction context
    memory_context_lines = []
    for mem in matched_memories:
        memory_context_lines.append(f"- {mem['context_summary']}")
        
    memory_payload_str = "\n".join(memory_context_lines)
    
    context_prefix_msg = {
        "role": "system",
        "content": (
            "Below is relevant context retrieved from the user's historical financial memories. "
            "Use this information if relevant to address the user's query:\n"
            f"{memory_payload_str}\n"
        )
    }
    
    # Prepend the system context message right before the last user message
    new_messages = []
    inserted = False
    for msg in messages:
        if msg == messages[-1] and msg.get("role") == "user":
            new_messages.append(context_prefix_msg)
            inserted = True
        new_messages.append(msg)
        
    if not inserted:
        new_messages.insert(0, context_prefix_msg)
        
    return {"messages": new_messages}
