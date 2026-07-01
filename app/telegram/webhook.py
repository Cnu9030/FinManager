import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
import httpx

from app.config import settings
from app.agents.conversation_graph import compiled_graph

logger = logging.getLogger("telegram_webhook")

router = APIRouter(prefix="/telegram", tags=["Telegram"])

# -----------------
# Telegram Pydantic Models for Webhook Payloads
# -----------------
class TelegramUser(BaseModel):
    id: int
    first_name: Optional[str] = None
    username: Optional[str] = None

class TelegramChat(BaseModel):
    id: int
    type: Optional[str] = None

class TelegramMessage(BaseModel):
    message_id: int
    from_user: TelegramUser = Field(..., alias="from")
    chat: TelegramChat
    text: Optional[str] = None

class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[TelegramMessage] = None

# Helper to send message back to Telegram
async def send_telegram_response(chat_id: int, text: str):
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not configured.")
        return
        
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, json=payload, timeout=10.0)
            r.raise_for_status()
            logger.info(f"Successfully sent reply to Telegram chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send message to Telegram: {e}", exc_info=True)

async def run_conversation_workflow(chat_id: int, text: str):
    """
    Executes the LangGraph conversation state graph asynchronously 
    and sends the final response to Telegram.
    """
    config = {"configurable": {"thread_id": str(chat_id)}}
    state_input = {
        "messages": [{"role": "user", "content": text}]
    }
    
    try:
        # Invoke LangGraph
        final_state = await compiled_graph.ainvoke(state_input, config)
        
        # Extract last assistant message
        response_text = "Transaction processed successfully."
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                response_text = msg.get("content", response_text)
                break
                
        # Send reply
        await send_telegram_response(chat_id, response_text)
    except Exception as e:
        logger.error(f"Error executing conversation graph workflow: {e}", exc_info=True)
        await send_telegram_response(chat_id, f"An error occurred while processing your request: {str(e)}")

@router.post("/webhook")
async def telegram_webhook(update: TelegramUpdate, background_tasks: BackgroundTasks):
    """
    Secure endpoint to receive updates from Telegram.
    Checks the user's ID to authorize the request, then runs LangGraph.
    """
    logger.info("Received update from Telegram webhook")
    
    if not update.message:
        return {"status": "ignored", "reason": "No message field present"}
        
    # Security Gate: check sender ID
    sender_id = update.message.from_user.id
    if sender_id != settings.EXPECTED_TELEGRAM_USER_ID:
        logger.warning(f"Unauthorized access attempt from user ID: {sender_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Unauthorized Telegram User ID"
        )
        
    user_text = update.message.text
    chat_id = update.message.chat.id
    
    if not user_text:
        return {"status": "ignored", "reason": "Empty message text"}
        
    logger.info(f"Authorized message from chat {chat_id}: {user_text}")
    
    # Process the conversation asynchronously in background tasks to free up the request thread
    background_tasks.add_task(run_conversation_workflow, chat_id, user_text)
    
    return {"status": "processing"}
