import logging
import os
import traceback
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
import httpx

from app.config import settings
from app.agents.conversation_graph import compiled_graph

logger = logging.getLogger("telegram_webhook")

router = APIRouter(prefix="/telegram", tags=["Telegram"])

# -----------------
# Gmail Crash Alert Utility
# -----------------
def send_gmail_alert(
    user_name: Optional[str],
    chat_id: int,
    user_message: str,
    error_summary: str,
    full_traceback: str
) -> bool:
    """
    Sends an HTML-formatted crash report email via Gmail SMTP.
    
    Args:
        user_name: Name of the user who triggered the error.
        chat_id: Telegram chat ID where the error occurred.
        user_message: The raw text message from the user that caused the crash.
        error_summary: Brief error message (str(exception)).
        full_traceback: Complete Python stack trace formatted via traceback.format_exc().
    
    Returns:
        bool: True if email sent successfully, False otherwise.
    """
    gmail_user = os.getenv("GMAIL_USER", "").strip()
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    
    if not gmail_user or not gmail_password:
        logger.warning("Gmail credentials not configured (GMAIL_USER or GMAIL_APP_PASSWORD missing).")
        return False
    
    try:
        # Create MIME message with alternative parts (plain text + HTML)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Finance Agent ALERT] Crash Notification from Chat {chat_id}"
        msg["From"] = gmail_user
        msg["To"] = gmail_user
        
        # Build HTML body
        html_body = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #d32f2f;">🚨 Finance Agent Crash Detected</h2>
            
            <h3>User Information</h3>
            <p><strong>Name:</strong> {user_name or 'Unknown'}</p>
            <p><strong>Telegram Chat ID:</strong> {chat_id}</p>
            
            <h3>User Input (Triggering Message)</h3>
            <p><code>{user_message}</code></p>
            
            <h3>Error Summary</h3>
            <p><strong style="color: #c62828;">{error_summary}</strong></p>
            
            <h3>Full Stack Trace</h3>
            <pre style="background-color: #f5f5f5; padding: 10px; border-left: 3px solid #d32f2f; overflow-x: auto;">
{full_traceback}
            </pre>
            
            <p style="margin-top: 30px; font-size: 12px; color: #666;">
              Auto-generated alert from Finance Agent Webhook Handler
            </p>
          </body>
        </html>
        """
        
        # Attach HTML part
        html_part = MIMEText(html_body, "html")
        msg.attach(html_part)
        
        # Send via Gmail SMTP
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Crash alert email sent successfully to {gmail_user}")
        return True
    except Exception as e:
        logger.error(f"Failed to send Gmail alert: {e}", exc_info=True)
        return False


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

async def run_conversation_workflow(chat_id: int, text: str, user_name: Optional[str] = None):
    """
    Executes the LangGraph conversation state graph asynchronously 
    and sends the final response to Telegram. Includes comprehensive error handling
    with user notification and crash logging to Gmail.
    """
    config = {"configurable": {"thread_id": str(chat_id)}}
    state_input = {
        "messages": [{"role": "user", "content": text}]
    }
    
    try:
        # Invoke LangGraph
        final_state = await compiled_graph.ainvoke(state_input, config)
        
        # Extract the last natural-language AI response from the graph output
        response_text = "I received your request, but I couldn't generate a readable reply. Please try again."
        messages = []
        if isinstance(final_state, dict):
            messages = final_state.get("messages", []) or []

        for msg in reversed(messages):
            if isinstance(msg, dict):
                message_type = msg.get("type") or msg.get("role")
                content = msg.get("content")
            else:
                message_type = getattr(msg, "type", None) or getattr(msg, "role", None)
                content = getattr(msg, "content", None)

            if message_type != "ai" and message_type != "assistant":
                continue

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        part_text = part.get("text") or part.get("content")
                        if part_text:
                            text_parts.append(str(part_text))
                content = "".join(text_parts).strip()

            if isinstance(content, str) and content.strip():
                response_text = content.strip()
                break

        # Send reply
        await send_telegram_response(chat_id, response_text)
    except Exception as app_error:
        # Capture full diagnostic information
        error_summary = str(app_error)
        full_traceback = traceback.format_exc()
        
        # Log the full traceback to standard logging (streams to Render container logs)
        logger.error(f"Exception in conversation workflow: {error_summary}", exc_info=True)
        
        # Send user-friendly notification via Telegram
        user_notification = (
            "⚠️ *An unexpected error occurred*\n\n"
            "I apologize for the inconvenience. Our team has been notified of this issue and is working to resolve it. "
            "Please try again in a moment, or contact support if the problem persists."
        )
        await send_telegram_response(chat_id, user_notification)
        
        # Send detailed crash report to Gmail for investigation
        try:
            send_gmail_alert(
                user_name=user_name,
                chat_id=chat_id,
                user_message=text,
                error_summary=error_summary,
                full_traceback=full_traceback
            )
        except Exception as email_err:
            logger.error(f"CRITICAL: Gmail alert system failed to send: {email_err}", exc_info=True)


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
    user_name = update.message.from_user.first_name
    
    if not user_text:
        return {"status": "ignored", "reason": "Empty message text"}
        
    logger.info(f"Authorized message from chat {chat_id}: {user_text}")
    
    # Process the conversation asynchronously in background tasks to free up the request thread
    background_tasks.add_task(run_conversation_workflow, chat_id, user_text, user_name)
    
    return {"status": "processing"}
