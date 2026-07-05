import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.agents.guardrails import get_source_id_by_name
from app.core.event_bus import event_bus
from app.agents.memory_agent import retrieve_memory_node

logger = logging.getLogger("conversation_graph")

MAX_MESSAGE_HISTORY = 20
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_TRACKING_PREFIX = "SYSTEM_TRACKING:"

REQUIRED_FIELDS_BY_INTENT = {
    "ADD_EXPENSE": ["amount", "source_name", "description"],
    "ADD_SOURCE": ["source_name"],
    "RECORD_INCOME": ["amount", "source_name", "description"],
    "SIMULATE_PURCHASE": ["amount"],
}

FIELD_LABELS = {
    "amount": "amount",
    "source_name": "source account",
    "description": "description",
    "owner_name": "owner name",
}


def append_messages(old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined_messages = (old or []) + (new or [])
    return combined_messages[-MAX_MESSAGE_HISTORY:]

class ConversationState(BaseModel):
    """
    Pydantic schema tracking the cognitive graph state.
    """
    messages: Annotated[List[Dict[str, Any]], append_messages] = Field(default_factory=list)
    extracted_intent: Optional[Dict[str, Any]] = None
    action_result: Optional[Dict[str, Any]] = None
    execution_status: str = "PENDING"

class IntentExtractionSchema(BaseModel):
    """
    Schema for structured JSON extraction by Gemini.
    """
    intent_type: str = Field(
        description=(
            "Type of intent (e.g., 'ADD_EXPENSE', 'ADD_SOURCE', 'RECORD_INCOME', 'CHECK_BALANCE', 'VIEW_REPORT', 'SIMULATE_PURCHASE'). "
            "ADD_EXPENSE represents money leaving the user as an outflow, spend, debit, charge, or purchase. "
            "RECORD_INCOME represents money entering the user as an inflow, earning, credit, deposit, salary, bonus, allowance, dividend, or transfer received."
        )
    )
    amount: Optional[float] = Field(None, description="Decimal amount of the transaction or purchase if applicable")
    source_name: Optional[str] = Field(None, description="Name of the source account/asset if applicable")
    owner_name: Optional[str] = Field(None, description="Name of the owner or self person if applicable")
    description: Optional[str] = Field(None, description="Description, category, or simulation query of the transaction/action")
    context_relation_missing: bool = Field(
        default=False,
        description="Set to True ONLY if there is an existing intent pending, but the latest user message shares NO relational context or continuity with it (e.g., starting an entirely separate transaction)."
    )


def _get_required_fields(intent_type: Optional[str]) -> List[str]:
    if not intent_type:
        return []
    return REQUIRED_FIELDS_BY_INTENT.get(intent_type, [])


def _get_missing_required_fields(intent: Dict[str, Any]) -> List[str]:
    required_fields = _get_required_fields(intent.get("intent_type"))
    missing_fields = []
    for field_name in required_fields:
        value = intent.get(field_name)
        if value in (None, "", [], {}):
            missing_fields.append(field_name)
    return missing_fields


def _has_valid_intent(existing_intent: Optional[Dict[str, Any]]) -> bool:
    return bool(existing_intent and existing_intent.get("intent_type"))


def _get_latest_user_message_content(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _merge_intent_data(existing_intent: Optional[Dict[str, Any]], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    if extracted_data.get("context_relation_missing"):
        cleaned_intent = {
            field_name: value
            for field_name, value in extracted_data.items()
            if field_name != "context_relation_missing" and value not in (None, "", [], {})
        }
        return cleaned_intent

    existing_intent = dict(existing_intent or {})
    new_intent_type = extracted_data.get("intent_type")
    old_intent_type = existing_intent.get("intent_type")

    if new_intent_type and old_intent_type and new_intent_type != old_intent_type:
        return {
            field_name: value
            for field_name, value in extracted_data.items()
            if field_name != "context_relation_missing" and value not in (None, "", [], {})
        }

    merged_intent = dict(existing_intent)

    for field_name, value in extracted_data.items():
        if field_name == "context_relation_missing":
            continue
        if value in (None, "", [], {}):
            continue
        if field_name == "intent_type" and merged_intent.get("intent_type"):
            continue
        merged_intent[field_name] = value

    return merged_intent


def _build_balance_response(action_result: Dict[str, Any]) -> str:
    current_balance = action_result.get("current_balance")
    if current_balance not in (None, "", [], {}):
        return f"The current balance retrieved from the database is {current_balance}."
    return "I couldn't retrieve the balance at this time."


def _build_intent_extraction_prompt(state: ConversationState) -> str:
    formatted_history = []
    for msg in state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted_history.append(f"{role.upper()}: {content}")

    latest_user_message = _get_latest_user_message_content(state.messages)
    existing_intent = state.extracted_intent or {}

    if _has_valid_intent(existing_intent):
        missing_fields = _get_missing_required_fields(existing_intent)
        return (
            "You are in a slot-filling loop for an already established financial intent.\n"
            "Evaluate whether the latest user message still relates to the pending intent flow.\n"
            "If the user has clearly started a new and unrelated transaction, set context_relation_missing to true and extract the new intent from scratch.\n"
            "If the message still belongs to the pending flow, preserve the existing intent_type and already extracted parameters, and only extract values explicitly provided in the latest user message for the missing fields.\n"
            f"Existing Intent JSON:\n{json.dumps(existing_intent, ensure_ascii=False, default=str, indent=2)}\n\n"
            f"Missing Fields: {json.dumps(missing_fields, ensure_ascii=False)}\n\n"
            f"Latest User Message:\n{latest_user_message}\n\n"
            "Conversation History:\n"
            + "\n".join(formatted_history)
        )

    return (
        "You are a precise finance tracking agent. Analyze the conversation history and extract the user's intent. "
        "Only populate fields that are explicitly or strongly implied by the conversation.\n\n"
        "Intent classification rules:\n"
        "- RECORD_INCOME: any financial inflow, earning, credit, deposit, addition of funds, salary, bonus, allowance, dividend, or money received/credited.\n"
        "  Trigger phrases include: received, salary, credited, earned, got paid, income, bonus, allowance, deposit, dividend.\n"
        "- ADD_EXPENSE: any financial outflow, spend, debit, charge, or purchase.\n"
        "  Trigger phrases include: spent, paid, bought, purchased, expense, cost, gave, sent money to.\n"
        "Any message where the user gains, receives, or is credited money must be categorized as RECORD_INCOME. "
        "Never classify salary or incoming gifts/transfers as an expense, even if the source name is omitted or missing in the initial statement.\n"
        "If the user says 'Received X from Y', treat it as RECORD_INCOME unless the surrounding context clearly indicates a different explicit intent.\n\n"
        "Conversation History:\n" + "\n".join(formatted_history)
    )


def _build_tracking_message(label: str, payload: Any) -> Dict[str, Any]:
    return {
        "role": "system",
        "content": f"{SYSTEM_TRACKING_PREFIX} {label}: {payload}"
    }


def _extract_tracking_messages(messages: list) -> list:
    """
    Extracts system tracking logs isolated strictly to the current turn
    to prevent historical context bleed from past checkpoint states.
    """
    tracking_messages = []
    if not messages:
        return tracking_messages

    # Step 1: Identify the boundary index of the current turn's user input
    last_user_idx = -1
    for i, m in enumerate(messages):
        # Extract role/type dynamically supporting both class instances and dicts
        if isinstance(m, dict):
            role = m.get("role", "") or getattr(m, "type", "")
        else:
            role = getattr(m, "type", "") or ""

        if not role and hasattr(m, "__class__"):
            role_name = m.__class__.__name__
            if "Human" in role_name or "User" in role_name:
                role = "user"

        role = str(role).lower()
        if role in ("user", "human"):
            last_user_idx = i

    # Step 2: Slice the message array to evaluate only the current execution window
    current_turn_messages = messages[last_user_idx + 1:] if last_user_idx != -1 else messages

    # Step 3: Parse execution metadata safely within the isolated turn window
    for m in current_turn_messages:
        if isinstance(m, dict):
            content = m.get("content", "")
            role = (m.get("role", "") or getattr(m, "type", "")).lower()
        else:
            content = getattr(m, "content", "")
            role = str(getattr(m, "type", "")).lower()

        if role and role != "system":
            continue

        if isinstance(content, str) and content.startswith(SYSTEM_TRACKING_PREFIX):
            tracking_messages.append(content)

    return tracking_messages


def _build_fallback_response(state: ConversationState, missing_fields: List[str], tracking_messages: List[str]) -> str:
    intent = state.extracted_intent or {}
    intent_type = intent.get("intent_type") or "your request"
    action_result = state.action_result or {}

    if intent.get("intent_type") == "CHECK_BALANCE":
        return _build_balance_response(action_result)

    if missing_fields:
        readable_fields = [FIELD_LABELS.get(field, field.replace("_", " ")) for field in missing_fields]
        if len(readable_fields) == 1:
            return f"I can help with that. Could you share the {readable_fields[0]} for this {intent_type.lower().replace('_', ' ')}?"
        if len(readable_fields) == 2:
            return (
                f"I can help with that. Could you share the {readable_fields[0]} and {readable_fields[1]} for this "
                f"{intent_type.lower().replace('_', ' ')}?"
            )
        field_text = ", ".join(readable_fields[:-1]) + f", and {readable_fields[-1]}"
        return f"I can help with that. Could you share the {field_text} for this {intent_type.lower().replace('_', ' ')}?"

    if tracking_messages:
        latest_tracking = tracking_messages[-1]
        if "report_data:" in latest_tracking:
            return "I have prepared the report details and shared them above."
        if "verification_failed:" in latest_tracking:
            return "I couldn't complete that request because it failed verification. Please review the details and try again."
        if "system_error:" in latest_tracking:
            return "I ran into a system error while processing that request. Please try again in a moment."

    if state.execution_status == "SIMULATED":
        return "I’ve completed the simulation and shared the outcome above."
    if state.execution_status == "COMPLETED":
        return "Your request has been completed successfully."
    if state.execution_status == "FAILED":
        return "I couldn’t complete that request right now. Please try again."

    return "I’m ready to help. Please share a bit more detail so I can continue."


def _get_latest_user_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _append_local_chat_history(user_text: str, bot_text: str) -> None:
    if not user_text or not bot_text:
        return

    timestamp_user = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_bot = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp_user}] USER: {user_text}\n"
        f"[{timestamp_bot}] BOT: {bot_text}\n"
        "--------------------------------------------------\n"
    )

    try:
        with open("local_chat_history.log", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
    except Exception as exc:
        logger.warning("Failed to write local chat history log: %s", exc, exc_info=True)


def _build_response_prompt(state: ConversationState, missing_fields: List[str], tracking_messages: List[str]) -> str:
    sanitized_messages = [
        message for message in state.messages
        if not (message.get("role") == "system" and str(message.get("content", "")).startswith(SYSTEM_TRACKING_PREFIX))
    ][-MAX_MESSAGE_HISTORY:]

    prompt_payload = {
        "execution_status": state.execution_status,
        "extracted_intent": state.extracted_intent,
        "action_result": state.action_result,
        "missing_required_fields": missing_fields,
        "tracking_messages": tracking_messages,
        "messages": state.messages[-MAX_MESSAGE_HISTORY:],
        "sanitized_messages": sanitized_messages,
    }

    balance_instruction = ""
    if (state.extracted_intent or {}).get("intent_type") == "CHECK_BALANCE":
        action_result = state.action_result or {}
        current_balance = action_result.get("current_balance")
        if current_balance not in (None, "", [], {}):
            balance_instruction = (
                f"The current balance retrieved from the database is {current_balance}. "
                "Include this exact figure in your natural language response to the user."
            )
        else:
            balance_instruction = "If no action_result is found, respond exactly with: I couldn't retrieve the balance at this time."

    return (
        "You are a professional, friendly financial assistant.\n"
        "Write the final user-facing reply in natural language.\n"
        "Use the conversational history, extracted intent, and execution status to respond appropriately.\n"
        "If the intent is recognized but required fields are missing or null, ask a polite follow-up question that requests only the missing information.\n"
        "Do not mention internal status labels, tool names, event bus details, or system tracking strings.\n"
        "If there is a verification failure or system error, explain it briefly and helpfully without sounding mechanical.\n"
        f"{balance_instruction}\n"
        "Return only the final response text.\n\n"
        "Context JSON:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, default=str, indent=2)}"
    )


def _is_resource_exhausted_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    code = getattr(error, "code", None)
    if code == 429:
        return True

    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True

    error_name = error.__class__.__name__.upper()
    error_text = str(error).upper()
    return "RESOURCE_EXHAUSTED" in error_name or "RESOURCE_EXHAUSTED" in error_text or "429" in error_text


def _extract_structured_intent_data(response_text: str) -> Dict[str, Any]:
    extracted_data = json.loads(response_text)
    if extracted_data.get("amount") is not None:
        extracted_data["amount"] = str(extracted_data["amount"])
    return extracted_data


def _create_groq_client():
    from groq import Groq

    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return Groq(api_key=settings.GROQ_API_KEY)


def _generate_structured_intent_with_groq(prompt: str) -> Dict[str, Any]:
    client = _create_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Return only valid JSON that matches the intent extraction schema."
                ),
            }
        ],
        temperature=0,
    )

    response_text = response.choices[0].message.content or "{}"
    return _extract_structured_intent_data(response_text)


def _fetch_balance_action_result(intent: Dict[str, Any]) -> Dict[str, Any]:
    source_name = intent.get("source_name")
    if not source_name:
        return {}

    source_id = get_source_id_by_name(source_name)
    if not source_id:
        return {}

    response = supabase.table("sources").select("id, name, current_balance").eq("id", source_id).execute()
    if not response.data:
        return {}

    source_data = response.data[0]
    return {
        "source_id": source_data.get("id"),
        "source_name": source_data.get("name"),
        "current_balance": str(source_data.get("current_balance")),
    }

async def intent_extractor_node(state: ConversationState) -> Dict[str, Any]:
    """
    Sends conversation history to Gemini 2.5 Flash via google-genai SDK 
    to extract structural intent using structured outputs.
    """
    logger.info("Running intent_extractor_node")

    prompt = _build_intent_extraction_prompt(state)
    
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IntentExtractionSchema,
            ),
        )
        
        # Load and parse structured output
        extracted_data = _extract_structured_intent_data(response.text or "{}")
        logger.info(f"Gemini raw structured extraction: {extracted_data}")
        
        # Ensure Decimal formatting for amount (e.g. serialize to string or handle appropriately)
        if extracted_data.get("amount") is not None:
            extracted_data["amount"] = str(extracted_data["amount"])

        merged_intent = _merge_intent_data(state.extracted_intent, extracted_data)
            
        return {
            "extracted_intent": merged_intent,
            "execution_status": "INTENT_EXTRACTED"
        }
    except Exception as e:
        if not _is_resource_exhausted_error(e):
            logger.error(f"Error during intent extraction: {e}", exc_info=True)
            return {
                "execution_status": "FAILED"
            }

        logger.warning("Primary model (Gemini) exhausted, falling back to Groq Llama 3.1 8B.")
        try:
            extracted_data = _generate_structured_intent_with_groq(prompt)
            logger.info(f"Groq raw structured extraction: {extracted_data}")

            merged_intent = _merge_intent_data(state.extracted_intent, extracted_data)
            return {
                "extracted_intent": merged_intent,
                "execution_status": "INTENT_EXTRACTED"
            }
        except Exception as fallback_error:
            logger.error(f"Error during Groq fallback intent extraction: {fallback_error}", exc_info=True)
        return {
            "execution_status": "FAILED"
        }

async def execute_action_node(state: ConversationState) -> Dict[str, Any]:
    """
    Bridges the Graph to the Event Bus: publishes intent.extracted,
    allowing guardrails to process and inspect before executing mutations.
    """
    logger.info("Running execute_action_node")
    if state.execution_status == "FAILED" or not state.extracted_intent:
        return {"execution_status": "FAILED"}

    if state.extracted_intent.get("intent_type") == "CHECK_BALANCE":
        action_result = _fetch_balance_action_result(state.extracted_intent)
        return {
            "execution_status": "COMPLETED",
            "action_result": action_result,
        }

    missing_fields = _get_missing_required_fields(state.extracted_intent)
    if missing_fields:
        logger.info(
            "Skipping execution for %s because required fields are missing: %s",
            state.extracted_intent.get("intent_type"),
            ", ".join(missing_fields)
        )
        return {
            "execution_status": "NEEDS_CLARIFICATION",
            "messages": [
                _build_tracking_message(
                    "missing_required_fields",
                    {
                        "intent_type": state.extracted_intent.get("intent_type"),
                        "missing_fields": missing_fields,
                    }
                )
            ],
        }
        
    # If the intent is to simulate a purchase, we bypass event bus state mutations
    if state.extracted_intent.get("intent_type") == "SIMULATE_PURCHASE":
        return {"execution_status": "READY_FOR_SIMULATION"}

    # Prepare transactional context event payload
    payload = {
        "intent": state.extracted_intent,
        "messages": state.messages,
        "status": "PROCESSING",
        "error": None,
        "report_data": None
    }
    
    try:
        # Publish event and await completion of the concurrent event subscribers (Audit, Rule Engine, etc.)
        await event_bus.publish("intent.extracted", payload)
        
        # A guardrail agent may intercept and inject an error string on validation failure
        if payload.get("error"):
            logger.warning(f"Guardrail intercepted action execution: {payload['error']}")
            return {
                "execution_status": "FAILED",
                "messages": [
                    _build_tracking_message("verification_failed", payload["error"])
                ]
            }
            
        # Check if reporting agent returned aggregate SQL data
        if payload.get("report_data"):
            return {
                "execution_status": "COMPLETED",
                "messages": [
                    _build_tracking_message("report_data", payload["report_data"])
                ]
            }
        return {
            "execution_status": "COMPLETED",
            "messages": []
        }
    except Exception as e:
        logger.error(f"Error during execute_action_node: {e}", exc_info=True)
        return {
            "execution_status": "FAILED",
            "messages": [
                _build_tracking_message("system_error", str(e))
            ]
        }

async def simulation_node(state: ConversationState) -> Dict[str, Any]:
    """
    Safely simulates purchase consequences by contrasting purchase requests 
    against available sub-ledger balances and total net worth.
    """
    logger.info("Running simulation_node")
    intent = state.extracted_intent or {}
    if state.execution_status != "READY_FOR_SIMULATION" or intent.get("intent_type") != "SIMULATE_PURCHASE":
        return {}
        
    amount_str = intent.get("amount")
    source_name = intent.get("source_name")
    
    if not amount_str:
        err_msg = {"role": "assistant", "content": "Simulation failed: purchase amount is required."}
        return {"execution_status": "FAILED", "messages": [err_msg]}
        
    try:
        purchase_amount = Decimal(str(amount_str))
    except Exception:
        err_msg = {"role": "assistant", "content": "Simulation failed: invalid amount format."}
        return {"execution_status": "FAILED", "messages": [err_msg]}

    from app.tools.validation_tool import get_self_net_worth, get_self_balance
    from app.agents.guardrails import get_source_id_by_name
    from app.utils.formatters import format_indian_currency
    
    # 1. Fetch total net worth
    try:
        net_worth = get_self_net_worth()
    except Exception as e:
        logger.error(f"Failed to query net worth for simulation: {e}")
        net_worth = Decimal("0.00")
        
    # 2. Fetch specific source balance if given
    source_balance = None
    if source_name:
        src_id = get_source_id_by_name(source_name)
        if src_id:
            try:
                source_balance = get_self_balance(src_id)
            except Exception as e:
                logger.error(f"Failed to query source balance for simulation: {e}")
                
    # 3. Build simulation breakdown
    formatted_purchase = format_indian_currency(purchase_amount)
    formatted_net_worth = format_indian_currency(net_worth)
    new_net_worth = net_worth - purchase_amount
    formatted_new_net_worth = format_indian_currency(new_net_worth)
    
    lines = [
        "=== Purchasing Simulation ===",
        f"• Item Cost: {formatted_purchase}",
        f"• Current Total Net Worth: {formatted_net_worth}",
        f"• Projected Net Worth after Purchase: {formatted_new_net_worth}"
    ]
    
    if source_balance is not None:
        formatted_src_bal = format_indian_currency(source_balance)
        new_src_bal = source_balance - purchase_amount
        formatted_new_src_bal = format_indian_currency(new_src_bal)
        lines.append(f"• Source ({source_name}) Current Allocation: {formatted_src_bal}")
        lines.append(f"• Source ({source_name}) Projected Allocation: {formatted_new_src_bal}")
        
    # Budget Boundary Contrast
    if new_net_worth < Decimal("0.00"):
        lines.append("\n⚠️ CRITICAL WARNING: This transaction exceeds your entire net worth boundaries!")
    elif source_balance is not None and new_src_bal < Decimal("0.00"):
        lines.append(f"\n⚠️ WARNING: This transaction exceeds your allocation in '{source_name}'.")
    else:
        lines.append("\n✅ Feasibility Check Passed: Purchase sits safely within current allocations.")
        
    sim_message = {
        "role": "assistant",
        "content": "\n".join(lines)
    }
    
    return {
        "execution_status": "SIMULATED",
        "messages": [sim_message]
    }

async def response_rendering_node(state: ConversationState) -> Dict[str, Any]:
    """
    Uses Gemini to render the final user-facing response from state.
    """
    logger.info("Running response_rendering_node")

    intent = state.extracted_intent or {}
    missing_fields = _get_missing_required_fields(intent)
    tracking_messages = _extract_tracking_messages(state.messages)
    action_result = state.action_result or {}

    if intent.get("intent_type") == "CHECK_BALANCE" and not action_result:
        rendered_response = _build_balance_response(action_result)
        latest_user_message = _get_latest_user_message(state.messages)
        _append_local_chat_history(latest_user_message, rendered_response)
        return {
            "messages": [{"role": "assistant", "content": rendered_response}],
            "execution_status": state.execution_status,
        }

    prompt = _build_response_prompt(state, missing_fields, tracking_messages)

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
            ),
        )

        rendered_response = (response.text or "").strip()
        if not rendered_response:
            rendered_response = _build_fallback_response(state, missing_fields, tracking_messages)
    except Exception as e:
        logger.error(f"Error during response rendering: {e}", exc_info=True)
        rendered_response = _build_fallback_response(state, missing_fields, tracking_messages)

    latest_user_message = _get_latest_user_message(state.messages)
    _append_local_chat_history(latest_user_message, rendered_response)

    return {
        "messages": [{"role": "assistant", "content": rendered_response}],
        "execution_status": state.execution_status,
    }

# -----------------
# LangGraph Workflow Construction
# -----------------
workflow = StateGraph(ConversationState)

# Add Nodes
workflow.add_node("retrieve_memory", retrieve_memory_node)
workflow.add_node("intent_extractor", intent_extractor_node)
workflow.add_node("execute_action", execute_action_node)
workflow.add_node("simulation", simulation_node)
workflow.add_node("response_rendering", response_rendering_node)

# Setup Transitions (Sequential execution incorporating memory retrieval and simulation logic)
workflow.add_edge(START, "retrieve_memory")
workflow.add_edge("retrieve_memory", "intent_extractor")
workflow.add_edge("intent_extractor", "execute_action")
workflow.add_edge("execute_action", "simulation")
workflow.add_edge("simulation", "response_rendering")
workflow.add_edge("response_rendering", END)

# Compile cognitive state graph with SupabaseCheckpointer
from app.core.db import supabase
from app.core.checkpointer import SupabaseCheckpointer

db_checkpointer = SupabaseCheckpointer(supabase)
compiled_graph = workflow.compile(checkpointer=db_checkpointer)
