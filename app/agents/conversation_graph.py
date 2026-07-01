import json
import logging
from decimal import Decimal
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.core.event_bus import event_bus
from app.agents.memory_agent import retrieve_memory_node

logger = logging.getLogger("conversation_graph")

class ConversationState(BaseModel):
    """
    Pydantic schema tracking the cognitive graph state.
    """
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_intent: Optional[Dict[str, Any]] = None
    execution_status: str = "PENDING"

class IntentExtractionSchema(BaseModel):
    """
    Schema for structured JSON extraction by Gemini.
    """
    intent_type: str = Field(description="Type of intent (e.g., 'ADD_EXPENSE', 'CHECK_BALANCE', 'VIEW_REPORT', 'SIMULATE_PURCHASE')")
    amount: Optional[float] = Field(None, description="Decimal amount of the transaction or purchase if applicable")
    source_name: Optional[str] = Field(None, description="Name of the source account/asset if applicable")
    owner_name: Optional[str] = Field(None, description="Name of the owner or self person if applicable")
    description: Optional[str] = Field(None, description="Description, category, or simulation query of the transaction/action")

async def intent_extractor_node(state: ConversationState) -> Dict[str, Any]:
    """
    Sends conversation history to Gemini 2.5 Flash via google-genai SDK 
    to extract structural intent using structured outputs.
    """
    logger.info("Running intent_extractor_node")
    
    # Format message history for LLM prompt
    formatted_history = []
    for msg in state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted_history.append(f"{role.upper()}: {content}")
        
    prompt = (
        "You are a precise finance tracking agent. Analyze the conversation history "
        "and extract the user's intent. Only populate fields that are explicitly "
        "or strongly implied by the conversation.\n\n"
        "Conversation History:\n" + "\n".join(formatted_history)
    )
    
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
        extracted_data = json.loads(response.text)
        logger.info(f"Gemini raw structured extraction: {extracted_data}")
        
        # Ensure Decimal formatting for amount (e.g. serialize to string or handle appropriately)
        if extracted_data.get("amount") is not None:
            extracted_data["amount"] = str(extracted_data["amount"])
            
        return {
            "extracted_intent": extracted_data,
            "execution_status": "INTENT_EXTRACTED"
        }
    except Exception as e:
        logger.error(f"Error during intent extraction: {e}", exc_info=True)
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
            failed_message = {
                "role": "assistant",
                "content": f"Verification failed: {payload['error']}"
            }
            return {
                "execution_status": "FAILED",
                "messages": state.messages + [failed_message]
            }
            
        # Check if reporting agent returned aggregate SQL data
        if payload.get("report_data"):
            report_message = {
                "role": "assistant",
                "content": payload["report_data"]
            }
            return {
                "execution_status": "COMPLETED",
                "messages": state.messages + [report_message]
            }
            
        success_message = {
            "role": "assistant",
            "content": f"Action executed successfully. Type: {state.extracted_intent.get('intent_type')}"
        }
        return {
            "execution_status": "COMPLETED",
            "messages": state.messages + [success_message]
        }
    except Exception as e:
        logger.error(f"Error during execute_action_node: {e}", exc_info=True)
        error_message = {
            "role": "assistant",
            "content": f"System error occurred: {str(e)}"
        }
        return {
            "execution_status": "FAILED",
            "messages": state.messages + [error_message]
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
        return {"execution_status": "FAILED", "messages": state.messages + [err_msg]}
        
    try:
        purchase_amount = Decimal(str(amount_str))
    except Exception:
        err_msg = {"role": "assistant", "content": "Simulation failed: invalid amount format."}
        return {"execution_status": "FAILED", "messages": state.messages + [err_msg]}

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
        "messages": state.messages + [sim_message]
    }

async def response_rendering_node(state: ConversationState) -> Dict[str, Any]:
    """
    Final node handling formatting or any post-processing required.
    """
    logger.info("Running response_rendering_node")
    return {}

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
