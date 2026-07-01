import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional

from app.agents.base_agent import BaseAgent
from app.core.event_bus import event_bus
from app.core.db import supabase
from app.tools.audit_tool import get_ledger_snapshot
from app.tools.validation_tool import get_self_person_id, get_self_balance

logger = logging.getLogger("guardrails")

def get_source_id_by_name(name: str) -> Optional[str]:
    """Helper to retrieve source ID by name (case-insensitive)."""
    if not name:
        return None
    try:
        response = supabase.table("sources").select("id").ilike("name", name).execute()
        if response.data:
            return response.data[0]["id"]
    except Exception as e:
        logger.error(f"Error querying source by name: {e}")
    return None

def get_person_id_by_name(name: str) -> Optional[str]:
    """Helper to retrieve person ID by name (case-insensitive)."""
    if not name:
        return None
    try:
        response = supabase.table("people").select("id").ilike("name", name).execute()
        if response.data:
            return response.data[0]["id"]
    except Exception as e:
        logger.error(f"Error querying person by name: {e}")
    return None

class AuditAgent(BaseAgent):
    """
    Subscribes to transaction requests to perform anti-duplication checks
    and takes database/ledger state snapshots.
    """
    @property
    def name(self) -> str:
        return "AuditAgent"

    @property
    def subscribes_to(self) -> List[str]:
        return ["intent.extracted"]

    @property
    def publishes(self) -> List[str]:
        return []

    async def handle_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        logger.info(f"{self.name} processing event: {event_name}")
        intent = payload.get("intent", {})
        intent_type = intent.get("intent_type")
        
        # We only audit transaction creation intents, e.g., ADD_EXPENSE
        if intent_type != "ADD_EXPENSE":
            return
            
        amount_str = intent.get("amount")
        source_name = intent.get("source_name")
        description = intent.get("description")
        
        if not amount_str or not source_name or not description:
            # Insufficient information to perform deduplication check
            return
            
        try:
            amount = Decimal(amount_str)
        except Exception:
            payload["error"] = "Invalid amount format in intent."
            return

        source_id = get_source_id_by_name(source_name)
        if not source_id:
            payload["error"] = f"Source account '{source_name}' does not exist."
            return

        # Perform anti-duplication guard: query transactions in the last 60 seconds
        time_boundary = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        try:
            # Query matching transaction amounts, descriptions, and sources in the sliding window
            response = supabase.table("transactions")\
                .select("id")\
                .eq("amount", str(amount))\
                .eq("description", description)\
                .eq("source_id", source_id)\
                .gte("created_at", time_boundary)\
                .execute()
                
            if response.data:
                payload["error"] = "Duplicate transaction detected within 60 seconds window."
                logger.warning(f"Duplicate transaction check failed for source: {source_id}, amount: {amount}")
                return
                
            # Log the before state using audit_tool snapshot helper
            before_snapshot = get_ledger_snapshot(source_id)
            logger.info(f"Audit log before-state snapshot captured for source {source_name}: {before_snapshot}")
            payload["before_state"] = before_snapshot
            
        except Exception as e:
            logger.error(f"Error in AuditAgent duplication check: {e}", exc_info=True)
            payload["error"] = f"Audit agent verification error: {str(e)}"

class RuleEngineAgent(BaseAgent):
    """
    Subscribes to verification passes to check database constraints
    and prevents sub-ledger allocations from falling below zero.
    """
    @property
    def name(self) -> str:
        return "RuleEngineAgent"

    @property
    def subscribes_to(self) -> List[str]:
        return ["intent.extracted"]

    @property
    def publishes(self) -> List[str]:
        return []

    async def handle_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        logger.info(f"{self.name} processing event: {event_name}")
        # If another agent (e.g. AuditAgent) already marked an error, skip further execution
        if payload.get("error"):
            return
            
        intent = payload.get("intent", {})
        intent_type = intent.get("intent_type")
        
        # Check rule boundaries for transactions that reduce balance (like ADD_EXPENSE)
        if intent_type != "ADD_EXPENSE":
            return
            
        amount_str = intent.get("amount")
        source_name = intent.get("source_name")
        owner_name = intent.get("owner_name")
        
        if not amount_str or not source_name:
            return
            
        try:
            amount = Decimal(amount_str)
        except Exception:
            return

        source_id = get_source_id_by_name(source_name)
        if not source_id:
            return

        # Determine owner ID (defaults to "self" person)
        owner_id = None
        if owner_name:
            owner_id = get_person_id_by_name(owner_name)
            
        if not owner_id:
            try:
                owner_id = get_self_person_id()
            except Exception as e:
                payload["error"] = f"Failed to retrieve self person: {str(e)}"
                return

        # Query allocated amount for owner and source
        try:
            response = supabase.table("ownership")\
                .select("allocated_amount")\
                .eq("source_id", source_id)\
                .eq("owner_id", owner_id)\
                .execute()
                
            current_allocation = Decimal("0.00")
            if response.data:
                current_allocation = Decimal(str(response.data[0]["allocated_amount"]))
                
            # Perform check: cannot fall below zero boundary
            if current_allocation - amount < Decimal("0.00"):
                payload["error"] = (
                    f"Business rule violation: transaction of {amount} would cause "
                    f"allocated balance to fall below zero (Current: {current_allocation})."
                )
                logger.warning(payload["error"])
                
        except Exception as e:
            logger.error(f"Error in RuleEngineAgent rule check: {e}", exc_info=True)
            payload["error"] = f"Rule engine verification error: {str(e)}"

# Instantiate and wire background agents to the Event Bus
audit_agent = AuditAgent()
rule_engine_agent = RuleEngineAgent()

event_bus.subscribe("intent.extracted", audit_agent.handle_event)
event_bus.subscribe("intent.extracted", rule_engine_agent.handle_event)
