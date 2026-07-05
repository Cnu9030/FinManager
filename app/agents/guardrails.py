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


def _looks_like_institutional_source(name: Optional[str]) -> bool:
    """Detect source labels that should map to the primary self profile instead of a person."""
    if not name:
        return False

    normalized_name = name.lower()
    institutional_keywords = ["bank", "wallet", "cash", "card", "crypto", "hdfc", "sbi"]
    return any(keyword in normalized_name for keyword in institutional_keywords)


def _create_person_record(name: str) -> str:
    """Insert a new person record and return its generated ID."""
    response = supabase.table("people").insert({
        "name": name,
        "connection": None,
        "is_self": False,
    }).execute()

    if response.data:
        return response.data[0]["id"]

    person_id = get_person_id_by_name(name)
    if person_id:
        return person_id

    raise RuntimeError(f"Failed to create person record for '{name}'.")


def _create_source_record(source_name: str) -> str:
    """Insert a new source record and return its generated ID."""
    source_payload = {
        "name": source_name,
        "current_balance": "0.00",
        "type": "cash",
    }

    try:
        response = supabase.table("sources").insert(source_payload).execute()
    except Exception as exc:
        logger.warning(
            "Source insert with type field failed for '%s'; retrying without type: %s",
            source_name,
            exc,
            exc_info=True,
        )
        response = supabase.table("sources").insert({
            "name": source_name,
            "current_balance": "0.00",
        }).execute()

    if response.data:
        return response.data[0]["id"]

    source_id = get_source_id_by_name(source_name)
    if source_id:
        return source_id

    raise RuntimeError(f"Failed to create source record for '{source_name}'.")


def _create_ownership_record(source_id: str, owner_id: str) -> str:
    """Create the ownership junction record that links a person to a source."""
    response = supabase.table("ownership").insert({
        "source_id": source_id,
        "owner_id": owner_id,
        "allocated_amount": "0.00",
    }).execute()

    if response.data:
        return response.data[0]["id"]

    raise RuntimeError("Failed to create ownership junction record.")

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

        if intent_type == "ADD_SOURCE":
            # Source provisioning is handled by SourceProvisioningAgent.
            return
        
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


class SourceProvisioningAgent(BaseAgent):
    """
    Provisions new sources and their ownership links when users request ADD_SOURCE.
    """
    @property
    def name(self) -> str:
        return "SourceProvisioningAgent"

    @property
    def subscribes_to(self) -> List[str]:
        return ["intent.extracted"]

    @property
    def publishes(self) -> List[str]:
        return []

    async def handle_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        logger.info(f"{self.name} processing event: {event_name}")

        if payload.get("error"):
            return

        intent = payload.get("intent", {})
        if intent.get("intent_type") != "ADD_SOURCE":
            return

        source_name = intent.get("source_name")
        owner_name = intent.get("owner_name")

        if not source_name:
            payload["error"] = "Source name is required to create a new source."
            return

        try:
            owner_id = None

            if owner_name:
                owner_id = get_person_id_by_name(owner_name)
                if not owner_id:
                    logger.info(
                        "Owner '%s' not found; creating a new person record for source provisioning.",
                        owner_name,
                    )
                    owner_id = _create_person_record(owner_name)

            if not owner_id:
                if _looks_like_institutional_source(source_name):
                    owner_id = get_self_person_id()
                else:
                    owner_id = get_person_id_by_name(source_name)
                    if not owner_id:
                        logger.info(
                            "No owner_name provided and source '%s' is not institutional; creating matching person record.",
                            source_name,
                        )
                        owner_id = _create_person_record(source_name)

            existing_source_id = get_source_id_by_name(source_name)
            if existing_source_id:
                payload["error"] = f"Source '{source_name}' already exists."
                return

            source_id = _create_source_record(source_name)
            ownership_id = _create_ownership_record(source_id, owner_id)

            payload["provisioned_source_id"] = source_id
            payload["provisioned_owner_id"] = owner_id
            payload["provisioned_ownership_id"] = ownership_id
            payload["source_name"] = source_name

        except Exception as e:
            logger.error(f"Error in SourceProvisioningAgent provisioning flow: {e}", exc_info=True)
            payload["error"] = f"Source provisioning failed: {str(e)}"

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
source_provisioning_agent = SourceProvisioningAgent()

event_bus.subscribe("intent.extracted", audit_agent.handle_event)
event_bus.subscribe("intent.extracted", rule_engine_agent.handle_event)
event_bus.subscribe("intent.extracted", source_provisioning_agent.handle_event)
