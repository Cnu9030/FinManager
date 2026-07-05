import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List, Optional

from app.agents.base_agent import BaseAgent
from app.agents.guardrails import get_person_id_by_name, get_source_id_by_name
from app.core.event_bus import event_bus
from app.core.db import supabase
from app.tools.validation_tool import get_self_person_id

logger = logging.getLogger("ledger_persistence_agent")


def _normalize_amount(amount_value: Any) -> Decimal:
    """Coerce transaction amount into a Decimal rounded for the database schema."""
    amount_decimal = Decimal(str(amount_value))
    return amount_decimal.quantize(Decimal("0.01"))


def _resolve_owner_id(owner_name: Optional[str]) -> str:
    """Resolve an owner to a person ID, falling back to the self profile."""
    if owner_name:
        owner_id = get_person_id_by_name(owner_name)
        if owner_id:
            return owner_id

    return get_self_person_id()


def _build_transaction_row(intent: Dict[str, Any], source_id: str) -> Dict[str, Any]:
    intent_type = intent.get("intent_type") or "UNKNOWN"
    description = (intent.get("description") or intent_type.replace("_", " ").title()).strip()
    category = description or intent_type.replace("_", " ").title()
    amount = _normalize_amount(intent.get("amount"))

    if intent_type == "ADD_EXPENSE":
        amount = -abs(amount)
    elif intent_type == "RECORD_INCOME":
        amount = abs(amount)

    return {
        "source_id": source_id,
        "amount": str(amount),
        "category": category,
        "description": description,
    }


class LedgerPersistenceAgent(BaseAgent):
    """
    Persists completed money movement intents into the transactions table.
    """

    @property
    def name(self) -> str:
        return "ledger_persistence_agent"

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

        intent = payload.get("intent", {}) or {}
        intent_type = intent.get("intent_type")
        if intent_type not in {"RECORD_INCOME", "ADD_EXPENSE"}:
            return

        amount_value = intent.get("amount")
        source_name = intent.get("source_name")
        owner_name = intent.get("owner_name")

        if amount_value in (None, "", [], {}):
            payload["error"] = "Database execution failed: transaction amount is required."
            return

        if not source_name:
            payload["error"] = "Database execution failed: source name is required."
            return

        try:
            source_id = get_source_id_by_name(source_name)
            if not source_id:
                payload["error"] = f"Database execution failed: Source account '{source_name}' does not exist."
                return

            owner_id = _resolve_owner_id(owner_name)
            transaction_row = _build_transaction_row(intent, source_id)

            logger.info(
                "Persisting ledger transaction for intent_type=%s, source=%s, owner=%s",
                intent_type,
                source_name,
                owner_id,
            )

            response = await asyncio.to_thread(
                lambda: supabase.table("transactions").insert(transaction_row).execute()
            )

            if not response.data:
                raise RuntimeError("Insert returned no rows.")

            payload["ledger_transaction_id"] = response.data[0].get("id")
            payload["ledger_persistence_status"] = "COMPLETED"
            payload["ledger_owner_id"] = owner_id
            payload["ledger_source_id"] = source_id

        except (InvalidOperation, TypeError, ValueError) as e:
            logger.error(f"Invalid transaction payload for ledger persistence: {e}", exc_info=True)
            payload["error"] = f"Database execution failed: {str(e)}"
        except Exception as e:
            logger.error(f"Ledger persistence failed: {e}", exc_info=True)
            payload["error"] = f"Database execution failed: {str(e)}"


ledger_persistence_agent = LedgerPersistenceAgent()
event_bus.subscribe("intent.extracted", ledger_persistence_agent.handle_event)