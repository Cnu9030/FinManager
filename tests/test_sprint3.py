import pytest
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.config import settings
from app.core.event_bus import event_bus
from app.agents.conversation_graph import compiled_graph, ConversationState
from app.agents.guardrails import audit_agent, rule_engine_agent
from app.telegram.webhook import router

# -----------------
# Webhook Endpoint Test Setup
# -----------------
app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.mark.asyncio
@patch("app.agents.conversation_graph.genai.Client")
async def test_intent_extractor_node_success(mock_genai_class):
    # Mock Gemini response content
    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"intent_type": "ADD_EXPENSE", "amount": 150.75, "source_name": "Savings", "owner_name": "Self", "description": "Dinner"}'
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_genai_class.return_value = mock_client_instance
    
    # Setup graph input state
    state = ConversationState(messages=[{"role": "user", "content": "I spent 150.75 on Dinner from Savings"}])
    
    # Import node function to test individually
    from app.agents.conversation_graph import intent_extractor_node
    res = await intent_extractor_node(state)
    
    assert res["execution_status"] == "INTENT_EXTRACTED"
    assert res["extracted_intent"]["intent_type"] == "ADD_EXPENSE"
    assert res["extracted_intent"]["amount"] == "150.75"
    assert res["extracted_intent"]["source_name"] == "Savings"

def test_telegram_webhook_unauthorized():
    # Attempt to post a message with an unauthorized sender ID
    payload = {
        "update_id": 99999,
        "message": {
            "message_id": 1,
            "from": {
                "id": 99999999,  # Unauthorized ID (settings.EXPECTED_TELEGRAM_USER_ID is 863445861)
                "is_bot": False,
                "first_name": "Hacker"
            },
            "chat": {
                "id": 12345,
                "type": "private"
            },
            "text": "Check balance"
        }
    }
    
    response = client.post("/telegram/webhook", json=payload)
    assert response.status_code == 401
    assert "Access Denied" in response.json()["detail"]

@patch("app.telegram.webhook.run_conversation_workflow")
def test_telegram_webhook_authorized(mock_run_workflow):
    # Post a message with the authorized sender ID
    payload = {
        "update_id": 10000,
        "message": {
            "message_id": 2,
            "from": {
                "id": settings.EXPECTED_TELEGRAM_USER_ID,
                "is_bot": False,
                "first_name": "Owner"
            },
            "chat": {
                "id": 863445861,
                "type": "private"
            },
            "text": "I spent 50 on Coffee"
        }
    }
    
    response = client.post("/telegram/webhook", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "processing"}
    mock_run_workflow.assert_called_once()

@pytest.mark.asyncio
@patch("app.agents.guardrails.supabase")
@patch("app.agents.guardrails.get_ledger_snapshot")
async def test_audit_agent_anti_duplication(mock_snapshot, mock_supabase):
    # Mocking supabase transactions query response to simulate a duplicate transaction
    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.gte.return_value.execute = MagicMock(
        return_value=MagicMock(data=[{"id": "duplicate-tx-id"}])
    )
    # Mocking get_source_id_by_name helper query
    mock_supabase.table.return_value.select.return_value.ilike.return_value.execute = MagicMock(
        return_value=MagicMock(data=[{"id": "src-uuid-123"}])
    )
    
    payload = {
        "intent": {
            "intent_type": "ADD_EXPENSE",
            "amount": "250.00",
            "source_name": "Cash",
            "description": "Lunch"
        },
        "messages": [],
        "status": "PROCESSING",
        "error": None
    }
    
    await audit_agent.handle_event("intent.extracted", payload)
    
    # Assert that an error is registered in the payload due to deduplication guard triggering
    assert payload["error"] == "Duplicate transaction detected within 60 seconds window."

@pytest.mark.asyncio
@patch("app.agents.guardrails.supabase")
@patch("app.agents.guardrails.get_self_person_id")
async def test_rule_engine_agent_insufficient_balance(mock_get_self, mock_supabase):
    mock_get_self.return_value = "self-person-uuid"
    
    # Mocking get_source_id_by_name helper
    mock_supabase.table.return_value.select.return_value.ilike.return_value.execute = MagicMock(
        return_value=MagicMock(data=[{"id": "src-uuid-123"}])
    )
    # Mocking ownership query balance of 10.00
    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = MagicMock(
        return_value=MagicMock(data=[{"allocated_amount": "10.00"}])
    )
    
    # Requesting an expense of 50.00
    payload = {
        "intent": {
            "intent_type": "ADD_EXPENSE",
            "amount": "50.00",
            "source_name": "Cash",
            "description": "Bus ticket"
        },
        "messages": [],
        "status": "PROCESSING",
        "error": None
    }
    
    await rule_engine_agent.handle_event("intent.extracted", payload)
    
    # Assert that rule engine successfully flags business rule violation
    assert "Business rule violation" in payload["error"]
    assert "fall below zero" in payload["error"]
