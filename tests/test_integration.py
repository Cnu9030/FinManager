import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
import httpx
from app.main import app
from app.config import settings

# -----------------
# Multi-Turn Integration Verification
# -----------------
@pytest.mark.asyncio
@patch("app.telegram.webhook.send_telegram_response")
@patch("app.agents.conversation_graph.genai.Client")
@patch("app.agents.memory_agent.generate_embedding")
@patch("app.agents.guardrails.get_source_id_by_name")
@patch("app.agents.guardrails.get_person_id_by_name")
@patch("app.agents.guardrails.get_self_person_id")
@patch("app.agents.guardrails.get_self_balance")
@patch("app.core.db.supabase")
async def test_multi_turn_integration_workflow(
    mock_supabase,
    mock_get_self_bal,
    mock_get_self_id,
    mock_get_person_id,
    mock_get_src_id,
    mock_gen_emb,
    mock_genai_class,
    mock_send_telegram
):
    # Setup mocks with valid RFC 4122 UUID strings
    mock_gen_emb.return_value = [0.1] * 768
    mock_get_src_id.return_value = "11111111-1111-1111-1111-111111111111"
    mock_get_person_id.return_value = "22222222-2222-2222-2222-222222222222"
    mock_get_self_id.return_value = "33333333-3333-3333-3333-333333333333"
    mock_get_self_bal.return_value = Decimal("200000.00")

    # Explicitly assign mock_supabase to all imported module instances to bypass import-time reference caching
    import app.agents.guardrails
    import app.agents.memory_agent
    import app.core.db
    app.agents.guardrails.supabase = mock_supabase
    app.agents.memory_agent.supabase = mock_supabase
    app.core.db.supabase = mock_supabase

    # Mock Supabase table transactions and checkpoints queries
    mock_execute = MagicMock(return_value=MagicMock(data=[]))
    mock_supabase.table.return_value.upsert.return_value.execute = mock_execute
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute = mock_execute
    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.gte.return_value.execute = mock_execute
    
    # Ownership balance query
    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = MagicMock(
        return_value=MagicMock(data=[{"allocated_amount": "200000.00"}])
    )



    # Route checkpointer to use the mock supabase client
    from app.agents.conversation_graph import compiled_graph
    compiled_graph.checkpointer.client = mock_supabase

    # Mock Gemini Client and Responses
    mock_gemini_client = MagicMock()
    mock_genai_class.return_value = mock_gemini_client
    
    # We will invoke webhook background workflow helper directly
    from app.telegram.webhook import run_conversation_workflow
    
    # --- TURN 1: Post salary addition transaction ---
    # Setup LLM response to extract ADD_EXPENSE (salary)
    mock_response_turn1 = MagicMock()
    mock_response_turn1.text = '{"intent_type": "ADD_EXPENSE", "amount": 100000.00, "source_name": "HDFC bank", "description": "salary"}'
    mock_gemini_client.models.generate_content.return_value = mock_response_turn1
    
    await run_conversation_workflow(settings.EXPECTED_TELEGRAM_USER_ID, "Received salary of ₹1,00,000 in HDFC bank.")
    
    # Assert Turn 1 response: check if it returns success message containing rupee symbol
    mock_send_telegram.assert_called_once()
    chat_id, response_text = mock_send_telegram.call_args[0]
    assert chat_id == settings.EXPECTED_TELEGRAM_USER_ID
    assert "successfully" in response_text.lower()
    
    # Reset mock for Turn 2
    mock_send_telegram.reset_mock()

    # --- TURN 2: Post allocation transfer transaction ---
    # Setup LLM response to extract sub-ledger ownership allocation
    mock_response_turn2 = MagicMock()
    mock_response_turn2.text = '{"intent_type": "ADD_EXPENSE", "amount": 30000.00, "source_name": "HDFC bank", "owner_name": "Father", "description": "aside"}'
    mock_gemini_client.models.generate_content.return_value = mock_response_turn2
    
    await run_conversation_workflow(settings.EXPECTED_TELEGRAM_USER_ID, "Keep ₹30,000 from my HDFC bank aside for Father.")
    
    # Assert Turn 2 response: check if system updates ownership allocation correctly
    mock_send_telegram.assert_called_once()
    chat_id, response_text = mock_send_telegram.call_args[0]
    assert chat_id == settings.EXPECTED_TELEGRAM_USER_ID
    assert "successfully" in response_text.lower()


# -----------------
# Webhook Security Verification
# -----------------
@pytest.mark.asyncio
@patch("app.core.db.supabase")
async def test_integration_webhook_security(mock_supabase):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Turn 3: Post with unauthorized user ID
        payload = {
            "update_id": 11111,
            "message": {
                "message_id": 99,
                "from": {
                    "id": 99999,  # Unauthorized Telegram ID
                    "is_bot": False,
                    "first_name": "Unauthorized"
                },
                "chat": {
                    "id": 99999,
                    "type": "private"
                },
                "text": "Received salary of ₹1,00,000"
            }
        }
        
        response = await client.post("/telegram/webhook", json=payload)
        
        # Assert that webhook instantly drops the request with HTTP 401 Unauthorized
        assert response.status_code == 401
        assert "Access Denied" in response.json()["detail"]
        
        # Ensure zero changes or queries were triggered to Supabase tables from guardrails
        mock_supabase.table.assert_not_called()
        mock_supabase.rpc.assert_not_called()
