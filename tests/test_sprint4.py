import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from app.agents.conversation_graph import ConversationState, simulation_node
from app.agents.memory_agent import generate_embedding, store_memory, retrieve_memories, retrieve_memory_node
from app.agents.reporting_agent import fetch_category_spending, fetch_monthly_trends, fetch_multi_party_net_worth, reporting_agent

# -----------------
# Task 1: Vector Memory Layer Tests
# -----------------
@patch("app.agents.memory_agent.genai.Client")
def test_generate_embedding(mock_genai_class):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
    mock_client.models.embed_content.return_value = mock_response
    mock_genai_class.return_value = mock_client

    emb = generate_embedding("sample text")
    assert len(emb) == 768
    assert emb[0] == 0.1

@patch("app.agents.memory_agent.generate_embedding")
@patch("app.agents.memory_agent.supabase")
def test_store_memory(mock_supabase, mock_gen_emb):
    mock_gen_emb.return_value = [0.1] * 768
    mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock(
        data=[{"id": "memory-uuid-1"}]
    )

    mem_id = store_memory("Gave Ravi ₹5,000")
    assert mem_id == "memory-uuid-1"

@patch("app.agents.memory_agent.generate_embedding")
@patch("app.agents.memory_agent.supabase")
def test_retrieve_memories(mock_supabase, mock_gen_emb):
    mock_gen_emb.return_value = [0.1] * 768
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(
        data=[{"id": "mem-1", "context_summary": "Spent ₹5,000 last year", "similarity": 0.85}]
    )

    mems = retrieve_memories("How much spent last year?")
    assert len(mems) == 1
    assert mems[0]["context_summary"] == "Spent ₹5,000 last year"

@pytest.mark.asyncio
@patch("app.agents.memory_agent.retrieve_memories")
async def test_retrieve_memory_node(mock_ret_mems):
    mock_ret_mems.return_value = [{"context_summary": "Past transaction context"}]
    state = ConversationState(messages=[{"role": "user", "content": "Tell me about my history"}])

    res = await retrieve_memory_node(state)
    assert len(res["messages"]) == 2
    assert "Past transaction context" in res["messages"][0]["content"]

# -----------------
# Task 2: SQL Reporting Agent Tests
# -----------------
@patch("app.agents.reporting_agent.supabase")
def test_fetch_category_spending_report(mock_supabase):
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(
        data=[{"category": "Food", "total_amount": "1500.50"}]
    )

    report = fetch_category_spending()
    assert "Food" in report
    assert "₹1,500.50" in report

@patch("app.agents.reporting_agent.supabase")
def test_fetch_monthly_trends_report(mock_supabase):
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(
        data=[{"month_date": "2026-06", "total_amount": "5000.00"}]
    )

    report = fetch_monthly_trends()
    assert "2026-06" in report
    assert "₹5,000.00" in report

@patch("app.agents.reporting_agent.supabase")
def test_fetch_multi_party_net_worth_report(mock_supabase):
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(
        data=[{"owner_name": "Ravi", "total_net_worth": "25000.00"}]
    )

    report = fetch_multi_party_net_worth()
    assert "Ravi" in report
    assert "₹25,000.00" in report

# -----------------
# Task 3: Financial Consequence Simulation Workflow Tests
# -----------------
@pytest.mark.asyncio
@patch("app.tools.validation_tool.get_self_net_worth")
@patch("app.tools.validation_tool.get_self_balance")
@patch("app.agents.guardrails.get_source_id_by_name")
async def test_simulation_node_feasibility(mock_get_src, mock_get_bal, mock_get_net_worth):
    # Setup mocks
    mock_get_net_worth.return_value = Decimal("50000.00")
    mock_get_bal.return_value = Decimal("20000.00")
    mock_get_src.return_value = "src-uuid-123"

    state = ConversationState(
        messages=[{"role": "user", "content": "Can I buy a phone for ₹15,000 from Savings?"}],
        extracted_intent={
            "intent_type": "SIMULATE_PURCHASE",
            "amount": "15000.00",
            "source_name": "Savings",
            "description": "phone"
        },
        execution_status="READY_FOR_SIMULATION"
    )

    res = await simulation_node(state)
    assert res["execution_status"] == "SIMULATED"
    assistant_content = res["messages"][-1]["content"]
    assert "Feasibility Check Passed" in assistant_content
    assert "₹15,000.00" in assistant_content
    assert "₹50,000.00" in assistant_content
    assert "₹35,000.00" in assistant_content

@pytest.mark.asyncio
@patch("app.tools.validation_tool.get_self_net_worth")
@patch("app.tools.validation_tool.get_self_balance")
@patch("app.agents.guardrails.get_source_id_by_name")
async def test_simulation_node_exceeds_networth(mock_get_src, mock_get_bal, mock_get_net_worth):
    # Setup mocks showing budget limits exceeded
    mock_get_net_worth.return_value = Decimal("30000.00")
    mock_get_bal.return_value = Decimal("10000.00")
    mock_get_src.return_value = "src-uuid-123"

    state = ConversationState(
        messages=[{"role": "user", "content": "Can I buy a laptop for ₹45,000 from Savings?"}],
        extracted_intent={
            "intent_type": "SIMULATE_PURCHASE",
            "amount": "45000.00",
            "source_name": "Savings",
            "description": "laptop"
        },
        execution_status="READY_FOR_SIMULATION"
    )

    res = await simulation_node(state)
    assistant_content = res["messages"][-1]["content"]
    assert "CRITICAL WARNING" in assistant_content
    assert "exceeds your entire net worth" in assistant_content

