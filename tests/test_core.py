import pytest
import asyncio
from typing import List, Dict, Any
from app.config import Settings
from app.core.event_bus import EventBus
from app.agents.base_agent import BaseAgent

def test_settings_load(monkeypatch):
    """Test configuration settings load correctly from mock environment."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "testkey")
    monkeypatch.setenv("GEMINI_API_KEY", "testgemini")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "testtelegram")
    monkeypatch.setenv("EXPECTED_TELEGRAM_USER_ID", "987654321")
    
    settings = Settings()
    assert settings.SUPABASE_URL == "https://test.supabase.co"
    assert settings.SUPABASE_KEY == "testkey"
    assert settings.GEMINI_API_KEY == "testgemini"
    assert settings.TELEGRAM_BOT_TOKEN == "testtelegram"
    assert settings.EXPECTED_TELEGRAM_USER_ID == 987654321

@pytest.mark.asyncio
async def test_event_bus_pub_sub():
    """Test that event bus correctly delivers events to subscribers."""
    bus = EventBus()
    received_payloads = []

    async def mock_handler(event_name: str, payload: dict):
        received_payloads.append(payload)

    bus.subscribe("test_event", mock_handler)
    await bus.publish("test_event", {"data": "hello"})
    
    assert len(received_payloads) == 1
    assert received_payloads[0]["data"] == "hello"

@pytest.mark.asyncio
async def test_event_bus_error_isolation():
    """Test that one failing subscriber does not halt other subscribers."""
    bus = EventBus()
    successful_runs = []

    async def failing_handler(event_name: str, payload: dict):
        raise ValueError("Simulated subscriber failure")

    async def successful_handler(event_name: str, payload: dict):
        successful_runs.append(payload)

    bus.subscribe("test_event", failing_handler)
    bus.subscribe("test_event", successful_handler)
    
    # This should not raise an exception
    await bus.publish("test_event", {"status": "ok"})
    
    assert len(successful_runs) == 1
    assert successful_runs[0]["status"] == "ok"

def test_base_agent_instantiation():
    """Test that BaseAgent abstract class prevents direct instantiation and enforces protocol."""
    # Direct instantiation should fail
    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore

    # Subclass without required members should fail
    class IncompleteAgent(BaseAgent):
        pass

    with pytest.raises(TypeError):
        IncompleteAgent()  # type: ignore

    # Complete subclass should instantiate fine
    class CorrectAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "TestAgent"
        
        @property
        def subscribes_to(self) -> List[str]:
            return ["event_a"]
            
        @property
        def publishes(self) -> List[str]:
            return ["event_b"]

        async def handle_event(self, event_name: str, payload: Dict[str, Any]) -> None:
            pass

    agent = CorrectAgent()
    assert agent.name == "TestAgent"
    assert agent.subscribes_to == ["event_a"]
    assert agent.publishes == ["event_b"]
