from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseAgent(ABC):
    """
    Abstract Base Class defining the contract for all derived agents.
    Every agent must specify its name, subscribe/publish configurations, 
    and implement an asynchronous event handler.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The identifier name of the agent."""
        pass

    @property
    @abstractmethod
    def subscribes_to(self) -> List[str]:
        """List of event names that this agent subscribes to."""
        pass

    @property
    @abstractmethod
    def publishes(self) -> List[str]:
        """List of event names that this agent publishes."""
        pass

    @abstractmethod
    async def handle_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Asynchronously process an incoming event.
        
        Args:
            event_name: The name of the event being broadcasted.
            payload: A dictionary containing event-specific data.
        """
        pass
