import asyncio
import logging
from typing import Callable, Dict, List, Any, Awaitable

logger = logging.getLogger("event_bus")

class EventBus:
    """
    An asynchronous in-memory Publisher-Subscriber Event Bus.
    Allows subscribing to events and publishing event payloads concurrently.
    Protects the broadcast flow from individual subscriber failures.
    """
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[str, Dict[str, Any]], Awaitable[None]]]] = {}

    def subscribe(self, event_name: str, callback: Callable[[str, Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a callback for a specific event name."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        if callback not in self._subscribers[event_name]:
            self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable[[str, Dict[str, Any]], Awaitable[None]]) -> None:
        """Unregister a callback for a specific event name."""
        if event_name in self._subscribers:
            try:
                self._subscribers[event_name].remove(callback)
            except ValueError:
                pass
            if not self._subscribers[event_name]:
                del self._subscribers[event_name]

    async def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Publish an event to all subscribed callbacks concurrently.
        Ensures a single failing subscriber does not collapse the entire broadcast.
        """
        callbacks = self._subscribers.get(event_name, [])
        if not callbacks:
            return

        async def _safe_execute(callback: Callable[[str, Dict[str, Any]], Awaitable[None]]) -> None:
            try:
                await callback(event_name, payload)
            except Exception as e:
                logger.error(
                    f"Error in subscriber {callback.__name__ if hasattr(callback, '__name__') else callback} "
                    f"handling event '{event_name}': {e}", 
                    exc_info=True
                )

        # Execute all callbacks concurrently using asyncio.gather
        await asyncio.gather(*(_safe_execute(cb) for cb in callbacks))

# Global instance of EventBus to be used across the application
event_bus = EventBus()
