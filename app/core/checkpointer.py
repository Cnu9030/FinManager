import logging
from typing import Any, Dict, Iterator, Optional, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

logger = logging.getLogger("supabase_checkpointer")

class SupabaseCheckpointer(BaseCheckpointSaver):
    """
    Custom state checkpointer for LangGraph hooked directly into Supabase.
    Persists and retrieves workflow thread execution states.
    """
    def __init__(self, supabase_client: Any) -> None:
        super().__init__()
        self.client = supabase_client

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get checkpoint by config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        logger.info(f"Retrieving checkpoint for thread_id={thread_id}, checkpoint_id={checkpoint_id}")

        try:
            query = self.client.table("checkpoints").select("*").eq("thread_id", str(thread_id))
            if checkpoint_id:
                query = query.eq("checkpoint_id", str(checkpoint_id))
            else:
                query = query.order("checkpoint_id", desc=True).limit(1)

            res = query.execute()
            if not res.data:
                return None

            row = res.data[0]
            
            # Reconstruct CheckpointTuple
            parent_config = None
            if row.get("parent_id"):
                parent_config = RunnableConfig(
                    configurable={
                        "thread_id": row["thread_id"],
                        "checkpoint_id": row["parent_id"]
                    }
                )

            return CheckpointTuple(
                config=RunnableConfig(
                    configurable={
                        "thread_id": row["thread_id"],
                        "checkpoint_id": row["checkpoint_id"]
                    }
                ),
                checkpoint=row["checkpoint"],
                metadata=row["metadata"],
                parent_config=parent_config
            )
        except Exception as e:
            logger.error(f"Error in SupabaseCheckpointer.get_tuple: {e}", exc_info=True)
            return None

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        """Store checkpoint and return configuration."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")
        logger.info(f"Storing checkpoint thread_id={thread_id}, checkpoint_id={checkpoint_id}")

        try:
            self.client.table("checkpoints").upsert({
                "thread_id": str(thread_id),
                "checkpoint_id": str(checkpoint_id),
                "checkpoint": checkpoint,
                "metadata": metadata,
                "parent_id": str(parent_id) if parent_id else None
            }).execute()
        except Exception as e:
            logger.error(f"Error in SupabaseCheckpointer.put: {e}", exc_info=True)
            raise e

        return RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id
            }
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the filter criteria."""
        logger.info("Listing checkpoints")
        try:
            query = self.client.table("checkpoints").select("*")
            if config:
                query = query.eq("thread_id", str(config["configurable"]["thread_id"]))
            if before:
                query = query.lt("checkpoint_id", str(before["configurable"]["checkpoint_id"]))
            if limit:
                query = query.limit(limit)

            res = query.execute()
            for row in res.data:
                parent_config = None
                if row.get("parent_id"):
                    parent_config = RunnableConfig(
                        configurable={
                            "thread_id": row["thread_id"],
                            "checkpoint_id": row["parent_id"]
                        }
                    )

                yield CheckpointTuple(
                    config=RunnableConfig(
                        configurable={
                            "thread_id": row["thread_id"],
                            "checkpoint_id": row["checkpoint_id"]
                        }
                    ),
                    checkpoint=row["checkpoint"],
                    metadata=row["metadata"],
                    parent_config=parent_config
                )
        except Exception as e:
            logger.error(f"Error in SupabaseCheckpointer.list: {e}", exc_info=True)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async implementation of get_tuple."""
        # Supabase Python client's execute is sync, so we run in executor or execute directly
        return self.get_tuple(config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        """Async implementation of put."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async implementation of list."""
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Any,
        task_id: str,
    ) -> None:
        """Store intermediate writes (no-op)."""
        pass

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Any,
        task_id: str,
    ) -> None:
        """Async store intermediate writes (no-op)."""
        pass

