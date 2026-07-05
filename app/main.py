import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.telegram.webhook import router as telegram_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup registration of decoupled background agents 
    on the event bus subscription loops.
    """
    logger.info("Initializing application and registering background agents...")
    
    # Import modules to execute their event bus subscription registration decorators
    import app.agents.guardrails
    import app.agents.ledger_persistence_agent
    import app.agents.reporting_agent
    
    logger.info("Event Bus structural agents registered successfully.")
    yield
    logger.info("Shutting down application orchestrator.")

app = FastAPI(
    title="Finance Agent Cognitive Platform",
    description="Sprint 5 Hybrid LangGraph + Event Bus Production Server",
    version="1.0.0",
    lifespan=lifespan
)

# Mount secured Telegram webhook endpoint router
app.include_router(telegram_router)

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify system status."""
    return {
        "status": "healthy",
        "service": "Finance Agent Platform",
        "engine": "LangGraph + Event Bus"
    }
