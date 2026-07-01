import logging
from typing import Dict, Any, List
from decimal import Decimal

from app.agents.base_agent import BaseAgent
from app.core.event_bus import event_bus
from app.core.db import supabase
from app.utils.formatters import format_indian_currency

logger = logging.getLogger("reporting_agent")

def fetch_category_spending() -> str:
    """Queries Category spending aggregates from PostgreSQL and formats response."""
    try:
        response = supabase.rpc("get_category_spending_summary").execute()
        records = response.data or []
        if not records:
            return "No spending records found."
            
        lines = ["Category Wise Spending Summary:"]
        for row in records:
            cat = row.get("category", "Unknown")
            amt = Decimal(str(row.get("total_amount", "0.00")))
            formatted_amt = format_indian_currency(amt)
            lines.append(f"- {cat}: {formatted_amt}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error compiling category spending: {e}", exc_info=True)
        return "Failed to compile category spending report."

def fetch_monthly_trends() -> str:
    """Queries Monthly spending aggregates from PostgreSQL and formats response."""
    try:
        response = supabase.rpc("get_monthly_spending_trend").execute()
        records = response.data or []
        if not records:
            return "No monthly spending trends found."
            
        lines = ["Monthly Spending Trends:"]
        for row in records:
            month = row.get("month_date", "Unknown")
            amt = Decimal(str(row.get("total_amount", "0.00")))
            formatted_amt = format_indian_currency(amt)
            lines.append(f"- {month}: {formatted_amt}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error compiling monthly trends: {e}", exc_info=True)
        return "Failed to compile monthly spending trends."

def fetch_multi_party_net_worth() -> str:
    """Queries Multi-party net worth breakdowns from PostgreSQL and formats response."""
    try:
        response = supabase.rpc("get_multi_party_net_worth").execute()
        records = response.data or []
        if not records:
            return "No net worth allocations found."
            
        lines = ["Multi-Party Net Worth Breakdown:"]
        for row in records:
            owner = row.get("owner_name", "Unknown")
            amt = Decimal(str(row.get("total_net_worth", "0.00")))
            formatted_amt = format_indian_currency(amt)
            lines.append(f"- {owner}: {formatted_amt}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error compiling net worth: {e}", exc_info=True)
        return "Failed to compile multi-party net worth report."

class ReportingAgent(BaseAgent):
    """
    ReportingAgent processes read-only reporting and analytical requests
    by calling SQL aggregation queries in Postgres.
    """
    @property
    def name(self) -> str:
        return "ReportingAgent"

    @property
    def subscribes_to(self) -> List[str]:
        return ["intent.extracted", "report.requested"]

    @property
    def publishes(self) -> List[str]:
        return []

    async def handle_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        logger.info(f"{self.name} processing event: {event_name}")
        intent = payload.get("intent", {})
        intent_type = intent.get("intent_type")
        
        if intent_type != "VIEW_REPORT":
            return
            
        desc = intent.get("description", "") or ""
        desc_upper = desc.upper()
        
        # Decide which aggregation view to call based on user descriptive instructions
        if "NET" in desc_upper or "OWNER" in desc_upper or "WORTH" in desc_upper:
            report_content = fetch_multi_party_net_worth()
        elif "TREND" in desc_upper or "MONTH" in desc_upper:
            report_content = fetch_monthly_trends()
        else:
            # Default to category spending report
            report_content = fetch_category_spending()
            
        payload["report_data"] = report_content
        logger.info("Report compiled successfully and appended to payload")

# Instantiate and wire reporting agent to the Event Bus
reporting_agent = ReportingAgent()
event_bus.subscribe("intent.extracted", reporting_agent.handle_event)
event_bus.subscribe("report.requested", reporting_agent.handle_event)
