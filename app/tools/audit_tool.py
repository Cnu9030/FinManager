from decimal import Decimal
from app.core.db import supabase

def get_ledger_snapshot(source_id: str) -> dict:
    """
    Captures a snapshot dictionary of the ledger state (source balance and ownership allocation)
    for a specific source account.
    """
    # Get source state
    source_res = supabase.table("sources").select("id, name, current_balance").eq("id", source_id).execute()
    if not source_res.data:
        return {}
        
    source_data = source_res.data[0]
    
    # Get ownership allocations for this source
    ownership_res = supabase.table("ownership").select("owner_id, allocated_amount").eq("source_id", source_id).execute()
    
    # Return serializable structure
    return {
        "source": {
            "id": source_data["id"],
            "name": source_data["name"],
            "current_balance": str(source_data["current_balance"])
        },
        "ownership": [
            {
                "owner_id": row["owner_id"],
                "allocated_amount": str(row["allocated_amount"])
            }
            for row in ownership_res.data
        ]
    }

def create_audit_log(transaction_id: str, before_state: dict, after_state: dict) -> str:
    """
    Inserts a record into the audit_logs table.
    """
    response = supabase.table("audit_logs").insert({
        "transaction_id": transaction_id,
        "before_state": before_state,
        "after_state": after_state
    }).execute()
    
    if not response.data:
        raise RuntimeError("Failed to create audit log")
        
    return response.data[0]["id"]
