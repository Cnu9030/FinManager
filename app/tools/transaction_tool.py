from decimal import Decimal
from app.core.db import supabase

def execute_ledger_transaction(
    source_id: str,
    owner_id: str,
    amount: Decimal,
    category: str,
    description: str
) -> str:
    """
    Triggers the Supabase RPC routine 'execute_ledger_entry' to process
    a ledger transaction deterministically.
    """
    if not isinstance(amount, Decimal):
        raise TypeError("amount must be a decimal.Decimal instance")
        
    response = supabase.rpc(
        "execute_ledger_entry",
        {
            "p_source_id": source_id,
            "p_owner_id": owner_id,
            "p_amount": str(amount),  # Serialize precisely as string to preserve decimal precision
            "p_cat": category,
            "p_desc": description
        }
    ).execute()
    
    return response.data
