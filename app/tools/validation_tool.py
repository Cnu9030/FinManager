from decimal import Decimal
from app.core.db import supabase

def get_self_person_id() -> str:
    """
    Retrieves the UUID of the self person (is_self = True).
    """
    response = supabase.table("people").select("id").eq("is_self", True).execute()
    if not response.data:
        raise ValueError("No self person configuration found in database.")
    return response.data[0]["id"]

def check_source_exists(source_id: str) -> bool:
    """
    Checks if the source account exists.
    """
    response = supabase.table("sources").select("id").eq("id", source_id).execute()
    return len(response.data) > 0

def get_self_balance(source_id: str) -> Decimal:
    """
    Gets the allocated balance for the self user for a specific source.
    Enforces filtering via the self_person_id.
    """
    self_person_id = get_self_person_id()
    response = supabase.table("ownership")\
        .select("allocated_amount")\
        .eq("source_id", source_id)\
        .eq("owner_id", self_person_id)\
        .execute()
        
    if not response.data:
        return Decimal("0.00")
    return Decimal(str(response.data[0]["allocated_amount"]))

def get_self_net_worth() -> Decimal:
    """
    Aggregates the net worth of the self user across all sources.
    Enforces filtering via the self_person_id.
    """
    self_person_id = get_self_person_id()
    response = supabase.table("ownership")\
        .select("allocated_amount")\
        .eq("owner_id", self_person_id)\
        .execute()
        
    total = Decimal("0.00")
    for row in response.data:
        total += Decimal(str(row["allocated_amount"]))
    return total
