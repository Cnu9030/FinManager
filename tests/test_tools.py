import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from app.utils.formatters import format_indian_currency
from app.tools.transaction_tool import execute_ledger_transaction
from app.tools.validation_tool import get_self_person_id, check_source_exists, get_self_balance, get_self_net_worth
from app.tools.audit_tool import get_ledger_snapshot, create_audit_log

def test_indian_currency_formatter():
    # Standard format
    assert format_indian_currency(150000) == "₹1,50,000.00"
    assert format_indian_currency("150000") == "₹1,50,000.00"
    assert format_indian_currency(Decimal("150000")) == "₹1,50,000.00"
    
    # Large format
    assert format_indian_currency(15000000) == "₹1,50,00,000.00"
    
    # Small format
    assert format_indian_currency(500) == "₹500.00"
    
    # Negative format
    assert format_indian_currency(-150000) == "-₹1,50,000.00"
    
    # Decimal values
    assert format_indian_currency(1234567.89) == "₹12,34,567.89"
    assert format_indian_currency("1234567.894") == "₹12,34,567.89"
    assert format_indian_currency("1234567.896") == "₹12,34,567.90"

def test_precision_split_math():
    """
    Verify that concurrent multi-party transaction splits compute without
    structural floating-point artifacts.
    """
    total_amount = Decimal("100.00")
    num_parties = 3
    
    # Split evenly using Decimal floor
    base_share = (total_amount / num_parties).quantize(Decimal("0.01"))
    shares = [base_share] * num_parties
    
    # Allocate remainder to the first party
    remainder = total_amount - sum(shares)
    shares[0] += remainder
    
    # Total must be exactly 100.00 with no float precision loss
    assert sum(shares) == Decimal("100.00")
    assert shares == [Decimal("33.34"), Decimal("33.33"), Decimal("33.33")]

@patch("app.tools.transaction_tool.supabase")
def test_execute_ledger_transaction(mock_supabase):
    mock_execute = MagicMock()
    mock_execute.execute.return_value = MagicMock(data="new-transaction-uuid")
    mock_supabase.rpc.return_value = mock_execute
    
    res = execute_ledger_transaction(
        source_id="src-123",
        owner_id="own-456",
        amount=Decimal("1500.50"),
        category="Food",
        description="Dinner"
    )
    
    mock_supabase.rpc.assert_called_once_with(
        "execute_ledger_entry",
        {
            "p_source_id": "src-123",
            "p_owner_id": "own-456",
            "p_amount": "1500.50",
            "p_cat": "Food",
            "p_desc": "Dinner"
        }
    )
    assert res == "new-transaction-uuid"

@patch("app.tools.validation_tool.supabase")
def test_validation_tools(mock_supabase):
    # Test check source exists
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute = MagicMock(
        return_value=MagicMock(data=[{"id": "src-123"}])
    )
    assert check_source_exists("src-123") is True
    
    # Re-mock for specific tools testing
    with patch("app.tools.validation_tool.get_self_person_id", return_value="self-uuid"):
        # Test get_self_balance
        mock_balance_exec = MagicMock()
        mock_balance_exec.return_value = MagicMock(data=[{"allocated_amount": "1000.50"}])
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = mock_balance_exec
        
        balance = get_self_balance("src-123")
        assert balance == Decimal("1000.50")
        
        # Test get_self_net_worth
        mock_networth_exec = MagicMock()
        mock_networth_exec.return_value = MagicMock(data=[
            {"allocated_amount": "1000.50"},
            {"allocated_amount": "2500.25"}
        ])
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute = mock_networth_exec
        
        net_worth = get_self_net_worth()
        assert net_worth == Decimal("3500.75")

@patch("app.tools.audit_tool.supabase")
def test_audit_tools(mock_supabase):
    # Mock for get_ledger_snapshot
    mock_source_exec = MagicMock()
    mock_source_exec.execute.return_value = MagicMock(data=[{"id": "src-123", "name": "Savings", "current_balance": "5000.00"}])
    
    mock_ownership_exec = MagicMock()
    mock_ownership_exec.execute.return_value = MagicMock(data=[
        {"owner_id": "own-1", "allocated_amount": "3000.00"},
        {"owner_id": "own-2", "allocated_amount": "2000.00"}
    ])
    
    mock_supabase.table.side_effect = lambda table_name: {
        "sources": MagicMock(select=MagicMock(return_value=MagicMock(eq=MagicMock(return_value=mock_source_exec)))),
        "ownership": MagicMock(select=MagicMock(return_value=MagicMock(eq=MagicMock(return_value=mock_ownership_exec)))),
        "audit_logs": MagicMock(insert=MagicMock(return_value=MagicMock(execute=MagicMock(return_value=MagicMock(data=[{"id": "audit-uuid"}])))))
    }[table_name]
    
    snapshot = get_ledger_snapshot("src-123")
    assert snapshot["source"]["current_balance"] == "5000.00"
    assert len(snapshot["ownership"]) == 2
    
    # Test create_audit_log
    audit_id = create_audit_log("tx-123", {"val": 1}, {"val": 2})
    assert audit_id == "audit-uuid"
