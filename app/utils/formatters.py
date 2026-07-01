from decimal import Decimal, ROUND_HALF_UP
from typing import Union

def format_indian_currency(amount: Union[int, float, str, Decimal]) -> str:
    """
    Formats a numerical value into localized Indian Currency format string (e.g., ₹1,50,000.00).
    """
    if amount is None:
        return "₹0.00"
        
    try:
        # Convert to Decimal for precision
        dec_amount = Decimal(str(amount))
    except (ValueError, TypeError):
        # Fallback in case of parsing issues
        return "₹0.00"

    # Handle sign
    is_negative = dec_amount < 0
    dec_amount = abs(dec_amount)

    # Round to 2 decimal places
    dec_amount = dec_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    # Split into integer and fractional parts
    parts = str(dec_amount).split('.')
    int_part = parts[0]
    frac_part = parts[1] if len(parts) > 1 else "00"

    # Format the integer part with Indian grouping (3, then 2, 2, ...)
    if len(int_part) <= 3:
        formatted_int = int_part
    else:
        last_three = int_part[-3:]
        remaining = int_part[:-3]
        
        # Group remaining digits in twos from right to left
        groups = []
        while len(remaining) > 0:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        
        # Reverse the groups because we collected them right-to-left
        groups.reverse()
        
        formatted_int = ",".join(groups) + "," + last_three

    prefix = "-₹" if is_negative else "₹"
    return f"{prefix}{formatted_int}.{frac_part}"
