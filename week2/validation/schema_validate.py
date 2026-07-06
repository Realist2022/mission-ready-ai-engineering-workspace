# Import required libraries for JSON handling and validation
import json
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

# 1) Define a strict JSON schema to validate LLM-generated product data
# This schema catches common mistakes LLMs make when generating structured data
PRODUCT_SCHEMA = {
    "type": "object",                           # Must be a JSON object
    "additionalProperties": False,              # No extra fields allowed (strict validation)
    "required": ["name", "price", "currency"], # These three fields are mandatory
    "properties": {
        "name": {"type": "string", "minLength": 1},           # Product name: non-empty string
        "price": {"type": "number", "minimum": 0},            # Price: positive number only
        "currency": {"type": "string", "enum": ["USD", "EUR", "NZD"]}, # Only these currencies allowed
        "color": {"type": "string", "enum": ["black", "white", "blue"]}, # Limited color options
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5} # Max 5 string tags
    }
}

# Create a validator instance using the latest JSON Schema draft with format checking
validator = Draft202012Validator(PRODUCT_SCHEMA, format_checker=FormatChecker())

# 2) Sample LLM output that contains common mistakes (intentionally flawed for demonstration)
# This simulates real-world LLM responses that often have formatting or validation issues
llm_output_text = """
{
  "name": "Nike Run 2024",
  "price": "$89",                 // ERROR: string with currency symbol (should be number)
  "currency": "US$",              // ERROR: not in allowed enum values
  "color": "dark black",          // ERROR: not in allowed color list
  "tags": ["shoes", 123, "run"]   // ERROR: contains non-string item (123)
}
"""

# 3) Helper function to safely parse JSON that might contain JavaScript-style comments
# LLMs sometimes add comments to JSON, which breaks standard JSON parsing
def strip_comments(s: str) -> str:
    lines = []
    for line in s.splitlines():           # Process each line individually
        if "//" in line:                  # If line contains a comment
            line = line[:line.index("//")]  # Remove everything from "//" onwards
        lines.append(line)                # Keep the cleaned line
    return "\n".join(lines)               # Rejoin all lines back into a string

# Attempt to parse the JSON after removing comments
try:
    candidate = json.loads(strip_comments(llm_output_text))  # Parse cleaned JSON
except json.JSONDecodeError as e:
    print("âŒ JSON syntax error:", e)  # Handle malformed JSON
    exit(1)

# 4) Function to validate parsed JSON against our schema and return all errors
def validate_payload(d: dict):
    # Get all validation errors and sort them by field path for consistent output
    errors = sorted(validator.iter_errors(d), key=lambda e: list(e.path))
    return errors

# Run initial validation to see what's wrong
errors = validate_payload(candidate)

# 5) Auto-correction function that tries to fix common LLM mistakes
# This handles typical formatting issues that LLMs often produce
def coerce_fixes(d: dict) -> dict:
    fixed = dict(d)  # Create a copy to avoid modifying the original

    # Fix price: convert "$89" string to 89.0 number
    if isinstance(fixed.get("price"), str):
        # Extract only digits and decimal points from price string
        digits = "".join(ch for ch in fixed["price"] if ch.isdigit() or ch == ".")
        if digits:
            fixed["price"] = float(digits)  # Convert to number

    # Fix currency: map common variations to standard codes
    if fixed.get("currency") == "US$":
        fixed["currency"] = "USD"  # Convert "US$" to proper ISO code

    # Fix color: handle case variations and similar values
    color_map = {"dark black": "black", "Black": "black", "Blue": "blue", "White": "white"}
    if "color" in fixed and fixed["color"] in color_map:
        fixed["color"] = color_map[fixed["color"]]  # Map to allowed value

    # Fix tags: remove non-string items and limit to maximum allowed
    if isinstance(fixed.get("tags"), list):
        # Filter out non-strings (like the number 123) and limit to 5 items
        fixed["tags"] = [t for t in fixed["tags"] if isinstance(t, str)][:5]

    return fixed

# Display initial validation errors if any exist
if errors:
    print("âŒ Invalid initially:")
    for e in errors:
        # Create a readable path to the problematic field (e.g., "tags[1]" or "price")
        loc = ".".join(map(str, e.path)) or "(root)"
        print(f"  - {loc}: {e.message}")

    # Try to auto-fix the issues
    candidate = coerce_fixes(candidate)
    # Re-validate after attempting fixes
    errors = validate_payload(candidate)

# 6) Display final validation results
if errors:
    print("\nâŒ Still invalid after fixes:")
    for e in errors:
        # Show remaining errors that couldn't be auto-fixed
        loc = ".".join(map(str, e.path)) or "(root)"
        print(f"  - {loc}: {e.message}")
else:
    # Success! Show the cleaned and validated JSON
    print("\nâœ… Valid product spec:")
    print(json.dumps(candidate, indent=2))  # Pretty-print the final result