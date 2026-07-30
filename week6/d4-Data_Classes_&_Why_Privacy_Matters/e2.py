import re

def redact_emails_and_phones(text):
    # NZ-style phones: optional +64, then optional 0, then digits with spaces/dashes
    phone_pattern = r"(\+?64\s?0?\s?\d[\d\-\s]{6,}\d|\b0\d[\d\-\s]{6,}\d\b)"
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(phone_pattern, "<PHONE>", text)
    return text

def safe_for_llm(text):
    """Apply all redactions before sending to the LLM."""
    return redact_emails_and_phones(text)

example = "Contact me at john.doe@example.com or +64 21 123 4567 for details."
print(safe_for_llm(example))