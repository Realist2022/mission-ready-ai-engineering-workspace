import re


def classify_input(text):
    if "@" in text:
        return "PII"
    if any(word in text.lower() for word in ["diagnosis", "medical"]):
        return "PHI"
    return "NON_SENSITIVE"


def classify_input_regex(text):
    """
    Very simple regex-based classifier:
    - Email-like patterns -> PII
    - Words 'diagnosis' or 'medical' -> PHI
    - Otherwise -> NON_SENSITIVE
    """
    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    if re.search(email_pattern, text):
        return "PII"

    if re.search(r"\b(diagnosis|medical)\b", text, flags=re.IGNORECASE):
        return "PHI"

    return "NON_SENSITIVE"


examples = [
    "My email is test@example.com",
    "I received a medical diagnosis",
    "How does an LLM work?",
]

for e in examples:
    print(e, "â†’", classify_input(e), "(simple rules)")
    print("   regex â†’", classify_input_regex(e))