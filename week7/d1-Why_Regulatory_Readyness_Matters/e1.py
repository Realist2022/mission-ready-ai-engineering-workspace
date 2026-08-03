def classify_risk(use_case):
    high_risk = ["medical", "finance", "education"]
    if use_case.lower() in high_risk:
        return "HIGH_RISK"
    return "LOW_RISK"

systems = ["spam filter", "medical", "finance", "chatbot"]

for s in systems:
    print(s, "â†’", classify_risk(s))