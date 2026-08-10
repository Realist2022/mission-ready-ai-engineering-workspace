def estimate_confidence(prompt: str) -> float:
    text = prompt.lower()

    risky_keywords = ["medical", "diagnosis", "prescription", "finance", "investment", "legal"]
    simple_keywords = ["hello", "hi", "how to", "explain", "example", "tutorial"]

    confidence = 0.5  # baseline

    if any(k in text for k in risky_keywords):
        confidence -= 0.3  

    if any(k in text for k in simple_keywords):
        confidence += 0.3  

    return max(0.0, min(1.0, confidence))

def model_with_confidence(prompt):
    confidence = estimate_confidence(prompt)

    if confidence < 0.6:
        return f"Iâ€™m not confident enough to answer this safely. (confidence={confidence:.2f})"
    return f"Here is my answer. (confidence={confidence:.2f})"

print(model_with_confidence("Medical advice"))
print(model_with_confidence("Explain how a for loop works in Python"))