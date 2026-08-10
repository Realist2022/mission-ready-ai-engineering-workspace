def model_predict(input_data):
    return "Approved"

def generate_explanation(input_data):
    reasons = []
    if input_data["income"] > 50000:
        reasons.append("Income level positively influenced the decision")
    if input_data["debt"] > 25000:
        reasons.append("Debt level negatively influenced the decision")
    if not reasons:
        reasons.append("Decision based on overall profile")
    return reasons

def generate_explanation_text(reasons):
    """Short, user-friendly explanation string."""
    if len(reasons) == 1:
        return reasons[0] + "."
    main = reasons[0]
    others = reasons[1:]
    return main + "; " + "; ".join(others) + "."

def simple_score(input_data):
    """Toy score to show how you might expose confidence."""
    base = 0.5
    if input_data["income"] > 50000:
        base += 0.2
    if input_data["debt"] > 25000:
        base -= 0.1
    return max(0.0, min(1.0, base))

def predict_with_explanation(input_data):
    reasons = generate_explanation(input_data)
    return {
        "result": model_predict(input_data),
        "score": simple_score(input_data),  # fake confidence score
        "explanation": reasons,             # structured reasons (for developers)
        "explanation_text": generate_explanation_text(reasons),  # for end users
        "confidence_note": "Explanation is a simplified summary, not a full model rationale",
        "explanation_quality": "heuristic_rules_v1",
    }

print(predict_with_explanation({"income": 60000, "debt": 30000}))