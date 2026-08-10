interaction_log = []

def log_interaction(prompt, context, response, user_feedback=None):
    """Log the interaction details."""
    interaction_log.append({
        "prompt": prompt,
        "context_used": context,
        "response": response,
        "feedback": user_feedback
    })

def summarize_log(log):
    """Very small summary to show how logged data can be used."""
    total = len(log)
    positive = sum(1 for e in log if e["feedback"] == "ðŸ‘")
    negative = sum(1 for e in log if e["feedback"] == "ðŸ‘Ž")
    return {
        "total_interactions": total,
        "positive_feedback": positive,
        "negative_feedback": negative,
    }

# First interaction
log_interaction(
    prompt="What is our refund policy?",
    context="Refund policy updated March 2024...",
    response="Our refund policy allows returns within 30 days.",
    user_feedback="ðŸ‘"
)

# Second interaction
log_interaction(
    prompt="What is our shipping policy?",
    context="Shipping policy updated Jan 2024...",
    response="We offer standard and express shipping options.",
    user_feedback="ðŸ‘Ž"
)

print(interaction_log)
print(summarize_log(interaction_log))