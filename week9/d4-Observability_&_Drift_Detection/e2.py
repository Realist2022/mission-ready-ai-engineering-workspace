import random

# Global "model quality" score we want to improve over time based on feedback.
knowledge_score = 0.5

def ai_response(query):
    
    global knowledge_score
    quality = knowledge_score + random.uniform(-0.2, 0.2)  # add some noise
    return max(0, min(1, quality))  # clamp between 0 and 1

def collect_feedback(response_quality):
    if response_quality > 0.6:
        return 1
    else:
        return 0

def update_model(feedback):
    global knowledge_score
    knowledge_score += 0.05 * (feedback - 0.5)
    knowledge_score = max(0, min(1, knowledge_score))

# Simulate a feedback loop over multiple interactions.
for step in range(20):
    response_quality = ai_response("Explain observability")
    feedback = collect_feedback(response_quality)
    update_model(feedback)

    print(
        f"Step {step} | "
        f"Response Quality: {response_quality:.2f} | "
        f"Feedback: {feedback} | "
        f"Model Score: {knowledge_score:.2f}"
    )