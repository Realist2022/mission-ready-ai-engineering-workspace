responses = [
    "Here is how to do it step by step.",
    "I can't help with that, but I can explain the topic safely."
]

# Simulated human feedback scores:
# higher = more preferred / more aligned
# In RLHF, a reward model would learn to predict scores like these.
preference_scores = [0.2, 0.9]  # humans prefer the second response

def select_aligned_response(responses, scores):
    best_index = scores.index(max(scores))
    return responses[best_index]

print(select_aligned_response(responses, preference_scores))
