def toy_dpo_update(chosen, rejected, policy_scores, step=0.1):
    """
    Very rough intuition:
    - Increase the score of the chosen response
    - Decrease the score of the rejected response
    This mimics how DPO pushes the policy toward preferred responses.
    """
    new_scores = policy_scores.copy()
    new_scores[chosen] += step
    new_scores[rejected] -= step
    return new_scores

# Two candidate policy responses for a risky prompt:
dpo_responses = [
    "Here is how to do it step by step.",
    "I can't help with that, but I can explain the topic safely."
]

# Pretend the current policy gives them equal log-scores / preference:
policy_scores = [0.0, 0.0]  # index 0 and 1

# Human (or policy preference data) says: response 1 is better than response 0.
chosen_index = 1
rejected_index = 0

updated_scores = toy_dpo_update(chosen_index, rejected_index, policy_scores)

print("Before DPO-like update:", policy_scores)
print("After  DPO-like update: ", updated_scores)
print("Policy would now pick:", dpo_responses[updated_scores.index(max(updated_scores))])
