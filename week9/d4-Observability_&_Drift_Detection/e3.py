import numpy as np
from scipy.stats import entropy

baseline = np.random.normal(0, 1, 1000)

def kl_divergence(p, q):
    p_hist, _ = np.histogram(p, bins=50, density=True)
    q_hist, _ = np.histogram(q, bins=50, density=True)
    return entropy(p_hist + 1e-8, q_hist + 1e-8)

def human_review(drift_score):
    if drift_score > 0.2:
        return "Approve adaptation"
    return "Monitor"

for step in range(15):

    if step < 7:
        new_data = np.random.normal(0, 1, 1000)
    else:
        new_data = np.random.normal(2, 1.5, 1000)

    drift_score = kl_divergence(baseline, new_data)
    print(f"Step {step} | Drift Score: {drift_score:.4f}")

    if drift_score > 0.1:
        print("âš  Drift detected â†’ Alert triggered")

        decision = human_review(drift_score)
        print("Human Review:", decision)

        if decision == "Approve adaptation":
            print("Adapting model safely...\n")