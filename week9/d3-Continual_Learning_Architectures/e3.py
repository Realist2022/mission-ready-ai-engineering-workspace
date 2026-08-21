import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy

# Simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# --- Baseline data distribution ---
baseline_data = np.random.normal(0, 1, (1000, 10))

def kl_divergence(p, q):
    p_hist, _ = np.histogram(p, bins=50, density=True)
    q_hist, _ = np.histogram(q, bins=50, density=True)

    return entropy(p_hist + 1e-8, q_hist + 1e-8)


# --- Simulate incoming data stream ---
for step in range(20):

    # introduce drift after step 10
    if step < 10:
        new_data = np.random.normal(0, 1, (200, 10))
    else:
        new_data = np.random.normal(2, 1.5, (200, 10))

    # --- Drift detection ---
    drift_score = kl_divergence(baseline_data.flatten(), new_data.flatten())

    print(f"Step {step} | Drift Score: {drift_score:.4f}")

    if drift_score > 0.1:
        print("Drift detected â†’ adapting model")

        x = torch.tensor(new_data, dtype=torch.float32)
        y = torch.randint(0, 2, (200,))

        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()   # reset gradients so we don't mix with the previous step
        loss.backward()         # compute how each parameter should change (gradients)
        optimizer.step()        # apply those changes to actually update the model