import torch
import torch.nn as nn
import torch.optim as optim

# Simple model: small neural net for a 2-class classification toy task.
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Train Task A ---
# Goal: train the model on the first task and reach a "good" set of parameters.
for _ in range(200):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    preds = model(x)
    loss = loss_fn(preds, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Finished Task A")

# Save old parameters
# Goal: keep a snapshot of the Task A parameters to protect them later.
old_params = {n: p.clone().detach() for n, p in model.named_parameters()}


# --- Estimate Fisher Information ---
# Goal: estimate how "important" each parameter was for Task A (EWC uses this).
# 'fisher' stores, for each parameter, a number (same shape as the parameter)
# that measures how sensitive the Task A loss was to that parameter: bigger
# values = more important for Task A, so EWC will penalize changing them more.
fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}


for _ in range(100):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    preds = model(x)
    loss = loss_fn(preds, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Accumulate squared gradients as a proxy for Fisher information.
    for n, p in model.named_parameters():
        fisher[n] += p.grad ** 2

for n in fisher:
    fisher[n] /= 100


# --- Train Task B with EWC ---
# Goal: learn a new task while adding a penalty if parameters move too far
# from their Task A values, weighted by their Fisher importance (EWC).
LAMBDA = 10

for _ in range(200):
    x = torch.randn(32, 10) + 2  # shifted distribution for Task B
    y = torch.randint(0, 2, (32,))
    
    preds = model(x)
    loss = loss_fn(preds, y)
    
    # EWC penalty: large if important params move far from old_params.
    ewc_penalty = 0
    for n, p in model.named_parameters():
        ewc_penalty += (fisher[n] * (p - old_params[n])**2).sum()
    
    total_loss = loss + LAMBDA * ewc_penalty
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print("Finished Task B with EWC")