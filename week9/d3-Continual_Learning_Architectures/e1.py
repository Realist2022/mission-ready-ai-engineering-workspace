import torch
import torch.nn as nn
import torch.optim as optim
import random

# Simple model: a small neural net that maps 10â€‘dim inputs to 2â€‘class outputs.
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# This creates an Adam optimizer that will update all parameters of `model`,
# and `lr=0.001` sets how big each training step is (the learning rate).

# --- Replay Buffer: stores a limited number of past (x, y) samples for rehearsal.
replay_buffer = []

BUFFER_SIZE = 200

def add_to_buffer(x, y):
    # Add a sample to the buffer, dropping the oldest when full.
    if len(replay_buffer) >= BUFFER_SIZE:
        replay_buffer.pop(0)
    replay_buffer.append((x, y))

def sample_buffer(batch_size):
    # Randomly sample a mini-batch from the replay buffer.
    return random.sample(replay_buffer, min(len(replay_buffer), batch_size))


# --- Task A Training: train the model on the first task and fill the replay buffer.
for _ in range(200):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    preds = model(x)
    loss = loss_fn(preds, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # store samples
    for i in range(32):
        add_to_buffer(x[i], y[i])

print("Finished Task A")

# --- Task B Training with Rehearsal: train on new data while mixing in old samples to reduce forgetting.
for _ in range(200):
    x_new = torch.randn(32, 10) + 2  # shifted distribution
    y_new = torch.randint(0, 2, (32,))
    
    # sample old data
    old_samples = sample_buffer(32)
    
    if old_samples:
        x_old = torch.stack([s[0] for s in old_samples])
        y_old = torch.tensor([s[1] for s in old_samples])
        
        x_combined = torch.cat([x_new, x_old])
        y_combined = torch.cat([y_new, y_old])
    else:
        x_combined, y_combined = x_new, y_new
    
    preds = model(x_combined)
    loss = loss_fn(preds, y_combined)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Finished Task B with rehearsal")