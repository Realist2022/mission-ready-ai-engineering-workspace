import time
import random
import numpy as np

def real_inference_simulation(batch_size):
    """Simulate real neural network inference with actual batched operations"""
    
    input_batch = np.random.randn(batch_size, 128)
    
    weights = np.random.randn(128, 64)
    bias = np.random.randn(64)
    
    hidden = np.dot(input_batch, weights) + bias  # Linear transformation
    activated = np.maximum(0, hidden)  # ReLU activation (vectorized)
    
    output_weights = np.random.randn(64, 10)
    output = np.dot(activated, output_weights)
    
    return f"Batch of {batch_size} â†’ output shape: {output.shape}"


print("\nâœ… REAL BATCHING (Efficient):")
for b in [1, 4, 8, 16, 32]:
    start = time.time()
    result = real_inference_simulation(b)
    duration = time.time() - start
    print(f"Batch {b:2d} â†’ {duration:.4f} sec | {result}")