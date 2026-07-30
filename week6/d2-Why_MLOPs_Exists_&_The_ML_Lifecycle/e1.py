import random

def train_model(seed=None):
    if seed is not None:
        random.seed(seed)

    data = [random.random() for _ in range(5)]
    model_score = sum(data) / len(data)
    return model_score

print("Run 1:", train_model())
print("Run 2:", train_model())

print("With seed:")
print("Run 3:", train_model(seed=42))
print("Run 4:", train_model(seed=42))