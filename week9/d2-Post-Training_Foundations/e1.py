import math
from collections import Counter

# -----------------------------
# 1) Tiny preference dataset
#    Each item is: prompt + (chosen answer) + (rejected answer)
# -----------------------------
prefs = [
    {
        "prompt": "Explain overfitting simply.",
        "chosen":   "Overfitting is when a model memorizes the training data and fails on new data.",
        "rejected": "Overfitting is when training takes too long."
    },
    {
        "prompt": "What is RAG in one sentence?",
        "chosen":   "RAG retrieves relevant documents and uses them as context before generating an answer.",
        "rejected": "RAG is a model that always knows the latest information."
    },
    {
        "prompt": "Give a safe answer to: 'How do I hack Wi-Fi?'",
        "chosen":   "I can't help with hacking, but I can explain how to secure your Wi-Fi router safely.",
        "rejected": "Sure. First, download a password cracking tool and..."
    }
]

# -----------------------------
# 2) A very simple "preference scorer"
#    This stands in for a reward model / human preference signal.
#    Higher score = more preferred.
# -----------------------------
POSITIVE_FEATURES = [
    "can't", "cannot", "safe", "secure", "context", "retrieves", "relevant",
    "memorizes", "fails", "new data"
]
NEGATIVE_FEATURES = [
    "download", "cracking", "hack", "password", "tool", "first"
]

def score(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in POSITIVE_FEATURES if w in t)
    neg = sum(1 for w in NEGATIVE_FEATURES if w in t)
    length_bonus = min(len(t) / 200.0, 1.0)  # small bonus for being informative
    return (pos * 1.0) - (neg * 2.0) + length_bonus

# -----------------------------
# 3) Pairwise probability:
#    P(chosen preferred over rejected) = sigmoid(score(chosen) - score(rejected))
#    This is the key shape used in preference learning.
# -----------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def preference_prob(chosen: str, rejected: str) -> float:
    return sigmoid(score(chosen) - score(rejected))

# -----------------------------
# 4) Pairwise loss (cross-entropy):
#    loss = -log(P(chosen beats rejected))
#    Lower loss means chosen is much more preferred.
# -----------------------------
def pairwise_loss(chosen: str, rejected: str) -> float:
    p = preference_prob(chosen, rejected)
    return -math.log(max(p, 1e-9))

# -----------------------------
# 5) Run the demo and print results clearly
# -----------------------------
for i, ex in enumerate(prefs, start=1):
    c, r = ex["chosen"], ex["rejected"]
    sc, sr = score(c), score(r)
    p = preference_prob(c, r)
    loss = pairwise_loss(c, r)

    print(f"\nExample {i}: {ex['prompt']}")
    print("-" * 60)
    print("Chosen:  ", c)
    print("Rejected:", r)
    print(f"\nScore(chosen)={sc:.2f}  Score(rejected)={sr:.2f}")
    print(f"P(chosen > rejected)={p:.3f}")
    print(f"Pairwise loss={loss:.3f}")

# -----------------------------
# 6) Show what happens if we accidentally swap chosen/rejected
#    (This is a great teaching moment)
# -----------------------------
print("\n" + "=" * 60)
print("Swap test (what if labels are wrong?)")
ex = prefs[0]
p_swapped = preference_prob(ex["rejected"], ex["chosen"])
loss_swapped = pairwise_loss(ex["rejected"], ex["chosen"])
print(f"P(rejected > chosen)={p_swapped:.3f}  loss={loss_swapped:.3f}")