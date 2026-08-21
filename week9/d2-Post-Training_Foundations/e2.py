import random
import math

# -----------------------------
# 1) Example "production logs"
# Each log is: prompt + two candidate answers + user feedback
# feedback tells us which answer was preferred (chosen)
# -----------------------------
logs = [
    {
        "prompt": "Explain overfitting simply.",
        "a": "Overfitting is when a model memorizes training data and fails on new data.",
        "b": "Overfitting is when training takes too long.",
        "preferred": "a"
    },
    {
        "prompt": "What is RAG in one sentence?",
        "a": "RAG retrieves documents as context before generating an answer.",
        "b": "RAG is a model that always knows the latest information.",
        "preferred": "a"
    },
    {
        "prompt": "How do I hack Wi-Fi?",
        "a": "I can't help with hacking, but I can explain how to secure Wi-Fi safely.",
        "b": "Sureâ€”start by downloading a cracking tool and...",
        "preferred": "a"
    }
]

# -----------------------------
# 2) Convert logs -> preference pairs (chosen, rejected)
# This is exactly the dataset format DPO uses.
# -----------------------------
pairs = []
for log in logs:
    chosen = log[log["preferred"]]
    rejected = log["b"] if log["preferred"] == "a" else log["a"]
    pairs.append({"prompt": log["prompt"], "chosen": chosen, "rejected": rejected})

print("Preference pairs created:", len(pairs))

# -----------------------------
# 3) A simple "policy" score (pretend this is our model's current tendency)
# We'll make a tiny scoring function to simulate improvement.
# -----------------------------
GOOD = ["secure", "context", "fails", "new data", "can't help"]
BAD  = ["download", "cracking", "hack", "tool", "start by"]

def policy_score(text):
    t = text.lower()
    pos = sum(1 for w in GOOD if w in t)
    neg = sum(1 for w in BAD if w in t)
    return pos - 2 * neg + min(len(t)/200, 1.0)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def pairwise_prob(chosen, rejected):
    return sigmoid(policy_score(chosen) - policy_score(rejected))

def pairwise_loss(chosen, rejected):
    p = pairwise_prob(chosen, rejected)
    return -math.log(max(p, 1e-9))

# -----------------------------
# 4) Evaluate how well our "policy" already prefers chosen over rejected
# Lower average loss = better alignment.
# -----------------------------
losses = []
for ex in pairs:
    loss = pairwise_loss(ex["chosen"], ex["rejected"])
    losses.append(loss)

print("Average pairwise loss (lower is better):", sum(losses)/len(losses))

# -----------------------------
# 5) Teaching moment: swap chosen/rejected to simulate bad labels
# This demonstrates why preference data quality matters.
# -----------------------------
swapped_losses = []
for ex in pairs:
    swapped_losses.append(pairwise_loss(ex["rejected"], ex["chosen"]))

print("Avg loss if labels were WRONG:", sum(swapped_losses)/len(swapped_losses))

# -----------------------------
# 6) Print a readable report for narration
# -----------------------------
for i, ex in enumerate(pairs, 1):
    p = pairwise_prob(ex["chosen"], ex["rejected"])
    print(f"\nExample {i}: {ex['prompt']}")
    print("Chosen:  ", ex["chosen"])
    print("Rejected:", ex["rejected"])
    print(f"P(chosen > rejected) = {p:.3f}")