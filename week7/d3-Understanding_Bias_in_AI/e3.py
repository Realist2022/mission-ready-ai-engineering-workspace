import pandas as pd

def model_decision(row):
    # simple rule-based "model"
    if row["income"] > 40000 and row["group"] == "A":
        return 1
    return 0

person = {"income": 50000, "group": "A"}
counterfactual = {"income": 50000, "group": "B"}

print("Original decision:", model_decision(person))
print("Counterfactual decision:", model_decision(counterfactual))