import pandas as pd

data = pd.DataFrame({
    "group": ["A"] * 80 + ["B"] * 20,
    "approved": [1]*60 + [0]*20 + [1]*5 + [0]*15
})

# Raw approval rates by group
approval_rates = data.groupby("group")["approved"].mean()

print("Approval rates by group:")
print(approval_rates)

# Add some simple bias diagnostics
group_counts = data["group"].value_counts()
approved_counts = data.groupby("group")["approved"].sum()

print("\nSample sizes by group:")
print(group_counts)

print("\nNumber approved by group:")
print(approved_counts)

# Measure disparity: ratio of minority group approval rate to majority group approval rate
rate_A = approval_rates["A"]
rate_B = approval_rates["B"]
disparity_ratio = rate_B / rate_A if rate_A > 0 else 0.0

print("\nDisparity analysis (simple):")
print(f"Group A approval rate: {rate_A:.2%}")
print(f"Group B approval rate: {rate_B:.2%}")
print(f"Disparity ratio (B / A): {disparity_ratio:.2f}")

# Very simple flag: if B is less than 80% of A, highlight potential bias
if disparity_ratio < 0.8:
    print("âš  Potential bias: Group B has a substantially lower approval rate than Group A.")