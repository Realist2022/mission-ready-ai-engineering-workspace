import pandas as pd
# New example: loan approval by age_group
data = pd.DataFrame({
    # 50 young applicants, 50 senior applicants
    "age_group": ["young"] * 50 + ["senior"] * 50,
    # young: 40 approved, 10 rejected
    # senior: 25 approved, 25 rejected
    "approved": [1]*40 + [0]*10 + [1]*25 + [0]*25,
})

# Approval rate by age_group
approval_rates = data.groupby("age_group")["approved"].mean()
print("Approval rates by age_group:")
print(approval_rates)

# Demographic parity gap: difference in approval rates
parity_gap = approval_rates["young"] - approval_rates["senior"]
print("Demographic parity gap (young - senior):", parity_gap)