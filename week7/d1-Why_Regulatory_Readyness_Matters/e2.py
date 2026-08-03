def classify_risk_contextual(
    use_case,
    user_type,          # e.g. "child", "public", "internal"
    deployment_region,  # e.g. "EU", "US", "NZ"
    data_sensitivity,   # e.g. "public", "internal", "sensitive", "highly_sensitive"
    autonomy_level,     # e.g. "manual", "assistive", "fully_autonomous"
):
    """
    Still very simple: bump to HIGH_RISK if any strong risk factor is present.
    """
    # Extra risk factors
    if user_type.lower() == "child":
        return "HIGH_RISK"

    if deployment_region.upper() == "EU" and data_sensitivity in ["sensitive", "highly_sensitive"]:
        return "HIGH_RISK"

    if autonomy_level == "fully_autonomous" and data_sensitivity != "public":
        return "HIGH_RISK"

    return "LOW_RISK"


# Example of contextual classifier usage (short demo)
print(
    "chatbot, adult, Asia, sensitive, assistive â†’",
    classify_risk_contextual(
        use_case="chatbot",
        user_type="adult",
        deployment_region="Asia",
        data_sensitivity="sensitive",
        autonomy_level="assistive",
    ),
)