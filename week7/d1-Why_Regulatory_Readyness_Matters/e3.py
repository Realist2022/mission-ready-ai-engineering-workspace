import datetime

audit_log = []

def log_change(component, version, reason, owner, user):
    audit_log.append({
        "time": datetime.datetime.utcnow().isoformat(),
        "component": component,
        "version": version,
        "reason": reason,
        "owner": owner,
        "user": user,  # who actually made the change
    })

def get_changes_for_component(component):
    return [e for e in audit_log if e["component"] == component]

log_change("prompt", "v1.2", "Improve refusal clarity", "product_owner", user="alice")
log_change("model", "gpt-4o-mini", "Latency optimization", "ml_lead", user="bob")
log_change("prompt", "v1.3", "Update safety instructions", "policy_lead", user="carol")

# Minimal output: full log + a short summary for one component
for entry in audit_log:
    print(entry)

prompt_changes = get_changes_for_component("prompt")
print("Total prompt changes:", len(prompt_changes))