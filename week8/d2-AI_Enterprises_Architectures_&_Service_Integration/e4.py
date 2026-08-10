def route_event(event):
    if event["type"] == "payment_failed":
        return "finance_team"
    if event["type"] == "ticket_created":
        return "support_team"
    return "default_queue"

event = {"type": "ticket_created"}
print("Routed to:", route_event(event))