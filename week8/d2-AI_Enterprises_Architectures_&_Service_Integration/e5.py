tool_registry = {
    "get_customer": {
        "description": "Fetch customer profile",
        "input_schema": {
            "customer_id": "string"
        }
    },
    "create_ticket": {
        "description": "Create a support ticket",
        "input_schema": {
            "title": "string",
            "priority": "low|medium|high"
        }
    }
}

def list_tools():
    return tool_registry.keys()

print("Available tools:", list_tools())