tools = {
    "get_customer": {"capability": "read"},
    "create_ticket": {"capability": "write"},
}

def select_tool(task_type):
    for tool, meta in tools.items():
        if meta["capability"] == task_type:
            return tool
    return None

print(select_tool("read"))