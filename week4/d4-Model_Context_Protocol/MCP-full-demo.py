import json, time, os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# Simple logger
# --------------------------
def log_event(ev):
    ev["timestamp"] = time.time()
    with open("trace.log", "a") as f:
        f.write(json.dumps(ev) + "\n")

# --------------------------
# Tools
# --------------------------
def add(a, b):
    log_event({
        "role": "tool",
        "tool": "add",
        "arguments": {"a": a, "b": b}
    })
    result = a + b
    log_event({
        "role": "tool_result",
        "tool": "add",
        "output": result
    })
    return result

tools = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    }
}]

# --------------------------
# Model call with tracing
# --------------------------
def call_model(user_msg):
    messages = [
        {"role": "system", "content": "You can use tools."},
        {"role": "user", "content": user_msg}
    ]

    log_event({"role": "user", "content": user_msg})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    msg = resp.choices[0].message
    log_event({
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in (msg.tool_calls or [])],
        "decision": "tool_call" if msg.tool_calls else "response"
    })

    # Execute if it's a tool call
    if msg.tool_calls:
        args_str = msg.tool_calls[0].function.arguments
        args = json.loads(args_str)  # Parse JSON string to dict
        result = add(args["a"], args["b"])
        return f"Tool result: {result}"

    return msg.content

# --------------------------
# Run
# --------------------------
print(call_model("What is 3 + 5?"))