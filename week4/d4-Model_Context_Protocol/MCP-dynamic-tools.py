import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Dynamic tool registry
TOOL_REGISTRY = {}
TOOL_DEFINITIONS = []

def register_tool(name, description, params, func):
    """Dynamically register a new tool"""
    TOOL_REGISTRY[name] = func
    TOOL_DEFINITIONS.append({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params
        }
    })
    print(f"âœ… Registered tool: {name}")

# Register tools dynamically
register_tool(
    "calculate_tax",
    "Calculate tax for an amount",
    {
        "type": "object",
        "properties": {"amount": {"type": "number"}},
        "required": ["amount"]
    },
    lambda amount: amount * 0.1
)

register_tool(
    "check_inventory",
    "Check if item is in stock",
    {
        "type": "object", 
        "properties": {"item": {"type": "string"}},
        "required": ["item"]
    },
    lambda item: f"{item}: In stock (5 units)"
)

# Show available tools
print(f"Available tools: {list(TOOL_REGISTRY.keys())}")

question = input("Ask a question: ")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question}],
    tools=TOOL_DEFINITIONS
)

message = resp.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    print(f"LLM chose: {func_name}")
    
    # Dynamic execution
    result = TOOL_REGISTRY[func_name](**args)
    print(f"Result: {result}")
else:
    print("LLM Response:", message.content)