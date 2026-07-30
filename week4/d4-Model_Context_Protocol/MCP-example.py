import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def add(a, b):
    print(f"ðŸ”§ Tool called: add({a}, {b})")
    result = a + b
    print(f"âœ… Tool result: {result}")
    return result

tools = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers together",
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

print("ðŸ¤– Asking: What is 5 + 7?")

# Initial conversation
messages = [{"role": "user", "content": "What is 5 + 7?"}]

# First LLM call
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

message = resp.choices[0].message
messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})

if message.tool_calls:
    print("ðŸ“ž LLM wants to call a tool:")
    
    # Execute each tool call
    for tool_call in message.tool_calls:
        print(f"   Tool: {tool_call.function.name}")
        print(f"   Args: {tool_call.function.arguments}")
        
        # Auto-execute the tool
        args = json.loads(tool_call.function.arguments)
        result = add(args["a"], args["b"])
        
        # Add tool result to conversation
        messages.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call.id
        })
    
    # Let LLM process the tool results and give final answer
    print("\nðŸ”„ LLM processing tool results...")
    final_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    print(f"ðŸŽ¯ LLM's final answer: {final_resp.choices[0].message.content}")
else:
    print("ðŸ’¬ LLM responded directly:")
    print(message.content)