import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Two simple tools
def get_weather(city):
    return f"Sunny, 22Â°C in {city}"

def get_client_details(client_id):
    return f"Client {client_id}: John Doe, Premium member"

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "get_client_details",
            "description": "Get client information from database",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string"}
                },
                "required": ["client_id"]
            }
        }
    }
]

# Ask question
question = input("Ask a question: ")
print(f"Question: {question}")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question}],
    tools=tools
)

message = resp.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    print(f"LLM chose: {func_name}")
    print(f"Arguments: {args}")
    
    # Execute the chosen tool
    if func_name == "get_weather":
        result = get_weather(args["city"])
    elif func_name == "get_client_details":
        result = get_client_details(args["client_id"])
    
    print(f"Result: {result}")