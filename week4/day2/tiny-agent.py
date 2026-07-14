my_api_key="" # Add your OpenAI API key here


from openai import OpenAI
client = OpenAI(api_key=my_api_key)

def get_weather(city: str):
    # Mock tool to keep demo simple
    return {"city": city, "temp": "22Â°C", "condition": "Sunny"}

def agent(query):
    # PLAN
    plan_prompt = f"Break this into steps and specify which tools to call: '{query}'"
    plan = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": plan_prompt}],
        temperature=0
    ).choices[0].message.content
    print("Plan:", plan)

    # ACT (based on plan - look for tool mentions)
    if "get_weather" in plan.lower() or "choose a weather api" in plan.lower():
        result = get_weather("Auckland")
    else:
        result = {"error": "No tool available"}

    print("Action Result:", result)

    # CHECK
    if "error" in result:
        return "I couldn't complete the task."
    else:
        return f"The weather in {result['city']} is {result['temp']} and {result['condition']}."

print(agent("What's the weather?"))