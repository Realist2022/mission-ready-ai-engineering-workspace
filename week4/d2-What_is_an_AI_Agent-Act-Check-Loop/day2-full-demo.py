my_api_key = "" # Add your OpenAI API key here
"""
Single-Agent Demo (Plan â†’ Act â†’ Check) with:
- Tools (mock weather, mock calendar)
- Reliability (timeouts, retries, exponential backoff)
- Memory/State (preferred city)
- Graceful fallback

Run:
    python single_agent_demo.py
"""

import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any
from openai import OpenAI

# OpenAI client

client = OpenAI(api_key=my_api_key)

# ---------- 1) TOOLS (mock external systems) ----------

def get_weather(city: str) -> Optional[Dict[str, str]]:
    """
    Mock weather API.
    30% chance of failure to demonstrate retries.
    Random latency to simulate real-world API delays.
    """
    # simulate latency
    time.sleep(random.uniform(0.1, 0.6))

    # simulate flaky failures
    if random.random() < 0.3:
        return None

    return {
        "city": city,
        "temp": f"{random.randint(15, 30)}Â°C",
        "condition": random.choice(["Sunny", "Cloudy", "Rain"])
    }

def check_calendar(date: str) -> Dict[str, Any]:
    """
    Mock calendar availability checker.
    """
    time.sleep(random.uniform(0.1, 0.4))
    return {
        "date": date,
        "available": random.choice([True, False])
    }


# ---------- 2) MEMORY / STATE ----------

@dataclass
class AgentState:
    memory: Dict[str, Any]

    def remember(self, key: str, value: Any):
        print(f"[State] Remembering {key} = {value}")
        self.memory[key] = value

    def recall(self, key: str):
        return self.memory.get(key)


# ---------- 3) RELIABILITY HELPERS ----------

def retry_with_backoff(func, *args, max_attempts=3, base_wait=1):
    """
    Wrap a tool call with retries + exponential backoff.
    """
    for attempt in range(1, max_attempts + 1):
        result = func(*args)

        if result is not None:
            return result

        wait = base_wait * (2 ** (attempt - 1))
        print(f"[Retry] Attempt {attempt} failed. Waiting {wait}s...")
        time.sleep(wait)

    return None


# ---------- 4) AGENT LOOP (Plan â†’ Act â†’ Check) ----------
# NOTE: For teaching, we keep planning simple and rule-assisted.

def plan_step(user_query: str) -> str:
    """
    Use OpenAI to create a plan for the user query.
    Available tools: get_weather, check_calendar
    """
    try:
        planning_prompt = f"""You are an AI assistant with these tools available:
- get_weather(city): Get current weather for a city
- check_calendar(date): Check if a date is available

User query: "{user_query}"

Create a brief plan (1-2 sentences) explaining what steps to take and which tools to use.
Example: "Plan: Use get_weather tool for the user's city, then summarize the weather conditions."
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        plan = response.choices[0].message.content.strip()
        return plan if plan.startswith("Plan:") else f"Plan: {plan}"
        
    except Exception as e:
        print(f"[Planning Error] {e}")
        # Fallback to simple rule-based planning
        q = user_query.lower()
        if "weather" in q:
            return "Plan: Use weather tool to fetch conditions, then summarize result."
        if "calendar" in q or "meeting" in q:
            return "Plan: Use calendar tool to check availability, then respond."
        return "Plan: No tool matched. Respond with fallback."


def act_step(plan: str, user_query: str, state: AgentState) -> Dict[str, Any]:
    """
    Route to the correct tool based on the plan.
    """
    plan_lower = plan.lower()

    if "get_weather" in plan_lower or "weather tool" in plan_lower:
        city = state.recall("preferred_city") or "Auckland"
        print(f"[Act] Calling get_weather(city='{city}') - based on plan")
        tool_result = retry_with_backoff(get_weather, city)
        return {"tool": "weather", "result": tool_result}

    if "check_calendar" in plan_lower or "calendar tool" in plan_lower:
        date = "tomorrow"
        print(f"[Act] Calling check_calendar(date='{date}') - based on plan")
        tool_result = check_calendar(date)
        return {"tool": "calendar", "result": tool_result}

    if "meeting" in plan_lower or "book" in plan_lower:
        # We demonstrate a fallback instead of full booking
        return {"tool": "meeting", "result": None, "error": "Booking tool not configured"}

    return {"tool": None, "result": None}


def check_step(action_payload: Dict[str, Any]) -> bool:
    """
    Validate tool results. If invalid, we fail gracefully.
    """
    if action_payload.get("error"):
        print(f"[Check] Tool error: {action_payload['error']}")
        return False

    result = action_payload.get("result")
    if result is None:
        print("[Check] Tool returned no data after retries.")
        return False

    print("[Check] Tool result looks valid.")
    return True


def respond(user_query: str, action_payload: Dict[str, Any], state: AgentState) -> str:
    """
    Produce final user-facing response.
    """
    tool = action_payload.get("tool")
    result = action_payload.get("result")

    if tool == "weather":
        # store memory
        state.remember("preferred_city", result["city"])
        return (
            f"The weather in {result['city']} is {result['temp']} "
            f"with {result['condition']}."
        )

    if tool == "calendar":
        if result["available"]:
            return f"Your calendar looks free {result['date']}."
        else:
            return f"You already have something booked {result['date']}."

    return "Sorry â€” I donâ€™t have a tool to handle that yet."


def run_agent_once(user_query: str, state: AgentState) -> str:
    print("\n" + "=" * 60)
    print(f"User: {user_query}")

    # PLAN
    plan = plan_step(user_query)
    print(plan)

    # ACT - now uses the plan!
    action_payload = act_step(plan, user_query, state)

    # CHECK
    ok = check_step(action_payload)

    if not ok:
        return (
            "Iâ€™m not confident enough to complete that action reliably. "
            "Try rephrasing or ask something else."
        )

    # RESPOND
    return respond(user_query, action_payload, state)


def main():
    state = AgentState(memory={})
    print("Single-Agent Demo. Type 'exit' to quit.\n")

    while True:
        user_query = input("User > ").strip()
        if user_query.lower() in ("exit", "quit"):
            break

        answer = run_agent_once(user_query, state)
        print("Agent >", answer)


if __name__ == "__main__":
    random.seed(42)
    main()