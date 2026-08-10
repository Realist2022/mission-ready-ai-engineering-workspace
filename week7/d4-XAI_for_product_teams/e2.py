import os
from openai import OpenAI
from dotenv import load_dotenv  # type: ignore

# Load variables from .env in this folder (including OPENAI_API_KEY)
load_dotenv()

# Read key from environment and fail with a clear message if missing
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set or is empty. "
        "Check your .env file and that you're running from the directory that contains it."
    )

client = OpenAI(api_key=api_key)

def explain_decision(features):
    explanation = []

    if features["income"] > 50000:
        explanation.append("Higher income increased approval likelihood")

    if features["debt"] > 25000:
        explanation.append("Higher debt reduced approval likelihood")

    if not explanation:
        explanation.append("Decision based on overall profile")

    return explanation

def explain_decision_natural(features):
    """Return a single, user-friendly sentence explaining this one decision."""
    reasons = explain_decision(features)

    if len(reasons) == 1:
        return f"For this application, the main factor was: {reasons[0].lower()}."
    else:
        # join all reasons into one readable sentence
        main = reasons[0].lower()
        others = [r.lower() for r in reasons[1:]]
        all_reasons = ", ".join(others)
        return (
            "For this application, the decision was influenced by: "
            f"{main}" + (", " + all_reasons if all_reasons else "") + "."
        )

def explain_decision_with_openai(features):
    """
    Use OpenAI to turn the structured reasons into a very clear, friendly explanation.
    """
    reasons = explain_decision(features)
    reasons_text = "; ".join(reasons)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain model decisions to non-technical users. "
                    "Be short, clear, and neutral. Do not invent extra reasons."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Turn these decision reasons into a single, user-friendly explanation. "
                    "Keep it one or two sentences:\n"
                    f"{reasons_text}"
                ),
            },
        ],
    )
    return response.choices[0].message.content

person = {"income": 60000, "debt": 30000}
print(explain_decision(person))              # structured explanation (list)
print(explain_decision_natural(person))      # rule-based natural language
print(explain_decision_with_openai(person))  # OpenAI-refined explanation