import re
import os
from openai import OpenAI
from dotenv import load_dotenv  # type: ignore

# Load variables from .env if present (OPENAI_API_KEY, etc.)
load_dotenv()

# Get the API key from environment and fail fast if missing
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

# Create OpenAI client â€“ key comes from env (OPENAI_API_KEY)
client = OpenAI(api_key=api_key)

def redact_emails(text):
    # Very simple email redaction
    return re.sub(r"\S+@\S+", "<EMAIL>", text)

def redact_ids(text):
    """
    Simple ID redaction:
    - 8+ digit sequences (e.g. customer IDs, account numbers)
    """
    return re.sub(r"\b\d{8,}\b", "<ID>", text)

def preprocess_locally(text):
    # All sensitive scrubbing happens *before* calling the LLM
    text = redact_emails(text)
    text = redact_ids(text)
    return text

def send_to_llm(prompt: str):
    """
    Real LLM call using OpenAI, assuming OPENAI_API_KEY is set in env/.env.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for a customer-support style application. "
                    "The text you receive has already had personal identifiers (emails, IDs, etc.) "
                    "replaced with placeholders like <EMAIL> and <ID>. "
                    "Your job is to infer what the user likely wants help with from the text "
                    "and provide a specific, practical, and informative answer. "
                    "Do NOT ask the user to provide more personal information or more details; "
                    "instead, give a best-effort, concrete response with examples or clear steps."
                ),
            },
            {
                "role": "user",
                "content": (
                    "This is redacted user input. Treat it as a short support question or request. "
                    "Ignore the fact that it is redacted and do NOT talk about redaction or privacy. "
                    "Explain what the user might be trying to do and give them useful guidance, "
                    "tips, or next steps based only on this text:\n\n"
                    f"{prompt}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content
    print("LLM INPUT:", prompt)
    print("LLM OUTPUT:", answer)

# Example user text that might contain sensitive data
user_input = "My email is alice@example.com and my customer ID is 123456789."

clean_input = preprocess_locally(user_input)
send_to_llm(clean_input)