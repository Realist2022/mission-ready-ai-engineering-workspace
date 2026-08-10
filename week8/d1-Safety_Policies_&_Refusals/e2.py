BANNED_KEYWORDS = ["hack", "kill"]

def safe_model(prompt):
    for word in BANNED_KEYWORDS:
        if word in prompt.lower():
            return "I canâ€™t help with that request."
    return f"I can help explain this safely: {prompt}"

while True:
    user = input("User: ")
    print("Model:", safe_model(user))