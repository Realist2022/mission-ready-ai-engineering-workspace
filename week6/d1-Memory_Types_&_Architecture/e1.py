short_term_memory = {}

def session_interaction(user_input):
    short_term_memory["last_input"] = user_input
    return f"I remember you just said: {short_term_memory['last_input']}"

# long_term_memory.py
import json

def save_preference(user_id, pref):
    with open("memory.json", "w") as f:
        json.dump({user_id: pref}, f)

def load_preference(user_id):
    try:
        with open("memory.json") as f:
            return json.load(f).get(user_id)
    except FileNotFoundError:
        return None

def chat_with_memory(user_id, message):
    # Short-term: conversation context
    response = session_interaction(message)
    
    # Long-term: user preferences
    pref = load_preference(user_id)
    if pref:
        response += f" | Your preference: {pref}"
    
    return response

# Demo
if __name__ == "__main__":    
    # Set user preference (long-term)
    save_preference("alice", "coffee")
    
    # Chat session (short-term)
    print("Session 1:")
    print(chat_with_memory("alice", "Hello"))
    print(chat_with_memory("alice", "How are you?"))
    
    # Reset short-term memory (new session)
    short_term_memory.clear()
    print("\nSession 2 (short-term cleared):")
    print(chat_with_memory("alice", "Good morning"))
    
    # Show short-term memory is truly reset
    print(f"Short-term memory now: {short_term_memory}")
    print(f"Long-term preference: {load_preference('alice')}")
    