memory_store = {}

def ask_consent(item):
    answer = input(f"May I remember this? '{item}' (yes/no): ")
    return answer.lower() == "yes"

def store_memory(key, value):
    if ask_consent(value):
        memory_store[key] = value
        print("Memory saved.")
    else:
        print("Memory not saved.")

def show_memory():
    print("Current memory:", memory_store)

# Demo: Consent-Aware Memory
if __name__ == "__main__":
    print("ðŸ” CONSENT-AWARE MEMORY DEMO")
    print("Only stores data with user permission\n")
    
    # Simulate storing different types of information
    test_data = [
        ("name", "Alice"),
        ("location", "New York"), 
        ("email", "alice@email.com")
    ]
        
    # Try to store each piece of data
    for key, value in test_data:
        store_memory(key, value)
    
    print("\nðŸ“Š Final result:")
    show_memory()
    print("\nðŸ’¡ Only consented data was stored!")