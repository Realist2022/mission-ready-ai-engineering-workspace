import time

memory_store = {}

def store_memory(key, value, ttl_seconds):
    memory_store[key] = {
        "value": value,
        "expires_at": time.time() + ttl_seconds
    }

def get_memory(key):
    item = memory_store.get(key)
    if not item:
        return None
    if time.time() > item["expires_at"]:
        del memory_store[key]
        return None
    return item["value"]

# Demo: Memory with TTL (Time To Live)
if __name__ == "__main__":
    print("â° TTL MEMORY DEMO")
    
    # Store memories with different expiry times
    store_memory("user_name", "Alice", ttl_seconds=2)
    store_memory("session_id", "abc123", ttl_seconds=5)
    
    print(f"Stored: {get_memory('user_name')} (2s TTL)")
    print(f"Stored: {get_memory('session_id')} (5s TTL)")
    
    # Wait and check expiry
    print("\nWaiting 6 seconds...")
    time.sleep(6)
    
    print(f"After 3s - user_name: {get_memory('user_name')}")  # Expired
    print(f"After 3s - session_id: {get_memory('session_id')}")  # Still valid
    
    print(f"\nðŸ’¡ Memory auto-expires - no manual cleanup needed!")