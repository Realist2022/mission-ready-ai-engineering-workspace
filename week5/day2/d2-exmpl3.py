import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = "Explain caching in LLMs in 3 sentences."

def measure_response_time(run_number):
    print(f"\nðŸ”„ Run {run_number}:")
    start = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content":prompt}],
        stream=True
    )

    first_token_time = None
    token_count = 0
    full_response = ""

    for chunk in resp:
        if chunk.choices[0].delta.content:
            token_count += 1
            full_response += chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = time.time()

    end = time.time()
    
    ttft = first_token_time - start if first_token_time else 0
    total_time = end - start
    tokens_per_sec = token_count / (end - first_token_time) if first_token_time else 0
    
    print(f"  TTFT: {ttft:.3f} seconds")
    print(f"  Total time: {total_time:.3f} seconds") 
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")
    print(f"  Token count: {token_count}")
    
    return ttft, total_time, tokens_per_sec

# Test the same prompt multiple times
print("ðŸ§ª Testing OpenAI API Caching Behavior")
print("=" * 50)

run1_ttft, run1_total, run1_tps = measure_response_time(1)
time.sleep(1)  # Brief pause
run2_ttft, run2_total, run2_tps = measure_response_time(2)
time.sleep(1)  # Brief pause  
run3_ttft, run3_total, run3_tps = measure_response_time(3)

print(f"\nðŸ“Š Summary:")
print(f"Run 1 - TTFT: {run1_ttft:.3f}s, Total: {run1_total:.3f}s")
print(f"Run 2 - TTFT: {run2_ttft:.3f}s, Total: {run2_total:.3f}s") 
print(f"Run 3 - TTFT: {run3_ttft:.3f}s, Total: {run3_total:.3f}s")

print(f"\nðŸ” Analysis:")
if run2_ttft < run1_ttft * 0.8:
    print("âœ… Possible caching detected - Run 2 was faster!")
else:
    print("âŒ No significant caching - OpenAI processes each request fresh")
    
print("\nðŸ’¡ Note: OpenAI API doesn't cache responses between requests.")
print("   Each call is processed independently for consistency and freshness.")