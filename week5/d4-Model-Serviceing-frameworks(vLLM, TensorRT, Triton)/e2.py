import random
import time

def slm_call(prompt):
    if random.random() < 0.3:
        raise TimeoutError("SLM timeout")
    return "[SLM] fast answer"

def llm_call(prompt):
    return "[LLM] reliable answer"

def answer_with_resilience(prompt, retries=2):
    for attempt in range(retries):
        try:
            return slm_call(prompt)
        except TimeoutError:
            time.sleep(0.1)

    return llm_call(prompt)

# Simple Demo
if __name__ == "__main__":
    print("ðŸ”„ RESILIENT INFERENCE DEMO")
    print("=" * 40)
    
    test_prompts = [
        "What is Python?",
        "Explain AI",
        "How does ML work?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nðŸ” Test {i}: {prompt}")
        
        # Show multiple attempts for same prompt
        for trial in range(3):
            result = answer_with_resilience(prompt, retries=2)
            if "[SLM]" in result:
                print(f"   Trial {trial+1}: âœ… SLM succeeded")
            else:
                print(f"   Trial {trial+1}: âŒ Fell back to LLM")
    
    print(f"\nðŸ’¡ Key insight: Retries help unreliable SLMs succeed more often!")