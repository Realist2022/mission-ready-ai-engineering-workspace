import random

def slm_answer(prompt: str) -> tuple[str, float]:
    # Fake confidence for demo: in real life, use logprobs, classifier, or self-score
    confidence = random.uniform(0.3, 0.95)
    answer = f"[SLM] Answer to: {prompt[:40]}..."
    return answer, confidence

def llm_answer(prompt: str) -> str:
    return f"[LLM] Higher-quality answer to: {prompt[:40]}..."

def answer_with_fallback(prompt: str, threshold: float = 0.75) -> str:
    ans, conf = slm_answer(prompt)
    if conf >= threshold:
        return f"{ans}\n(confidence={conf:.2f} âœ… kept SLM)"
    else:
        better = llm_answer(prompt)
        return f"{better}\n(confidence={conf:.2f} âŒ escalated to LLM)"

# Demo: SLM to LLM Fallback System
if __name__ == "__main__":
    print("ðŸ¤– SLM TO LLM FALLBACK SYSTEM DEMO")
    print("=" * 50)
    print("ðŸ’¡ Strategy: Use cheap SLM first, expensive LLM only when needed\n")
    
    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "How do I make chocolate chip cookies?",
        "What are the implications of AI alignment?",
        "Write a Python function to sort a list"
    ]
    
    total_slm_used = 0
    total_llm_used = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"ðŸ” Query {i}: {prompt}")
        result = answer_with_fallback(prompt, threshold=0.75)
        print(f"ðŸ“ {result}")
        
        # Count usage for cost analysis
        if "kept SLM" in result:
            total_slm_used += 1
        else:
            total_llm_used += 1
        
        print("-" * 50)
    
    # Cost analysis
    print(f"\nðŸ’° COST ANALYSIS:")
    print(f"   SLM calls: {total_slm_used} Ã— $0.001 = ${total_slm_used * 0.001:.3f}")
    print(f"   LLM calls: {total_llm_used} Ã— $0.020 = ${total_llm_used * 0.020:.3f}")
    print(f"   Total cost: ${(total_slm_used * 0.001) + (total_llm_used * 0.020):.3f}")
    print(f"   vs All-LLM: ${len(prompts) * 0.020:.3f}")
    savings = ((len(prompts) * 0.020) - ((total_slm_used * 0.001) + (total_llm_used * 0.020))) / (len(prompts) * 0.020) * 100
    print(f"   ðŸ’¡ Savings: {savings:.1f}% vs using LLM for everything!")