#!/usr/bin/env python3
"""
ðŸš€ Advanced Inference Performance Engineering Demo
Topics: KV Cache, Paged Attention, Speculative Decoding

This demo shows real implementations and performance comparisons!
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import random

print("ðŸ”¥ INFERENCE PERFORMANCE ENGINEERING DEMO")
print("=" * 60)

# ============================================================================
# ðŸ§  PART 1: KV CACHE DEMONSTRATION
# ============================================================================

class SimpleAttention:
    """Naive attention without KV caching"""
    
    def __init__(self, d_model=64):
        self.d_model = d_model
        self.scale = d_model ** -0.5
        # Simulate learned weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
    def forward(self, x, past_tokens=None):
        """Forward pass - recalculates everything every time"""
        if past_tokens is not None:
            x = np.concatenate([past_tokens, x], axis=0)
        
        # Compute Q, K, V for ENTIRE sequence every time (inefficient!)
        Q = x @ self.W_q
        K = x @ self.W_k  
        V = x @ self.W_v
        
        # Attention computation
        scores = Q @ K.T * self.scale
        attn_weights = self.softmax(scores)
        output = attn_weights @ V
        
        return output, x  # Return full sequence for next iteration

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class KVCachedAttention:
    """Efficient attention with KV caching"""
    
    def __init__(self, d_model=64, max_seq_len=512):
        self.d_model = d_model
        self.scale = d_model ** -0.5
        self.max_seq_len = max_seq_len
        
        # Same weights as naive version
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
        # KV CACHE - the secret sauce! ðŸŽ¯
        self.k_cache = np.zeros((max_seq_len, d_model))
        self.v_cache = np.zeros((max_seq_len, d_model))
        self.cache_length = 0
        
    def forward(self, x_new):
        """Forward pass - only processes NEW tokens!"""
        batch_size = x_new.shape[0]
        
        # Only compute Q, K, V for NEW tokens
        Q_new = x_new @ self.W_q
        K_new = x_new @ self.W_k
        V_new = x_new @ self.W_v
        
        # Update cache with new K, V
        start_idx = self.cache_length
        end_idx = self.cache_length + batch_size
        self.k_cache[start_idx:end_idx] = K_new
        self.v_cache[start_idx:end_idx] = V_new
        self.cache_length = end_idx
        
        # Get cached K, V for entire sequence
        K_all = self.k_cache[:self.cache_length]
        V_all = self.v_cache[:self.cache_length]
        
        # Attention with new Q and cached K, V
        scores = Q_new @ K_all.T * self.scale
        attn_weights = self.softmax(scores)
        output = attn_weights @ V_all
        
        return output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def clear_cache(self):
        """Clear the KV cache for new sequence"""
        self.k_cache.fill(0)
        self.v_cache.fill(0)
        self.cache_length = 0


def demo_kv_cache():
    print("\nðŸ§  PART 1: KV CACHE DEMONSTRATION")
    print("-" * 40)
    
    print("ðŸŽ¯ Scenario: Generating a 50-token sequence")
    print("   - Naive: Recalculates K,V for entire sequence each step")
    print("   - Cached: Reuses previous K,V computations\n")
    
    d_model = 64
    sequence_length = 50
    
    # Initialize both models
    naive_attn = SimpleAttention(d_model)
    cached_attn = KVCachedAttention(d_model)
    
    # Simulate token generation
    times_naive = []
    times_cached = []
    
    print("Generating tokens...")
    past_tokens = None
    
    for i in range(sequence_length):
        # New token embedding
        new_token = np.random.randn(1, d_model)
        
        # Naive approach
        start = time.time()
        _, past_tokens = naive_attn.forward(new_token, past_tokens)
        times_naive.append(time.time() - start)
        
        # Cached approach  
        start = time.time()
        _ = cached_attn.forward(new_token)
        times_cached.append(time.time() - start)
        
        if i % 10 == 0:
            print(f"  Token {i:2d}: Naive={times_naive[-1]*1000:.1f}ms, Cached={times_cached[-1]*1000:.1f}ms")
    
    speedup = np.mean(times_naive) / np.mean(times_cached)
    print(f"\nðŸ“Š Results:")
    print(f"   Average time - Naive: {np.mean(times_naive)*1000:.2f}ms")
    print(f"   Average time - Cached: {np.mean(times_cached)*1000:.2f}ms")
    print(f"   ðŸš€ Speedup: {speedup:.1f}x faster with KV caching!")
    
    return times_naive, times_cached

# ============================================================================
# ðŸ“„ PART 2: PAGED ATTENTION SIMULATION
# ============================================================================

class PagedAttentionManager:
    """Simulates vLLM's paged attention memory management"""
    
    def __init__(self, page_size=16, total_pages=64):
        self.page_size = page_size  # Tokens per page
        self.total_pages = total_pages
        self.pages = {}  # page_id -> np.array of tokens
        self.free_pages = set(range(total_pages))
        self.allocated_pages = {}  # session_id -> [page_ids]
        
    def allocate_sequence(self, session_id: str, estimated_length: int):
        """Allocate pages for a new sequence"""
        pages_needed = (estimated_length + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < pages_needed:
            # Simulate page eviction (LRU-style)
            self._evict_pages(pages_needed - len(self.free_pages))
        
        allocated = []
        for _ in range(pages_needed):
            if self.free_pages:
                page_id = self.free_pages.pop()
                allocated.append(page_id)
                self.pages[page_id] = np.zeros(self.page_size)
        
        self.allocated_pages[session_id] = allocated
        return allocated
    
    def _evict_pages(self, count):
        """Evict oldest pages (simplified LRU)"""
        sessions_to_evict = list(self.allocated_pages.keys())[:count]
        for session_id in sessions_to_evict:
            self.free_sequence(session_id)
    
    def free_sequence(self, session_id: str):
        """Free all pages for a sequence"""
        if session_id in self.allocated_pages:
            for page_id in self.allocated_pages[session_id]:
                self.free_pages.add(page_id)
                del self.pages[page_id]
            del self.allocated_pages[session_id]
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        used_pages = sum(len(pages) for pages in self.allocated_pages.values())
        fragmentation = len(self.allocated_pages) * self.page_size - sum(
            len(self.allocated_pages[sid]) * self.page_size 
            for sid in self.allocated_pages
        )
        
        return {
            'total_pages': self.total_pages,
            'used_pages': used_pages,
            'free_pages': len(self.free_pages),
            'active_sessions': len(self.allocated_pages),
            'fragmentation': fragmentation
        }


def demo_paged_attention():
    print("\nðŸ“„ PART 2: PAGED ATTENTION SIMULATION")
    print("-" * 40)
    
    print("ðŸŽ¯ Scenario: Multiple chat sessions with varying lengths")
    print("   - Shows how paging reduces memory fragmentation")
    print("   - Demonstrates automatic page eviction\n")
    
    manager = PagedAttentionManager(page_size=16, total_pages=32)
    
    # Simulate various conversation lengths
    sessions = [
        ("chat_short_1", 25),    # 2 pages
        ("chat_medium_1", 80),   # 5 pages  
        ("chat_long_1", 200),    # 13 pages
        ("chat_short_2", 30),    # 2 pages
        ("chat_medium_2", 90),   # 6 pages
        ("chat_long_2", 250),    # 16 pages - will trigger eviction!
    ]
    
    print("Allocating sequences:")
    for session_id, length in sessions:
        allocated = manager.allocate_sequence(session_id, length)
        stats = manager.get_memory_stats()
        
        print(f"  {session_id:12} ({length:3d} tokens, {len(allocated):2d} pages)")
        print(f"    Memory: {stats['used_pages']:2d}/{stats['total_pages']} pages used, "
              f"{stats['active_sessions']} active sessions")
        
        if stats['free_pages'] < 5:
            print(f"    âš ï¸  Low memory! Only {stats['free_pages']} pages free")
        print()
    
    final_stats = manager.get_memory_stats()
    efficiency = (final_stats['used_pages'] / final_stats['total_pages']) * 100
    
    print(f"ðŸ“Š Final Memory Statistics:")
    print(f"   Memory efficiency: {efficiency:.1f}%")
    print(f"   Active sessions: {final_stats['active_sessions']}")
    print(f"   ðŸŽ¯ Key benefit: No memory fragmentation like traditional allocation!")

# ============================================================================
# âš¡ PART 3: SPECULATIVE DECODING SIMULATION  
# ============================================================================

class DraftModel:
    """Fast, smaller model for generating draft tokens"""
    
    def __init__(self, accuracy=0.7):
        self.accuracy = accuracy  # How often draft is correct
        self.inference_time = 0.01  # Very fast (10ms per token)
    
    def generate_draft(self, prompt: str, num_tokens: int = 4) -> List[str]:
        """Generate draft tokens quickly"""
        time.sleep(self.inference_time * num_tokens)
        
        # Simulate draft generation
        draft_tokens = [f"draft_{i}" for i in range(num_tokens)]
        return draft_tokens
    
    def is_correct_prediction(self) -> bool:
        """Simulate whether draft token matches target model"""
        return random.random() < self.accuracy


class TargetModel:
    """Larger, more accurate model for verification"""
    
    def __init__(self):
        self.inference_time = 0.1  # Slower (100ms per token)
    
    def verify_and_generate(self, prompt: str, draft_tokens: List[str]) -> Tuple[List[str], int]:
        """Verify draft tokens and generate correct ones"""
        # Simulate verification time (parallel verification is faster)
        time.sleep(self.inference_time * 0.5)  # Parallel verification speedup
        
        # Check how many draft tokens are correct
        draft_model = DraftModel()  # For checking correctness
        accepted_tokens = []
        
        for token in draft_tokens:
            if draft_model.is_correct_prediction():
                accepted_tokens.append(f"verified_{token}")
            else:
                # First wrong token - generate correct one and stop
                accepted_tokens.append(f"corrected_{len(accepted_tokens)}")
                break
        
        return accepted_tokens, len(accepted_tokens)


def demo_speculative_decoding():
    print("\nâš¡ PART 3: SPECULATIVE DECODING SIMULATION")
    print("-" * 40)
    
    print("ðŸŽ¯ Scenario: Generate 20 tokens with and without speculative decoding")
    print("   - Draft model: Fast but 70% accurate")
    print("   - Target model: Slow but highly accurate")
    print("   - Shows wall-clock time improvement\n")
    
    draft_model = DraftModel(accuracy=0.7)
    target_model = TargetModel()
    
    target_tokens = 20
    draft_lookahead = 4
    
    # Method 1: Traditional sequential generation
    print("ðŸŒ Traditional Sequential Generation:")
    start_time = time.time()
    
    sequential_tokens = []
    for i in range(target_tokens):
        # Each token requires full target model inference
        time.sleep(target_model.inference_time)
        sequential_tokens.append(f"seq_token_{i}")
        
        if i % 5 == 0:
            elapsed = time.time() - start_time
            print(f"   Generated {i+1:2d} tokens in {elapsed:.2f}s")
    
    sequential_time = time.time() - start_time
    print(f"   Total time: {sequential_time:.2f}s")
    
    # Method 2: Speculative decoding
    print(f"\nâš¡ Speculative Decoding (lookahead={draft_lookahead}):")
    start_time = time.time()
    
    speculative_tokens = []
    generated_count = 0
    
    while generated_count < target_tokens:
        remaining = min(draft_lookahead, target_tokens - generated_count)
        
        # Draft model generates multiple tokens quickly
        draft_tokens = draft_model.generate_draft("prompt", remaining)
        
        # Target model verifies in parallel and corrects
        verified_tokens, accepted_count = target_model.verify_and_generate("prompt", draft_tokens)
        
        speculative_tokens.extend(verified_tokens)
        generated_count += accepted_count
        
        if generated_count % 5 == 0 or generated_count >= target_tokens:
            elapsed = time.time() - start_time
            acceptance_rate = accepted_count / len(draft_tokens) * 100
            print(f"   Generated {min(generated_count, target_tokens):2d} tokens in {elapsed:.2f}s "
                  f"(accepted {accepted_count}/{len(draft_tokens)} = {acceptance_rate:.0f}%)")
    
    speculative_time = time.time() - start_time
    speedup = sequential_time / speculative_time
    
    print(f"\nðŸ“Š Results:")
    print(f"   Sequential time: {sequential_time:.2f}s")
    print(f"   Speculative time: {speculative_time:.2f}s") 
    print(f"   ðŸš€ Speedup: {speedup:.1f}x faster!")
    print(f"   ðŸ’¡ Key insight: Draft model accuracy directly affects speedup")

# ============================================================================
# ðŸ“Š MAIN DEMO EXECUTION
# ============================================================================

def main():
    print("ðŸŽ¬ Starting comprehensive inference optimization demo...\n")
    
    # Run all demonstrations
    times_naive, times_cached = demo_kv_cache()
    demo_paged_attention() 
    demo_speculative_decoding()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("ðŸŽ¯ DEMO SUMMARY - KEY TAKEAWAYS")
    print("=" * 60)
    
    print("\n1ï¸âƒ£ KV CACHE:")
    print("   âœ… Avoids recomputing attention for previous tokens")
    print("   âœ… Linear memory growth, but massive time savings")
    print("   âœ… Essential for any practical text generation")
    
    print("\n2ï¸âƒ£ PAGED ATTENTION:")
    print("   âœ… Eliminates memory fragmentation")
    print("   âœ… Enables efficient batching of variable-length sequences") 
    print("   âœ… vLLM's key innovation for serving efficiency")
    
    print("\n3ï¸âƒ£ SPECULATIVE DECODING:")
    print("   âœ… Uses fast draft model to predict future tokens")
    print("   âœ… Target model verifies in parallel")
    print("   âœ… Wall-clock speedup without quality loss")
    
    print(f"\nðŸš€ These optimizations are why modern LLM serving is possible!")
    print("   Real systems combine ALL these techniques for maximum performance.")
    
    return times_naive, times_cached

if __name__ == "__main__":
    main()