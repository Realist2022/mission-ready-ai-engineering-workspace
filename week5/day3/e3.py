#!/usr/bin/env python3
"""
âš¡ Real Speculative Decoding with OpenAI
Draft model: gpt-3.5-turbo (fast)
Main model: gpt-4o-mini (more accurate)
"""
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RealDraftModel:
    def __init__(self):
        self.model = "gpt-3.5-turbo"  # Faster, cheaper
    
    def generate(self, prompt, k=3):
        """Generate k draft continuations quickly"""
        print(f"  ðŸš€ Draft model generating {k} options...")
        start = time.time()
        
        drafts = []
        for i in range(k):
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"Complete this sentence with one meaningful word: '{prompt}'"}],
                max_tokens=3,  # Allow space for complete words
                temperature=0.8  # More creative for variety
            )
            word = response.choices[0].message.content.strip()
            drafts.append(word)
        
        draft_time = time.time() - start
        print(f"  â±ï¸  Draft time: {draft_time:.2f}s")
        return drafts

class RealMainModel:
    def __init__(self):
        self.model = "gpt-4o-mini"  # More accurate
    
    def verify(self, prompt, draft_word):
        """Verify if draft word is good quality"""
        print(f"  ðŸ” Verifying: '{draft_word}'...")
        start = time.time()
        
        verification_prompt = f"""
        Given the context: "{prompt}"
        Rate if "{draft_word}" is a good continuation (1-10 scale).
        Answer only with a number.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": verification_prompt}],
            max_tokens=1,
            temperature=0.1  # Low temperature for consistent scoring
        )
        
        try:
            score = int(response.choices[0].message.content.strip())
            verify_time = time.time() - start
            print(f"  ðŸ“Š Score: {score}/10 ({verify_time:.2f}s)")
            return score >= 7  # Accept if score >= 7
        except:
            return False
    
    def correct_word(self, prompt):
        """Generate correct continuation when draft fails"""
        print(f"  ðŸŽ¯ Generating correct continuation...")
        start = time.time()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"{prompt} Continue with one good word:"}],
            max_tokens=1,
            temperature=0.3  # Lower temperature for quality
        )
        
        word = response.choices[0].message.content.strip()
        correct_time = time.time() - start
        print(f"  âœ¨ Generated: '{word}' ({correct_time:.2f}s)")
        return word

def main():
    print("âš¡ REAL SPECULATIVE DECODING WITH OPENAI")
    print("=" * 50)
    
    draft_model = RealDraftModel()
    main_model = RealMainModel()
    
    prompt = "The weather today is very"
    print(f"ðŸŽ¯ Prompt: '{prompt}'\n")
    
    # Generate drafts
    print("ðŸ“ Step 1: Generate draft continuations")
    draft_words = draft_model.generate(prompt, k=3)
    print(f"   Draft options: {draft_words}\n")
    
    # Verify each draft
    print("ðŸ” Step 2: Verify each draft")
    for i, word in enumerate(draft_words):
        if main_model.verify(prompt, word):
            print(f"  âœ… ACCEPTED: '{word}' - Good quality!")
            final_result = f"{prompt} {word}"
            break
        else:
            print(f"  âŒ REJECTED: '{word}' - Poor quality")
    else:
        # All drafts rejected, generate correct one
        print(f"\nðŸŽ¯ Step 3: All drafts rejected, generating correct word")
        correct_word = main_model.correct_word(prompt)
        final_result = f"{prompt} {correct_word}"
    
    print(f"\nðŸ“Š FINAL RESULT:")
    print(f"   '{final_result}'")
    print(f"\nðŸ’¡ This demonstrates real speculative decoding:")
    print(f"   - Fast draft model explores options")
    print(f"   - Accurate main model ensures quality")

if __name__ == "__main__":
    main()