import json


def is_safe_content(text):
    """Check if response contains no harmful content"""
    banned_words = ["hate", "violence", "illegal", "harmful"]
    text_lower = text.lower()
    return not any(word in text_lower for word in banned_words)

def has_proper_length(text, min_len=10, max_len=500):
    """Check if response has appropriate length"""
    return min_len <= len(text.strip()) <= max_len


safe_text = "The weather is nice today."
unsafe_text = "This contains hate speech."

short_text = "Hi"
good_text = "This is a reasonable length response with useful information."
long_text = "A" * 600  # Too long


print("\nContent Safety:")
print(is_safe_content(safe_text))     # True
print(is_safe_content(unsafe_text))   # False

print("\nLength Validation:")
print(has_proper_length(short_text))  # False (too short)
print(has_proper_length(good_text))   # True
print(has_proper_length(long_text))   # False (too long)
