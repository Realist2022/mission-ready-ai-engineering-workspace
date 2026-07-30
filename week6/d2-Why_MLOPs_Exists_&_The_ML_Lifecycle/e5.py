#!/usr/bin/env python3
"""
Real ML Regression Test Example
Simple sentiment classifier with actual regression testing
"""
import random

class SentimentClassifier:
    """Simple sentiment classifier - returns confidence scores"""
    
    def __init__(self, version="v1.0"):
        self.version = version
        # Simulate different model performance based on version
        if version == "v1.0":
            self.base_accuracy = 0.89
            self.avg_latency = 45  # ms
        elif version == "v1.1":
            self.base_accuracy = 0.91  # Improved!
            self.avg_latency = 48  # Slightly slower
        else:  # "v2.0" - broken model
            self.base_accuracy = 0.82  # Regressed!
            self.avg_latency = 35  # Faster but less accurate
    
    def predict(self, text):
        """Simulate prediction with some randomness"""
        # Add realistic variance
        accuracy = self.base_accuracy + random.uniform(-0.05, 0.05)
        latency = self.avg_latency + random.uniform(-5, 10)
        
        # Simple "prediction" based on keywords
        positive_words = ["good", "great", "love", "excellent"]
        score = 0.5  # neutral baseline
        
        for word in positive_words:
            if word in text.lower():
                score += 0.3
        
        return {
            "sentiment": "positive" if score > 0.6 else "negative",
            "confidence": min(accuracy, 1.0),
            "latency_ms": max(latency, 10)  # minimum 10ms
        }

def run_regression_test(model, test_cases):
    """Run regression tests on the model"""
    print(f"ðŸ§ª Testing {model.version}")
    
    results = []
    total_latency = 0
    
    for text in test_cases:
        result = model.predict(text)
        results.append(result)
        total_latency += result["latency_ms"]
    
    # Calculate metrics
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    
    print(f"   Avg confidence: {avg_confidence:.3f}")
    print(f"   Avg latency: {avg_latency:.1f}ms")
    
    # Regression thresholds
    MIN_CONFIDENCE = 0.85
    MAX_LATENCY = 50
    
    # Run tests
    if avg_confidence < MIN_CONFIDENCE:
        print(f"   âŒ FAIL: Confidence {avg_confidence:.3f} < {MIN_CONFIDENCE}")
        return False
    
    if avg_latency > MAX_LATENCY:
        print(f"   âŒ FAIL: Latency {avg_latency:.1f}ms > {MAX_LATENCY}ms")
        return False
    
    print(f"   âœ… PASS: All thresholds met")
    return True

if __name__ == "__main__":
    print("ðŸš€ REAL REGRESSION TEST EXAMPLE")
    print("=" * 40)
    
    # Test cases
    test_texts = [
        "This movie is great!",
        "I love this product",
        "Terrible experience",
        "Good service"
    ]
    
    # Test different model versions
    models = [
        SentimentClassifier("v1.0"),  # Should pass
        SentimentClassifier("v1.1"),  # Should pass (better)
        SentimentClassifier("v2.0")   # Should fail (regressed)
    ]
    
    print("Test cases:", test_texts)
    print("\nThresholds: confidence â‰¥ 0.85, latency â‰¤ 50ms\n")
    
    for model in models:
        passed = run_regression_test(model, test_texts)
        if not passed:
            print(f"   ðŸš« Block deployment of {model.version}")
        print()