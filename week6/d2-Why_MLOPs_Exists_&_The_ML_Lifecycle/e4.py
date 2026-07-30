def evaluate_model():
    return {
        "accuracy": 0.87,
        "latency_ms": 420,
        "bias_gap": 0.03
    }

THRESHOLDS = {
    "accuracy": 0.85,
    "latency_ms": 500,
    "bias_gap": 0.05
}

def run_regression_tests(metrics):
    assert metrics["accuracy"] >= THRESHOLDS["accuracy"], "Accuracy regression"
    assert metrics["latency_ms"] <= THRESHOLDS["latency_ms"], "Latency regression"
    assert metrics["bias_gap"] <= THRESHOLDS["bias_gap"], "Bias regression"
    print("âœ… All regression tests passed")

# Demo: ML Regression Testing
if __name__ == "__main__":
    print("ðŸ§ª ML REGRESSION TESTING DEMO")
    print("=" * 40)
    
    print("ðŸ“‹ Quality thresholds:")
    for metric, threshold in THRESHOLDS.items():
        print(f"  {metric}: {threshold}")
    
    print("\nðŸ” Test 1: Good model (should pass)")
    metrics_good = evaluate_model()
    print(f"Current metrics: {metrics_good}")
    try:
        run_regression_tests(metrics_good)
    except AssertionError as e:
        print(f"âŒ Test failed: {e}")
    
    print("\nðŸ” Test 2: Accuracy regression (should fail)")
    metrics_bad_acc = {
        "accuracy": 0.82,  # Below 0.85 threshold
        "latency_ms": 400,
        "bias_gap": 0.02
    }
    print(f"Current metrics: {metrics_bad_acc}")
    try:
        run_regression_tests(metrics_bad_acc)
    except AssertionError as e:
        print(f"âŒ Test failed: {e}")
    
    print("\nðŸ” Test 3: Latency regression (should fail)")
    metrics_bad_latency = {
        "accuracy": 0.90,
        "latency_ms": 600,  # Above 500ms threshold
        "bias_gap": 0.01
    }
    print(f"Current metrics: {metrics_bad_latency}")
    try:
        run_regression_tests(metrics_bad_latency)
    except AssertionError as e:
        print(f"âŒ Test failed: {e}")
    
    print("\nðŸ’¡ Regression tests prevent quality degradation!")
    print("   - Catch performance drops before deployment")
    print("   - Ensure fairness standards are maintained")
    print("   - Block bad model updates automatically")