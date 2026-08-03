def detect_watermark(text: str, marker="âŸ‚", threshold=0.05):
    tokens = text.split()
    ratio = tokens.count(marker) / max(len(tokens), 1)
    return ratio > threshold, ratio

# Example watermarked text (simple demo with a few marker tokens)
wm = "This is a demo text with a watermark marker âŸ‚ inside."

detected, score = detect_watermark(wm)
print("Detected:", detected, "Score:", score)