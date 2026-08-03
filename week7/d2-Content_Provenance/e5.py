def detect_watermark(text: str, marker="âŸ‚", threshold=0.05):
    tokens = text.split()
    ratio = tokens.count(marker) / max(len(tokens), 1)
    return ratio > threshold, ratio

def remove_marker(text, marker="âŸ‚"):
    return text.replace(marker, "")

wm = "This is a demo text with a watermark marker âŸ‚ inside."

cleaned = remove_marker(wm)
detected, score = detect_watermark(cleaned)

print("After editing:")
print("Detected:", detected, "Score:", score)