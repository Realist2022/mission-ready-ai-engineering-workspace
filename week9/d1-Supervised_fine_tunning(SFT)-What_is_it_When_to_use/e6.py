def quality_filter(example):
    text = example["input"].strip()

    # Reject very short samples
    if len(text) < 10:
        return False

    # Reject very long samples (toy upper bound)
    if len(text) > 1000:
        return False

    # Reject if it looks like a single token / junk (no spaces)
    if " " not in text:
        return False

    # Reject obvious toxic patterns (toy example)
    banned_words = ["hate", "kill", "bomb"]
    for word in banned_words:
        if word in text.lower():
            return False

    return True

filtered_data = [ex for ex in train_examples if quality_filter(ex)]
print(f"Kept {len(filtered_data)} of {len(train_examples)} examples")