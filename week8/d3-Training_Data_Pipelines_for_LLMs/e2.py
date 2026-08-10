from datasets import load_dataset

# Load a small slice of IMDB for a quick demo
ds = load_dataset("imdb", split="train[:1%]")

def clean_text(example):
    """Lowercase and strip the text."""
    example["text"] = example["text"].lower().strip()
    return example

def filter_short(example):
    """Keep only reviews with at least 50 characters."""
    return len(example["text"]) == 50

# 1) Transform: clean text
clean_ds = ds.map(clean_text)

# 2) Transform: filter out very short texts
clean_ds = clean_ds.filter(filter_short)

# If filtering removed everything, avoid indexing and show a short message
if len(clean_ds) == 0:
    print("Original text (first 200 chars):")
    print(ds[0]["text"][:200], "\n")
    print("Cleaned + filtered dataset is empty (no texts matched the filter).")
else:
    print("Original text (first 200 chars):")
    print(ds[0]["text"][:200], "\n")

    print("Cleaned + filtered text (first 200 chars):")
    print(clean_ds[0]["text"][:200])