from datasets import load_dataset

# Load only a small sample of the IMDB dataset for a quick demo
# (split="train[:5%]" means: first 5% of the train split)
ds = load_dataset("imdb", split="train[:5%]")

# Basic info about the dataset
print(ds)                 # shows number of rows, columns, features
print(ds[0])              # first example (a dict with "text" and "label")

# Optionally inspect a few more rows
for i in range(3):
    print(f"\nExample {i}:")
    print("Text:", ds[i]["text"][:200], "...")
    print("Label:", ds[i]["label"])