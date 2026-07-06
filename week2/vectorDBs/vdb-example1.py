# pip install -U sentence-transformers numpy
from sentence_transformers import SentenceTransformer
import numpy as np


# 1) Embed a tiny "corpus"
"""
This is like creating a vector database collection.
"""
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = [
    "RAG retrieves document chunks before answering.",
    "User retention improves with great onboarding.",
    "The weather is sunny today in Auckland.",
    "Embeddings turn text into vectors for search."
]
C = model.encode(corpus, normalize_embeddings=True)  # shape: (4, 384)

# 2) Embed a query
q = "How do I use vectors to find related text?"
Q = model.encode([q], normalize_embeddings=True)     # shape: (1, 384)

# 3) Cosine similarity = dot product (because vectors are normalized)
"""
This is the HEART of every vector database:
"""
scores = (Q @ C.T).ravel()   # Q @ C.T: Matrix multiplication computes similarity to ALL documents at once
topk_idx = np.argsort(-scores)[:2] # Finds top-k results (what vector DBs call "nearest neighbors")

# 4) Display results  
print("Query:", q, "\n")
for i in topk_idx:
    print(f"score={scores[i]:.3f} | {corpus[i]}")