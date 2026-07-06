from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Any, Tuple

# -----------------------------
# 1) Demo corpus (2 short docs)
# -----------------------------
# This simulates having multiple documents in your knowledge base
# Each document has metadata (ID, title) and the actual text content
# In real applications, this could be loaded from files, databases, or APIs
DOCS = [
    {
        "doc_id": "doc-ai",
        "title": "Intro to Large Language Models",
        "text": (
            "Large language models use transformer architectures to process text. "
            "They represent meaning as dense vectors called embeddings. "
            "Attention mechanisms help models focus on relevant parts of input. "
            "RAG systems retrieve relevant chunks and feed them back to the model. "
            "Good chunking improves retrieval quality and final answer accuracy."
        ),
    },
    {
        "doc_id": "doc-analytics",
        "title": "Product Analytics Guide",
        "text": (
            "Product analytics tracks user behavior to improve features. "
            "Metrics include retention, engagement, and conversion rate. "
            "Event data is collected from apps and websites and stored in warehouses. "
            "Dashboards help teams test hypotheses and prioritize experiments. "
            "Clear instrumentation is essential for trustworthy insights."
        ),
    },
]


# -------------------------------------------
# 2) Simple word-based chunker with overlap
#    (keeps the demo dependency-free)
# -------------------------------------------
# CHUNKING: Breaking large documents into smaller, manageable pieces
# Why chunk? Large docs exceed model context limits and hurt retrieval precision
# Overlap ensures important information isn't lost at chunk boundaries
def make_chunks(
    text: str,
    doc_id: str,
    title: str,
    max_words: int = 30,
    overlap: int = 8,
) -> List[Dict[str, Any]]:

    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    # SLIDING WINDOW: Move through text creating overlapping segments
    # This ensures no information is lost at arbitrary boundaries
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(
            {
                "chunk_id": f"{doc_id}-{chunk_id}",
                "doc_id": doc_id,
                "title": title,
                "text": chunk_text,
                "start_word": start,
                "end_word": end,
            }
        )

        if end == len(words):
            break
        # slide window forward with overlap
        start = end - overlap
        chunk_id += 1

    return chunks

# CORPUS PREPARATION: Process all documents into a unified chunk collection
# This creates the searchable knowledge base by chunking each document
# and combining them into one flat list for embedding and retrieval
def build_corpus_chunks(docs: List[Dict[str, str]], max_words=30, overlap=8):
    all_chunks: List[Dict[str, Any]] = []
    for d in docs:
        all_chunks.extend(
            make_chunks(d["text"], d["doc_id"], d["title"], max_words=max_words, overlap=overlap)
        )
    return all_chunks


# --------------------------------------
# 3) Embed all chunks using a small model
# --------------------------------------
# EMBEDDINGS: Convert text chunks into dense vector representations
# These vectors capture semantic meaning in mathematical form (384 dimensions)
# Similar meanings â†’ similar vectors â†’ high cosine similarity
def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",  # Fast, lightweight sentence transformer
) -> Tuple[np.ndarray, SentenceTransformer]:

    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]  # Extract just the text content
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True) # normalize_embeddings=True keeps vectors L2-normalized which is great for cosine similarity.

    return emb, model


# -----------------------------------------
# 4) Simple semantic search (cosine similarity)
# -----------------------------------------
# SIMILARITY SEARCH: Find chunks most semantically related to user query
# This is the core of RAG retrieval - matching query meaning to relevant content
def search(
    query: str,
    chunks: List[Dict[str, Any]],
    chunk_emb,  # torch.Tensor - pre-computed chunk embeddings
    model: SentenceTransformer,
    top_k: int = 3,
) -> List[Dict[str, Any]]:

    # Convert query to same vector space as chunks
    q_emb = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    
    # COSINE SIMILARITY: Measures angle between vectors (0=opposite, 1=identical)
    # util.semantic_search efficiently computes similarities and returns top matches
    hits = util.semantic_search(q_emb, chunk_emb, top_k=top_k)[0]

    # RESULT ENRICHMENT: Combine similarity scores with chunk metadata
    results = []
    for h in hits:
        ch = chunks[h["corpus_id"]]  # Get original chunk data
        results.append(
            {
                "score": float(h["score"]),  # Similarity score (0-1)
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "title": ch["title"],
                "text": ch["text"],
                "span": (ch["start_word"], ch["end_word"]),  # Position in original doc
            }
        )
    return results


# -------------------------------------
# 5) Run the demo end-to-end (main)
# -------------------------------------
# COMPLETE RAG RETRIEVAL PIPELINE DEMONSTRATION
def main():
    # STEP 1: CHUNKING - Break documents into retrievable segments
    print("\n=== Building chunks ===")
    chunks = build_corpus_chunks(DOCS, max_words=30, overlap=8)
    for c in chunks:
        print(f"- {c['chunk_id']} ({c['title']}): {c['text'][:70]}...")

    print(f"\nTotal chunks: {len(chunks)}")

    # STEP 2: EMBEDDING - Convert text to mathematical vectors
    print("\n=== Embedding chunks (this may take a few seconds the first time) ===")
    chunk_emb, model = embed_chunks(chunks)
    print(f"Embeddings shape: {tuple(chunk_emb.shape)}  # (num_chunks, dim)")

    # STEP 3: QUERY TESTING - Demonstrate semantic retrieval
    # These queries test different domains to show embedding effectiveness
    queries = [
        "How do retrieval systems use chunks with large language models?",  # AI/RAG query
        "Which metrics help evaluate product features and user behavior?",   # Analytics query
    ]

    # SEMANTIC SEARCH DEMONSTRATION: For each query, find most relevant chunks
    for q in queries:
        print(f"\n=== Query: {q}")
        results = search(q, chunks, chunk_emb, model, top_k=3)
        # Display results with similarity scores and metadata for analysis
        for i, r in enumerate(results, 1):
            print(
                f"[{i}] score={r['score']:.3f} | {r['title']} | {r['chunk_id']} | words {r['span'][0]}-{r['span'][1]}"
            )
            print(f"    {r['text']}\n")

    # OVERLAP IMPORTANCE DEMONSTRATION: Test chunk boundary handling
    # This query contains text that might span chunk boundaries
    boundary_query = "Attention mechanisms help models focus on relevant parts of input."
    print(f"\n=== Boundary Query (overlap test): {boundary_query}")
    results = search(boundary_query, chunks, chunk_emb, model, top_k=2)
    for i, r in enumerate(results, 1):
        print(
            f"[{i}] score={r['score']:.3f} | {r['title']} | {r['chunk_id']} | words {r['span'][0]}-{r['span'][1]}"
        )
        print(f"    {r['text']}\n")


if __name__ == "__main__":
    main()