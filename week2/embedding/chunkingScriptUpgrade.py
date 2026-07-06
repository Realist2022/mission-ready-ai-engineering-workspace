from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

# Demo corpus
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


class DocumentChunker:
    """Handles the business logic of breaking documents down into smaller segments."""
    def __init__(self, max_words: int = 30, overlap: int = 8):
        self.max_words = max_words
        self.overlap = overlap

    def chunk_document(self, doc: Dict[str, str]) -> List[Dict[str, Any]]:
        text = doc["text"]
        doc_id = doc["doc_id"]
        title = doc["title"]
        
        words = text.split()
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + self.max_words, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"{doc_id}-{chunk_id}",
                "doc_id": doc_id,
                "title": title,
                "text": chunk_text,
                "start_word": start,  # Kept consistent with search method
                "end_word": end,      # Kept consistent with search method
            })

            if end == len(words):
                break
            start = end - self.overlap
            chunk_id += 1

        return chunks

    def chunk_corpus(self, docs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


class SemanticSearchEngine:
    """Manages the embedding model, stores text chunks, and handles vector search."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0,))

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Indexes text chunks by generating and storing their embeddings."""
        if not chunks:
            return
            
        self.chunks.extend(chunks)
        texts = [c["text"] for c in chunks]
        
        # Generate new embeddings
        new_embeddings = self.model.encode(
            texts, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        
        # Append to existing embeddings if they already exist
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries the vector index and returns sorted, enriched results."""
        if self.embeddings.size == 0:
            raise ValueError("The search engine index is empty. Add documents first.")

        # Encode query to the same vector space
        q_emb = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        
        # Compute similarities
        hits = util.semantic_search(q_emb, self.embeddings, top_k=top_k)[0]

        results = []
        for h in hits:
            ch = self.chunks[h["corpus_id"]]
            results.append({
                "score": float(h["score"]),
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "title": ch["title"],
                "text": ch["text"],
                "span": (ch["start_word"], ch["end_word"]),
            })
        return results


# -------------------------------------
# Execution workflow using the new classes
# -------------------------------------
def main():
    # 1. Initialize our components
    chunker = DocumentChunker(max_words=30, overlap=8)
    search_engine = SemanticSearchEngine()

    # 2. Process and index the corpus
    print("\n=== Processing Documents ===")
    chunks = chunker.chunk_corpus(DOCS)
    search_engine.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks successfully.")

    # 3. Perform searches with a clean API
    queries = [
        "How do retrieval systems use chunks with large language models?",
        "Which metrics help evaluate product features and user behavior?"
    ]

    for q in queries:
        print(f"\n=== Query: {q}")
        results = search_engine.search(q, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.3f} | {r['title']} | {r['chunk_id']}")
            print(f"    {r['text']}\n")


if __name__ == "__main__":
    main()