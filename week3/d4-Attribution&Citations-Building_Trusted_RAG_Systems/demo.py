api_key = ""    # Set your OpenAI API key here for full functionality
import os
import numpy as np
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer, util
from openai import OpenAI


# -----------------------------
# 1) Small knowledge base
# -----------------------------
KB: List[Dict] = [
    {
        "text": "Refunds are processed within ten business days after approval.",
        "source": "RefundPolicy.pdf"
    },
    {
        "text": "Refund requests must include the original order number.",
        "source": "RefundPolicy.pdf"
    },
    {
        "text": "Returns are accepted within 30 days of delivery.",
        "source": "ReturnsPolicy.pdf"
    },
    {
        "text": "Returned items must be in their original packaging.",
        "source": "ReturnsPolicy.pdf"
    },
    {
        "text": "Customer support is available 24/7 via email and live chat.",
        "source": "SupportGuide.pdf"
    },
    {
        "text": "Express shipping typically takes 1â€“2 business days.",
        "source": "ShippingInfo.pdf"
    },
]

# -----------------------------
# 2) Embedding model + helpers
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 1
CONF_THRESHOLD = 0.55  # tweak this to be stricter/looser

def build_embeddings(model: SentenceTransformer) -> np.ndarray:
    texts = [d["text"] for d in KB]
    embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embs

def retrieve(
    query: str,
    model: SentenceTransformer,
    kb_embs: np.ndarray,
    top_k: int = TOP_K
) -> Tuple[List[Dict], List[float], np.ndarray]:
    """Return top-k docs + similarity scores + query embedding."""
    q_emb = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)[0]
    sims = [float(util.cos_sim(q_emb, kb_embs[i])) for i in range(len(KB))]
    # sort by similarity descending
    idxs = np.argsort(sims)[::-1][:top_k]
    retrieved_docs = [KB[i] for i in idxs]
    retrieved_sims = [sims[i] for i in idxs]
    return retrieved_docs, retrieved_sims, q_emb

def compute_confidence(sims: List[float]) -> float:
    """Simple confidence = mean of top-k similarity scores."""
    if not sims:
        return 0.0
    return float(np.mean(sims))

# -----------------------------
# 3) Prompt building with citations
# -----------------------------
def build_prompt_with_citations(query: str, docs: List[Dict]) -> str:
    """
    Build a prompt that:
    - Shows each chunk with its source tag.
    - Asks the model to cite sources like [RefundPolicy.pdf].
    """
    context_lines = []
    for d in docs:
        context_lines.append(f"[{d['source']}] {d['text']}")
    context = "\n".join(context_lines)

    system_instruction = (
        "You are a helpful assistant answering questions about company policies.\n"
        "Use ONLY the information in the provided context.\n"
        "When you state a fact, include the source in square brackets, e.g. [RefundPolicy.pdf].\n"
        "If the context does not contain the answer, say you don't know.\n"
    )

    user_block = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer clearly, with citations:"
    )

    prompt = system_instruction + "\n" + user_block
    return prompt

# -----------------------------
# 4) LLM call (OpenAI)
# -----------------------------
def call_llm(prompt: str) -> str:
    if not api_key:
        # Friendly fallback if no key
        return (
            "âš ï¸ OPENAI_API_KEY not set.\n"
            "Here is the prompt that WOULD be sent to the model:\n\n"
            f"{prompt}"
        )

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# 5) Main pipeline
# -----------------------------
def run_rag_query(query: str):
    print("\n" + "=" * 60)
    print(f"User query: {query}")
    print("=" * 60)

    # 1) Load model + build embeddings (you might cache this in real apps)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    kb_embs = build_embeddings(model)

    # 2) Retrieve top-k
    docs, sims, q_emb = retrieve(query, model, kb_embs, top_k=TOP_K)

    print("\nTop retrieved chunks (with similarity scores):")
    for d, s in zip(docs, sims):
        print(f"- sim={s:.3f} | {d['source']} | {d['text']}")

    # 3) Confidence gating
    conf = compute_confidence(sims)
    print(f"\nComputed confidence (mean of top {TOP_K} sims): {conf:.3f}")

    if conf < CONF_THRESHOLD:
        print("\nâš ï¸ Confidence below threshold.")
        print("Assistant: I'm not confident I have enough information to answer that reliably.")
        print("          Please try rephrasing your question or check the policy documents directly.")
        return

    print("\nâœ… Confidence is high enough. Proceeding with citation-aware generation...")

    # 4) Build prompt with citations
    prompt = build_prompt_with_citations(query, docs)

    # 5) Call LLM
    answer = call_llm(prompt)

    print("\n--- Model Answer ---")
    print(answer)

    print("\n--- Sources Used ---")
    used_sources = {d["source"] for d in docs}
    for s in used_sources:
        print(f"- {s}")
        
def main():
    # You can hard-code a few demo queries, or read from input.
    demo_queries = [
        # "How long do refunds take and what do I need to include?",
        "What is your policy on returns?",
        # "Do you offer medical insurance to employees?",  # should be low confidence
    ]

    for q in demo_queries:
        run_rag_query(q)


if __name__ == "__main__":
    main()
  
