from sentence_transformers import util, SentenceTransformer
import numpy as np

# Initialize model and create sample data
model = SentenceTransformer('all-MiniLM-L6-v2')
query = "How do I get a refund?"
docs = ["Refunds take 10 days", "Returns within 30 days", "Contact support for help"]

# Create embeddings
q_emb = model.encode([query])[0]
doc_embs = model.encode(docs)

def compute_confidence(query_emb, doc_embs, top_k=3):
    sims = [float(util.cos_sim(query_emb, d)) for d in doc_embs]
    sims.sort(reverse=True)
    top_scores = sims[:top_k]
    return np.mean(top_scores)

conf = compute_confidence(q_emb, doc_embs)
if conf < 0.1:
    print("âš ï¸ I'm not confident enough to answer.")
else:
    print(f"Confidence {conf:.2f} â€” Proceed with citations.")