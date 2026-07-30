from sentence_transformers import CrossEncoder

# Define the missing variables
query = "How do I process a refund?"
retrieved_docs = [
    "Returns must be made within 30 days of purchase",
    "Refunds are processed within 5-7 business days", 
    "Customer service is available Monday through Friday",
    "Shipping costs are non-refundable",
    "Refund requests require original receipt"
]

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [(query, doc) for doc in retrieved_docs]
scores = reranker.predict(pairs)

ranked = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
print("Top reranked documents:")
for i, doc in enumerate(ranked[:3], 1):
    print(f"{i}. {doc}")    
    