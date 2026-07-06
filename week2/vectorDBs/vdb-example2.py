# pip install chromadb sentence-transformers
from chromadb import Client
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma = Client()

# 1) Create a collection
collection = chroma.create_collection(name="knowledge_base")

# 2) Add docs + metadata
texts = [
    "RAG retrieves document chunks for LLMs.",
    "Product analytics tracks user engagement.",
    "FAISS enables fast vector search.",
]
embeddings = model.encode(texts).tolist()
metas = [{"topic":"AI"}, {"topic":"Analytics"}, {"topic":"Infra"}]
ids = ["doc1", "doc2", "doc3"]  # Required: unique IDs for each document
collection.add(documents=texts, embeddings=embeddings, metadatas=metas, ids=ids)

# 3) Query semantically
query = "How do I search with embeddings?"
results = collection.query(query_texts=[query], n_results=2)
print(results)