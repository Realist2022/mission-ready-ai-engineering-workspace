from chromadb import Client
from sentence_transformers import SentenceTransformer
import time


# Step 1: Initialize components
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma = Client()
collection = chroma.create_collection(name="knowledge_base")

documents = [
    "Customer support chatbots use RAG to provide accurate answers from company documentation.",
    "Data privacy compliance requires careful handling of user information in vector databases.",
    "Scaling vector search to millions of documents needs efficient indexing strategies like HNSW.",
    "Embedding models should be fine-tuned on domain-specific data for better retrieval quality.",
    "Hybrid search combines vector similarity with traditional keyword matching for best results.",
    "Vector database sharding distributes data across multiple nodes to handle large datasets."
]

# Rich metadata for filtering and analysis
metadata = [
    {"category": "AI Applications", "sensitivity": "low", "department": "Product"},
    {"category": "Data Governance", "sensitivity": "high", "department": "Legal"},  
    {"category": "Infrastructure", "sensitivity": "medium", "department": "Engineering"},
    {"category": "ML Engineering", "sensitivity": "low", "department": "Data Science"},
    {"category": "Search Technology", "sensitivity": "low", "department": "Engineering"},
    {"category": "Scalability", "sensitivity": "medium", "department": "Engineering"}
]

doc_ids = [f"kb_{i+1:03d}" for i in range(len(documents))]
print(f"âœ“ Prepared {len(documents)} documents with metadata")

# Step 3: Embedding & Storage with timing
print("\nðŸ§® Embedding and storage")
start_time = time.time()
embeddings = model.encode(documents).tolist()
collection.add(documents=documents, embeddings=embeddings, metadatas=metadata, ids=doc_ids)
embed_time = time.time() - start_time
print(f"âœ“ Embedded and stored {len(documents)} docs in {embed_time:.3f}s")

# Step 4: Multi-query search with interpretation
print("\nðŸ” Intelligent search and interpretation")

queries = [
    {"text": "How can we scale our search system?", "context": "Engineering planning"},
    {"text": "What privacy concerns should we consider?", "context": "Compliance review"},
    {"text": "How to improve search accuracy?", "context": "Product optimization"}
]

for i, query_info in enumerate(queries, 1):
    print(f"\n--- Query {i}: {query_info['text']} ---")
    print(f"Context: {query_info['context']}")
    
    # Perform search with timing
    search_start = time.time()
    results = collection.query(query_texts=[query_info['text']], n_results=2)
    search_time = time.time() - search_start
    
    print(f"Search time: {search_time:.4f}s")
    print("Top results:")
    
    for j, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        relevance_score = 1 - results["distances"][0][j]  # Convert distance to similarity
        print(f"  {j+1}. [{meta['category']}] Score: {relevance_score:.3f}")
        print(f"     {doc}")
        print(f"     Department: {meta['department']} | Sensitivity: {meta['sensitivity']}")

# Step 5: Quick demos of key concepts
print("\nðŸ”’ PRIVACY: Filtering by sensitivity level")
def filter_by_access(query, level="employee"):
    results = collection.query(query_texts=[query], n_results=6)
    allowed = ["low"] if level == "employee" else ["low", "medium", "high"]
    filtered = [(doc, meta) for doc, meta in zip(results["documents"][0], results["metadatas"][0]) 
                if meta["sensitivity"] in allowed]
    print(f"  {level} sees {len(filtered)}/6 results (sensitivity filter applied)")
    return filtered

employee_results = filter_by_access("privacy data", "employee")
admin_results = filter_by_access("privacy data", "admin")

print(f"\nðŸ”„ SHARDING: Split data across collections for scale")

# Create 2 shards
shard1 = chroma.create_collection(name="shard_1")
shard2 = chroma.create_collection(name="shard_2")

# Split documents between shards
mid = len(documents) // 2
shard1.add(documents=documents[:mid], embeddings=embeddings[:mid], 
           metadatas=metadata[:mid], ids=[f"s1_{i}" for i in range(mid)])
shard2.add(documents=documents[mid:], embeddings=embeddings[mid:], 
           metadatas=metadata[mid:], ids=[f"s2_{i}" for i in range(len(documents)-mid)])
print(f"  âœ“ Split {len(documents)} docs: {mid} in shard1, {len(documents)-mid} in shard2")

# Query across shards
def search_shards(query):
    # Search each shard separately
    results1 = shard1.query(query_texts=[query], n_results=2)
    results2 = shard2.query(query_texts=[query], n_results=2)
    
    # Combine results from both shards
    all_results = []
    for doc, dist in zip(results1["documents"][0], results1["distances"][0]):
        all_results.append({"doc": doc, "distance": dist, "shard": "shard1"})
    for doc, dist in zip(results2["documents"][0], results2["distances"][0]):
        all_results.append({"doc": doc, "distance": dist, "shard": "shard2"})
    
    # Sort by best match (lowest distance)
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:2]  # Return top 2

# Test sharded search
sharded_results = search_shards("scaling systems")
print(f"  Sharded search results:")
for i, result in enumerate(sharded_results, 1):
    score = 1 - result['distance']  # Convert distance to similarity score
    print(f"    {i}. [{result['shard']}] Score: {score:.3f} - {result['doc'][:50]}...")
