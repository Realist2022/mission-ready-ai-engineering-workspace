from sentence_transformers import SentenceTransformer, util
import networkx as nx
import matplotlib.pyplot as plt

texts = [
  "Refunds are processed within 10 days.",
  "Returns must be requested within 30 days.",
  "Customer complaints are handled by support.",
  "Payment errors cause transaction delays."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_tensor=True)

# Connect nodes if cosine similarity > 0.5
G = nx.Graph()
for i, t in enumerate(texts):
    G.add_node(i, text=t)
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        if util.cos_sim(embeddings[i], embeddings[j]) > 0.5:
            G.add_edge(i, j)

# Create visualization
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout

# Draw the graph
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1500, font_size=12, font_weight='bold')

# Add text labels showing document content
labels = {}
for i, text in enumerate(texts):
    labels[i] = f"{i}: {text[:20]}..."
    
nx.draw_networkx_labels(G, pos, labels, font_size=8)
plt.title("Document Similarity Graph\n(Connected if similarity > 0.5)")
plt.show()