"""
Simple RAG 3.0 Demo with Reranking
----------------------------------
Shows: Graph retrieval â†’ Reranking â†’ Answer generation
"""

import numpy as np                                     # For numerical operations
import networkx as nx                                  # For creating graphs
import matplotlib.pyplot as plt                       # For visualization
from sentence_transformers import SentenceTransformer, CrossEncoder  # For embeddings and reranking

class SimpleRAG3:
    def __init__(self):
        """Initialize models and knowledge graph"""
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Model to create embeddings
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # Model to rerank results
        self.graph = nx.Graph()                        # Graph to store document connections
        self.docs = {}                                 # Dictionary: doc_id -> text
        
    def add_knowledge(self, doc_id, text, connections=None):
        """Add a document to the knowledge graph with optional connections"""
        self.docs[doc_id] = text                       # Store document text
        self.graph.add_node(doc_id, text=text)         # Add document as graph node
        
        # Connect to other documents
        if connections:                                # If connections list provided
            for other_id in connections:               # Loop through each connection
                if other_id in self.docs:              # Check if target document exists
                    self.graph.add_edge(doc_id, other_id)  # Create edge between documents
    
    def graph_retrieval(self, query, top_k=5):
        """Find relevant docs using graph traversal (semantic + connections)"""
        # Find best starting document
        query_emb = self.embedder.encode([query])[0]   # Convert query to embedding vector
        scores = []                                    # List to store similarity scores
        
        for doc_id, text in self.docs.items():        # Loop through all documents
            doc_emb = self.embedder.encode([text])[0]  # Convert document to embedding
            similarity = np.dot(query_emb, doc_emb)    # Calculate cosine similarity
            scores.append((similarity, doc_id))       # Store score and document ID
        
        scores.sort(reverse=True)                      # Sort by highest similarity first
        start_doc = scores[0][1]                       # Get best matching document ID
        
        # Expand through graph connections
        retrieved = [start_doc]                        # Start with best matching document
        for doc_id in list(retrieved):                 # For each document in current results
            for neighbor in self.graph.neighbors(doc_id):  # Find connected documents
                if neighbor not in retrieved and len(retrieved) < top_k:  # If not already found and under limit
                    retrieved.append(neighbor)         # Add connected document to results
        
        print(f"Graph retrieval: {retrieved}")         # Show what was found
        return retrieved                               # Return list of document IDs
    
    def rerank_results(self, query, doc_ids):
        """Rerank documents using cross-encoder for better relevance"""
        # Create query-document pairs for reranker
        pairs = []                                     # List to store query-document pairs
        for doc_id in doc_ids:                         # For each document ID
            pairs.append([query, self.docs[doc_id]])   # Create [query, document_text] pair
        
        # Get relevance scores
        scores = self.reranker.predict(pairs)          # Get relevance scores from cross-encoder
        
        # Sort by relevance score
        ranked = list(zip(scores, doc_ids))            # Combine scores with document IDs
        ranked.sort(reverse=True)                      # Sort by highest score first
        
        reranked_docs = [doc_id for score, doc_id in ranked]  # Extract just the document IDs
        print(f"After reranking: {reranked_docs}")     # Show reranked results
        return reranked_docs                           # Return reranked document IDs
    
    def generate_context(self, doc_ids):
        """Combine retrieved documents into context"""
        context = "\n".join([f"- {self.docs[doc_id]}" for doc_id in doc_ids])  # Join documents with bullet points
        return context                                 # Return formatted context string
    
    def answer_query(self, query):
        """Main RAG 3.0 pipeline: retrieve â†’ rerank â†’ generate"""
        print(f"Query: {query}")                       # Display the user's question
        
        # Step 1: Graph-based retrieval
        candidates = self.graph_retrieval(query, top_k=4)  # Find 4 candidate documents using graph
        
        # Step 2: Rerank for relevance
        final_docs = self.rerank_results(query, candidates)[:3]  # Rerank and take top 3
        
        # Step 3: Generate context
        context = self.generate_context(final_docs)    # Combine final documents into context
        
        answer = f"Based on retrieved knowledge:\n{context}\n\nThis information addresses your query about: {query}"  # Create final answer
        return answer, final_docs                      # Return answer and document list
    
    def visualize(self, query, retrieved_docs):
        """Show the knowledge graph with highlighted retrieved documents"""
        plt.figure(figsize=(10, 6))                    # Create figure with specific size
        pos = nx.spring_layout(self.graph, seed=42)    # Calculate node positions for layout
        
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightgray',   # Draw all nodes in gray
                              node_size=800, alpha=0.7)
        
        nx.draw_networkx_nodes(self.graph, pos, nodelist=retrieved_docs,  # Draw retrieved nodes in red
                              node_color='red', node_size=1000)
        
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)  # Draw connections between nodes
        
        # Add labels
        labels = {node: node for node in self.graph.nodes()}  # Create labels for each node
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)  # Display node labels
        
        plt.title(f"RAG 3.0: {query}")                # Set title with query
        plt.axis('off')                               # Hide axes
        plt.tight_layout()                            # Adjust layout
        plt.savefig('rag3_simple.png', dpi=150)       # Save image file
        print("Graph saved as 'rag3_simple.png'")     # Confirm save

def main():
    # Create RAG system
    rag = SimpleRAG3()                                # Initialize the RAG system
    
    # Add knowledge with connections
    rag.add_knowledge("refund_policy", "Refunds processed within 10 days",   # Add refund policy document
                     ["refund_req", "support"])       # Connect to requirements and support
    rag.add_knowledge("refund_req", "Need order number for refunds",         # Add refund requirements
                     ["refund_policy"])               # Connect back to refund policy
    rag.add_knowledge("return_policy", "30-day return window",               # Add return policy
                     ["packaging_req"])               # Connect to packaging requirements
    rag.add_knowledge("packaging_req", "Original packaging required",        # Add packaging requirements
                     ["return_policy"])               # Connect back to return policy
    rag.add_knowledge("support", "Contact support for help",                # Add support info
                     ["refund_policy"])               # Connect to refund policy
    rag.add_knowledge("shipping", "Standard shipping 3-5 days")             # Add shipping info (no connections)
    
    print("=== RAG 3.0 with Reranking ===\n")       # Display header
    
    # Run query
    query = "How do I get a refund?"                 # Define user question
    answer, retrieved = rag.answer_query(query)      # Run the full RAG pipeline
    
    print(f"\n--- Final Answer ---")                 # Display final answer header
    print(answer)                                    # Show the generated answer
    
    # Visualize
    rag.visualize(query, retrieved)                  # Create and save visualization

if __name__ == "__main__":
    main()