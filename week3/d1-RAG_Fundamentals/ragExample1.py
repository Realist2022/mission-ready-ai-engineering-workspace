# Simple RAG Example - Works with current LangChain versions
# RAG = Retrieval-Augmented Generation
# This demo shows how RAG retrieves relevant documents first, then uses them to generate better answers

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# 1ï¸âƒ£ KNOWLEDGE BASE CREATION
# We start with a small collection of documents that will serve as our knowledge base
# In production, this would be thousands or millions of documents
texts = [
    "RAG combines retrieval with generation to provide factual answers.",
    "Embeddings represent text meaning as high-dimensional vectors.",
    "Vector databases enable fast semantic search across documents."
]

print("\nðŸ—ï¸  STEP 1: Building Knowledge Base")
try:
    # Transform text into numerical vectors that capture semantic meaning
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store embeddings in FAISS (Facebook AI Similarity Search) for fast retrieval
    db = FAISS.from_texts(texts, emb)
    
    # Create retriever that will find the most relevant documents for any query
    retriever = db.as_retriever(search_kwargs={"k": 2})  # Return top 2 matches

    # 2ï¸âƒ£ RETRIEVAL PHASE
    # When user asks a question, we first find the most relevant documents
    print("\nðŸ” STEP 2: Retrieval Phase") 
    query = "How does RAG improve answer accuracy?"
    print(f"Query: '{query}'")
    
    # This performs semantic search - not just keyword matching!
    docs = retriever.invoke(query)
    print(f"âœ“ Found {len(docs)} most relevant documents using cosine similarity")
    
    # 3ï¸âƒ£ CONTEXT PRESENTATION
    # Show what information was retrieved to help answer the question
    print("\nðŸ“š STEP 3: Retrieved Context")
    for i, doc in enumerate(docs, 1):
        print(f"  Document {i}: {doc.page_content}")

except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure you have: pip install langchain-community langchain-huggingface faiss-cpu")