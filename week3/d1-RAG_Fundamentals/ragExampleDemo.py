my_api_key = ""

from dotenv import load_dotenv
load_dotenv()

"""
PIPELINE FLOW:
Documents â†’ Chunks â†’ Embeddings â†’ Vector DB â†’ Query â†’ Retrieve â†’ Generate Answer
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import uuid
import os


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# ---------------------------
# ðŸ“š STEP 1: Sample Knowledge Base
# ---------------------------
# In a real system, these would come from files, databases, or APIs
# Each document has:
# - id: unique identifier for tracking
# - title: human-readable name
# - text: the actual content to search through
RAW_DOCS = [
    {
        "id": "hr-policy",
        "title": "HR Leave Policy 2024",
        "text": """
Our company provides 20 days of paid annual leave per year for full-time employees.
Employees can carry over up to 5 unused leave days to the next year.
Sick leave is separate and covered under the Health & Wellness policy.
For special circumstances, additional unpaid leave may be approved by HR.
"""
    },
    {
        "id": "rag-notes",
        "title": "RAG System Overview",
        "text": """
Retrieval-Augmented Generation (RAG) combines a retriever and a generator.
The retriever finds relevant document chunks using embeddings and vector search.
The generator, usually a large language model, uses those chunks to answer questions.
RAG reduces hallucinations by grounding answers in real documents.
"""
    },
    {
        "id": "product-analytics",
        "title": "Product Analytics Basics",
        "text": """
Product analytics focuses on how users interact with a product.
Key metrics include activation, engagement, retention, and conversion.
Teams use these insights to improve onboarding, features, and user experience.
Accurate tracking and event naming are critical for trustworthy analytics.
"""
    }
]


# ---------------------------
# âœ‚ï¸ STEP 2: Document Chunking
# ---------------------------
# WHY CHUNK? Large documents don't fit in LLM context windows.
# Chunking splits documents into smaller, searchable pieces while preserving meaning.
# 
# CHUNKING STRATEGY:
# - chunk_size=300: Each piece ~300 characters (balance between context and specificity)  
# - chunk_overlap=60: Overlap prevents splitting related sentences
# - separators: Try paragraph breaks first, then sentences, then words
def build_chunks(raw_docs):
    """
    ðŸ”ª CHUNKING PROCESS:
    1. Take each document
    2. Split into overlapping chunks using smart separators
    3. Attach metadata (source tracking) to each chunk
    4. Return list of chunks ready for embedding
    
    RESULT: Original 3 docs become ~4 searchable chunks with source info
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    for doc in raw_docs:
        for chunk_text in splitter.split_text(doc["text"]):
            if chunk_text.strip():
                chunks.append(
                    {
                        "page_content": chunk_text.strip(),
                        "metadata": {
                            "source_id": doc["id"],
                            "title": doc["title"]
                        }
                    }
                )
    return chunks


# ---------------------------
# ðŸ§® STEP 3: Vector Embeddings & Storage  
# ---------------------------
# WHY EMBEDDINGS? Computers can't understand text directly. 
# Embeddings convert text meaning into numbers (vectors) that enable similarity search.
#
# EMBEDDING MODEL: all-MiniLM-L6-v2 
# - Converts text â†’ 384-dimensional vector
# - Captures semantic meaning (not just keywords)
# - "vacation days" and "annual leave" have similar vectors
#
# VECTOR DATABASE: FAISS (Facebook AI Similarity Search)
# - Stores vectors for fast similarity search
# - Can handle millions of vectors efficiently
def build_vectorstore(chunks):
    """
    ðŸ”¢ EMBEDDING PROCESS:
    1. Initialize embedding model (HuggingFace all-MiniLM-L6-v2)
    2. Convert each chunk text â†’ 384-dimensional vector
    3. Store vectors + metadata in FAISS database
    4. Return searchable vector database
    
    RESULT: Text chunks become searchable by semantic similarity
    """
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    texts = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embed_model,
        metadatas=metadatas
    )
    return vectordb


# ---------------------------
# ðŸ”— STEP 4: RAG System Assembly
# ---------------------------
# RETRIEVER: Finds most relevant chunks using vector similarity
# - search_type="similarity": Use cosine similarity between vectors
# - k=3: Return top 3 most relevant chunks per query
#
# LLM (Language Model): Generates answers from retrieved context
# - gpt-4o-mini: Cost-effective OpenAI model 
# - temperature=0: Deterministic answers (no creativity)
#
# PROMPT TEMPLATE: Instructions for how LLM should use retrieved context
def build_qa_system(vectordb, use_openai=True):
    """
    ðŸ”§ SYSTEM ASSEMBLY:
    1. Create retriever from vector database (finds relevant chunks)
    2. Initialize LLM for answer generation (if API key available)
    3. Create prompt template (instructions for LLM)
    4. Return complete RAG system ready for queries
    
    RESULT: System that can retrieve + generate for any question
    """
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    system = {"retriever": retriever}
    
    if use_openai:
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )

                   # 🔥 FUNNY PROMPT TEMPLATE
            prompt_template = PromptTemplate.from_template(
                """You are a helpful assistant who must answer the question 
using ONLY the provided context. However, you must also be funny — 
add light humour, playful commentary, or dad‑jokes, while still giving 
a correct and grounded answer.

Context:
{context}

Question: {question}

Funny Answer:"""
            )

            system.update({"llm": llm, "prompt": prompt_template})
            print("✔ OpenAI LLM successfully configured with FUNNY mode!")
        except Exception as e:
            print(f"OpenAI setup failed: {e}")
            print("Continuing with retrieval-only mode...\n")
    
    return system     
            
#             prompt_template = PromptTemplate.from_template(
#                 """Based on the following context, answer the question.
                
# Context:
# {context}

# Question: {question}

# Answer:"""
#             )
#             system.update({"llm": llm, "prompt": prompt_template})
#             print("âœ… OpenAI LLM successfully configured!")
#         except Exception as e:
#             print(f"OpenAI setup failed: {e}")
#             print("Continuing with retrieval-only mode...\n")
    
#     return system


# ---------------------------
# ðŸŽ¯ STEP 5: RAG Query Execution
# ---------------------------
# THE RAG PROCESS:
# 1. RETRIEVE: Find chunks semantically similar to user question
# 2. CONTEXTUALIZE: Combine retrieved chunks into context  
# 3. GENERATE: LLM creates answer based on retrieved context
# 4. TRACE: Show which sources were used (transparency)
#
# TRACEABILITY: Each query gets unique trace ID for debugging/auditing
def run_query(qa_system, question: str):
    """
    ðŸ”„ QUERY EXECUTION FLOW:
    1. Generate unique trace ID for this query
    2. Use retriever to find most relevant chunks  
    3. Display retrieved context (transparency)
    4. If LLM available: generate answer from context
    5. If no LLM: show what would happen (demo mode)
    
    RESULT: Grounded answer with full source traceability
    """
    trace_id = str(uuid.uuid4())
    print(f"\n=== TRACE ID: {trace_id} ===")
    print(f"User Question: {question}\n")

    # Step 1: Retrieve relevant documents
    retriever = qa_system["retriever"]
    source_docs = retriever.invoke(question)

    print("Retrieved Context Chunks:")
    for i, doc in enumerate(source_docs, start=1):
        meta = doc.metadata
        preview = doc.page_content[:160].replace("\n", " ")
        print(
            f"[{i}] {meta.get('title')} "
            f"(source_id={meta.get('source_id')})"
        )
        print(f"    {preview}...\n")

    # Step 2: Generate answer if LLM is available
    if "llm" in qa_system and "prompt" in qa_system:
        llm = qa_system["llm"]
        prompt = qa_system["prompt"]
        
        # Prepare context for the LLM
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        # Generate answer using LLM
        formatted_prompt = prompt.format(context=context, question=question)
        answer = llm.invoke(formatted_prompt).content

        print("Final Answer:")
        print(answer)
    else:
        print("ðŸ¤– LLM Answer Generation:")
        print("(LLM not available - showing retrieval results only)")
        print("With the retrieved context above, an LLM would synthesize")
        print("a comprehensive answer based on the relevant chunks.")
        
    print("\n" + "=" * 40 + "\n")


def main():
    """
    ðŸš€ MAIN EXECUTION: Complete RAG Pipeline Demo
    
    PIPELINE STAGES:
    1. Document Processing: Raw text â†’ Chunks with metadata
    2. Vector Storage: Chunks â†’ Embeddings â†’ Searchable database  
    3. System Assembly: Retriever + LLM â†’ Complete RAG system
    4. Query Testing: Run sample questions through full pipeline
    
    DEMO QUERIES test different document types:
    - HR Policy: Factual extraction ("How many vacation days?")
    - Technical Docs: Concept explanation ("What is RAG?") 
    - Analytics Guide: List retrieval ("What are key metrics?")
    """
    
    # STAGE 1: Document chunking
    # Transform raw documents into searchable pieces
    chunks = build_chunks(RAW_DOCS)
    print(f"âœ… Document chunking complete: {len(chunks)} searchable chunks created")

    # STAGE 2: Vector database creation  
    # Convert text chunks to embeddings and store for similarity search
    vectordb = build_vectorstore(chunks)
    print("âœ… Vector database built: FAISS index ready for semantic search\n")

    # STAGE 3: RAG system assembly
    # Combine retriever + LLM into complete question-answering system
    print("ðŸ”§ Building RAG system with OpenAI LLM...")
    
    qa_system = build_qa_system(vectordb, use_openai=True)
    print("âœ… RAG system ready! (Full generation mode)\n")

    # STAGE 4: Live demonstration
    # Test the complete pipeline with realistic business questions
    print("ðŸ” Testing FULL RAG pipeline with LLM generation:")
    print("Each query shows: Retrieval â†’ Context â†’ Generated Answer\n")
    
    # DEMO QUERIES: Test different types of information retrieval
    run_query(qa_system, "How many days of annual leave do we get?")  # Factual extraction
    run_query(qa_system, "What is RAG and why is it useful?")        # Concept explanation  
    run_query(qa_system, "What are common product analytics metrics?") # List retrieval
    


if __name__ == "__main__":
    main()