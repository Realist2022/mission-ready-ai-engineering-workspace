import os
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

class HybridRAGPipeline:
    def __init__(
        self, 
        api_key, 
        docs=None, 
        embedding_model="all-MiniLM-L6-v2", 
        llm_model="gpt-4o-mini",
        chunk_size=150, 
        chunk_overlap=20
    ):
        # 1. Set Configuration
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 2. Initialize Models
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)
        
        # 3. Initialize State
        self.bm25_retriever = None
        self.dense_retriever = None
        
        # 4. Ingest Initial Docs
        if docs:
            self.ingest_documents(docs)

    def ingest_documents(self, docs):
        """Splits documents and builds both sparse and dense retrievers."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.create_documents(docs)

        # Build Sparse Retriever (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)

        # Build Dense Retriever (FAISS)
        faiss_db = FAISS.from_documents(chunks, embedding=self.embeddings)
        self.dense_retriever = faiss_db.as_retriever(search_kwargs={"k": 2})
        print(f"Successfully ingested {len(docs)} documents into {len(chunks)} chunks.")

    def hybrid_retrieve(self, query):
        """Fetches and merges documents from both retrievers."""
        if not self.bm25_retriever or not self.dense_retriever:
            raise ValueError("Retrievers are not initialized. Please ingest documents first.")
            
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        # Naive merge (keep top unique docs)
        merged = {d.page_content: d for d in dense_docs + sparse_docs}
        return list(merged.values())

    def ask(self, query):
        """Retrieves context and generates an answer using the LLM."""
        retrieved_docs = self.hybrid_retrieve(query)
        hybrid_context = "\n".join([d.page_content for d in retrieved_docs])
        
        prompt = f"Context:\n{hybrid_context}\n\nQuestion: {query}"
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "context": retrieved_docs
        }

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    my_api_key = "sk-your-key-here"
    
    my_docs = [
        "Error E1234 occurs when checkout service cannot reach payment API.",
        "RAG combines retrieval with generation to improve factual grounding.",
        "Employees receive 20 days of annual leave each year.",
        "Payment gateway connection timeouts may trigger code E1234.",
        "Hybrid retrieval mixes BM25 keyword search and vector search."
    ]

    # Instantiate the pipeline
    rag = HybridRAGPipeline(api_key=my_api_key, docs=my_docs)

    # Ask a question
    query = "What causes error E1234?"
    result = rag.ask(query)
    
    print(f"\nQuery: {query}")
    print(f"Generated Answer: {result['answer']}")


    