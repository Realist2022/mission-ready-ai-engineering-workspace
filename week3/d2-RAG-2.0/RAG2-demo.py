my_api_key = ""

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key (add your key here)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------------
# 1ï¸âƒ£  Our mini corpus
# -----------------------------------------------------------
docs = [
    "Error E1234 occurs when checkout service cannot reach payment API.",
    "RAG combines retrieval with generation to improve factual grounding.",
    "Employees receive 20 days of annual leave each year.",
    "Payment gateway connection timeouts may trigger code E1234.",
    "Hybrid retrieval mixes BM25 keyword search and vector search."
]

# -----------------------------------------------------------
# 2ï¸âƒ£  Split text into chunks
# -----------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.create_documents(docs)

# -----------------------------------------------------------
# 3ï¸âƒ£  Sparse retriever (BM25)
# -----------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(chunks)

# -----------------------------------------------------------
# 4ï¸âƒ£  Dense retriever (embeddings + FAISS)
# -----------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(chunks, embedding=embeddings)
dense_retriever = faiss_db.as_retriever(search_kwargs={"k": 2})

# -----------------------------------------------------------
# 5ï¸âƒ£  Hybrid retriever (simple union + scoring)
# -----------------------------------------------------------
def hybrid_retrieve(query, alpha=0.6):
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query)

    # naive merge (keep top unique docs)
    merged = {d.page_content: d for d in dense_docs + sparse_docs}
    return list(merged.values())

# -----------------------------------------------------------
# 6ï¸âƒ£  Compare results
# -----------------------------------------------------------
query = "checkout payment error E1234"

print("\n=== Query:", query, "===\n")

print("BM25 Retriever:")
for d in bm25_retriever.invoke(query):
    print("-", d.page_content)

print("\nDense Retriever:")
for d in dense_retriever.invoke(query):
    print("-", d.page_content)

print("\nHybrid Retriever:")
result = hybrid_retrieve(query)
for d in result:
    print("-", d.page_content)

# -----------------------------------------------------------
# 7ï¸âƒ£  Optional: plug into LLM generator
# -----------------------------------------------------------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
hybrid_context = "\n".join([d.page_content for d in result])
prompt = f"Context:\n{hybrid_context}\n\nQuestion: What causes error E1234?"
print("\nGenerated Answer:\n", llm.invoke(prompt).content)