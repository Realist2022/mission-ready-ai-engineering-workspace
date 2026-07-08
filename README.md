# Mission-Ready AI Engineering

A hands-on learning workspace covering practical AI engineering: embeddings, vector
databases, structured-output validation, Retrieval-Augmented Generation (RAG), and an
end-to-end computer-vision + agentic-RAG project for detecting crop blight.

## Project Structure

```
dataSet/              # Training images (blight / healthy leaf classes)
playground/           # End-to-end blight detection experiments
  blightTester.py                # Baseline vision classifier
  blightTesterWithVDB.py         # Classifier + Chroma vector DB
  blightTesterHybridRag.py       # Classifier + hybrid RAG
  blightTestWithRagOop.py        # OOP vision model + agentic RAG (LangGraph)
  blight_model.pth               # Saved model checkpoint
tester/               # Utilities (e.g. GPU availability check)
week1/                # Week 1 exercises
week2/
  embedding/          # Sentence-transformer embeddings & chunking
  validation/         # JSON schema / constrained output validation
  vectorDBs/          # Chroma vector database examples
week3/
  day1/               # RAG fundamentals
  day2/               # Hybrid RAG (BM25 + FAISS)
  day3/ day4/         # Further RAG work
```

## Prerequisites

- Python 3.10+
- An OpenAI API key (for the RAG/agent scripts that use `ChatOpenAI`)
- A GPU is optional; the vision code falls back to CPU (or Apple MPS) automatically

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API key:

   ```
   OPENAI_API_KEY=your-key-here
   ```

## Running the Blight Detection Project

The main example combines a ResNet-50 vision classifier with an agentic RAG advisor:

```powershell
python playground/blightTestWithRagOop.py
```

- Set `RETRAIN = True` in the script to train from scratch on the current dataset.
- With `RETRAIN = False` (default) it loads `playground/blight_model.pth` if present.
- Place a `test_leaf.jpg` in the project root to run a prediction; if blight is
  detected, the agentic RAG pipeline returns mitigation strategies.

## Notes

- `chromadb` and `faiss-cpu` provide vector search; swap `faiss-cpu` for `faiss-gpu`
  if you have a compatible CUDA setup.
- The RAG scripts require network access and a valid `OPENAI_API_KEY`.
