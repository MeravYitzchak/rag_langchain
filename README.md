

# RAG QA System with LangChain

A minimal Retrieval-Augmented Generation (RAG) system for answering questions based on internal documents using LangChain, Hugging Face, and FAISS.

---

## What It Does

This project allows you to ask natural language questions about internal company documents (e.g., policies, meeting notes).  
It works by retrieving relevant context via vector search and generating grounded answers using a Hugging Face QA model.

---

## Project Structure

```

rag-qa-system/
│
├── documents/                 # Input .txt files (internal documents)
├── processed\_documents.json  # Cleaned and chunked passages
├── faiss\_index/              # FAISS vector store
│
├── preprocessing.py           # Step 1: Clean and split documents
├── build\_embedding.py        # Step 2: Embed text and build FAISS index
├── query\_retriever.py        # Step 3: Ask questions via terminal
│
├── ARCHITECTURE.md            # System design and tech decisions
├── README.md                  # You're here
└── requirements.txt           # Dependencies

````

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MeravYitzchak/rag-question-system
cd rag-qa-system
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install langchain
pip install langchain-community
pip install sentence-transformers
pip install faiss-cpu
pip install transformers
pip install langchain_huggingface
pip freeze > requirements.txt
```

---

## ▶ How to Run

### 🔹 Step 1: Preprocess Documents

Prepare the raw `.txt` files (cleaning, chunking):

```bash
python preprocessing.py
```

### 🔹 Step 2: Build Embeddings and FAISS Index

Generate vector embeddings and create the index:

```bash
python build_embedding.py
```

### 🔹 Step 3: Start Interactive Question Interface

Ask questions using a Hugging Face QA model and the vector retriever:

```bash
python query_retriever.py
```

```bash
python example_queries.py
```

---

## Sample Questions

You can try asking things like:

1. What were the main topics discussed at the engineering synchronization meeting?
2. What are the rules that apply to working from home?
3. What are the specific requirements for creating a strong password according to the company's IT security policy?
4. How many paid vacation days are employees entitled to each year?
5. Is there a specific deadline for completing the mandatory legal forms?

---

## Security & Deployment Notes

* `allow_dangerous_deserialization=True` is used for loading FAISS index. Only use if you're sure the index is trusted.
* For production, consider vector DBs like Pinecone or Weaviate and deploying via FastAPI/Docker.
* No data is sent externally — QA model runs locally via Hugging Face Transformers.
* Ensure access controls and model isolation in real deployments for data security.

---

## Requirements Recap

```bash
langchain
langchain-community
sentence-transformers
faiss-cpu
transformers
```

