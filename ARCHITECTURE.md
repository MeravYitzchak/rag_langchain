# 📐 RAG QA System – Architecture Overview

This document provides an overview of the architecture for the Retrieval-Augmented Generation (RAG) system used to answer questions based on internal company documents.

---

## 🧱 System Components

### 1. Document Preprocessing
- **Input:** `.txt` files from the `./documents` folder
- **Process:**
  - Clean the text
  - Split into chunks (~100 words)
  - Store each chunk with metadata (`document` name, `chunk_id`, `text`)

### 2. Embedding & Indexing
- **Embedding Model:** `all-MiniLM-L6-v2` from Sentence Transformers
- **Steps:**
  - Generate vector embeddings for each chunk
  - Store the embeddings in a FAISS index (`faiss_index.bin`)
  - Save text + metadata to `embeddings.pkl`

### 3. Query Retrieval
- **Process:**
  - User types a question
  - The system embeds the query
  - FAISS retrieves top relevant chunks
  - Relevant chunks are shown as preview

### 4. Answer Generation
- **Model:** A QA model like `distilbert-base-uncased-distilled-squad` from Hugging Face
- **Process:**
  - Use retrieved chunks as context
  - Generate an answer to the question
  - Return the final answer to the user

---

## 🔄 Workflow Summary
[Raw .txt files] 
      ↓
[Preprocessing → Chunking]
      ↓
[Embedding with SentenceTransformer]
      ↓
[FAISS Index Building]
      ↓
[User Query]
      ↓
[Embedding → Retrieval → Generation]
      ↓
[Answer Displayed]



rag-qa-system/
│
├── documents/                # Input .txt files
├── processed_documents.json  # Chunked and cleaned output
├── embeddings.pkl            # Saved embeddings and metadata
├── faiss_index.bin           # Vector store
├── preprocess.py             # Step 1: Clean and chunk docs
├── build_faiss_index.py      # Step 2: Generate embeddings and index
├── query_retriever.py        # Step 3: Retrieve and answer
├── README.md                 # Setup & usage instructions
└── ARCHITECTURE.md           # This file

## 🛠️ Technologies Used

-- Python 3.11
-- sentence-transformers
-- faiss-cpu
-- transformers (for QA model)
-- scikit-learn, numpy, json, pickle

## 🚀 Possible Improvements
-- Add a web or chatbot UI
-- Use a local LLM (like LLaMA or Mistral) for offline generation
-- Integrate vector databases like Pinecone or Weaviate
-- Handle multilingual documents



