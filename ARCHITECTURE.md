
# RAG QA System – Architecture Overview

This document provides an overview of the architecture for the Retrieval-Augmented Generation (RAG) system used to answer questions based on internal company documents.

---

## System Components

### 1. **Document Preprocessing**

* **Input:** `.txt` files from the `./documents` folder  
* **Process:**
  - Normalize and clean text
  - Split into short chunks (~100 tokens)
  - Store results in `processed_documents.json`

### 2. **Embedding & Indexing**

* **Embedding Model:** `all-MiniLM-L6-v2` via `sentence-transformers`  
* **Steps:**
  - Generate embeddings for each chunk
  - Store vector representations using **FAISS** (`faiss_index/`)
  - Persist metadata alongside embeddings (e.g., chunk ID, source document)

### 3. **Query Retrieval**

* **LangChain Retriever:**
  - Embed user question using the same embedding model
  - Perform similarity search with FAISS
  - Return top-K relevant document chunks

### 4. **Answer Generation**

* **Model:** `distilbert-base-uncased-distilled-squad` via Hugging Face pipeline  
* **Process:**
  - Use retrieved chunks as context
  - Generate answer using question-answering pipeline
  - Return the result + metadata from top document

---

## Workflow Summary

1. **Preprocessing**: Clean and split documents into chunks  
2. **Embedding**: Generate semantic vectors using Hugging Face model  
3. **Indexing**: Store vectors in FAISS for fast retrieval  
4. **Querying**: Embed query → retrieve top chunks from FAISS  
5. **Answering**: Generate answer using a Hugging Face QA model

---

## Directory Structure


rag-qa-system/
│
├── documents/                # Input .txt files
├── processed\_documents.json # Cleaned & chunked documents
├── faiss\_index/             # FAISS vector index
│
├── preprocessing.py          # Clean + split raw documents
├── build\_embedding.py       # Embedding and FAISS indexing
├── query\_query_retriever.py # LangChain-powered question answering
│
├── ARCHITECTURE.md           # This file
├── README.md                 # Setup and usage instructions
└── requirements.txt          # All dependencies

---

## Technologies Used

* Python 3.11  
* [LangChain](https://www.langchain.com/)  
* sentence-transformers  
* FAISS (Facebook AI Similarity Search)  
* Hugging Face Transformers  
* scikit-learn, NumPy, JSON, Pickle  

---

## Why LangChain?

LangChain allows seamless integration of:

- FAISS-based vector retrieval  
- Hugging Face pipelines  
- Source tracking  
- Chain abstraction for modular design

Its use simplified the pipeline while keeping it modular and scalable.

---
##  Technology Choices – Summary

This system utilizes several carefully selected technology components to ensure high performance, ease of setup and use, and efficient data processing. Below are the key components of the system and the reasons for choosing them:

# Embedding Model: all-MiniLM-L6-v2 (Sentence-Transformers)
The model selected for generating embeddings of document chunks is all-MiniLM-L6-v2 from the Sentence-Transformers library.
This model strikes an ideal balance between speed and accuracy, and is small in size, allowing for fast processing even on resource-constrained systems. It is particularly well-suited for semantic search tasks in large document corpora, like in this system.

# Vector Database: FAISS
FAISS - for fast vector storage and retrieval. It provides a lightweight and efficient solution for storing vector indices and performing similarity searches. FAISS excels in fast retrieval, ensuring that the system will run smoothly even as the corpus grows.

# Answering Model: distilbert-base-uncased-distilled-squad
The QA model selected for generating answers is distilbert-base-uncased-distilled-squad.
This is an extractive QA model from Hugging Face, specializing in question-answering from text. It is well-suited for questions with specific answers drawn from the context, providing a balance of accuracy and efficiency while using minimal resources.

# Framework: LangChain
For orchestrating the RAG pipeline, LangChain was used.
LangChain is a development framework that simplifies the integration between various components of the system, such as the retriever, language model, and QA model. It provides efficient tools for managing chains, which simplifies the development process and offers greater control over each step in the pipeline. Although simpler alternatives could be used for a basic system, LangChain provides flexibility and structure that helps in scaling and handling more complex use cases in the future.


## Deployment Considerations

* **On-premise:** Best for internal/private company data. Lightweight and secure.
* **Cloud-based options:** Scale better; can replace FAISS with managed vector DB (e.g., Pinecone, Weaviate).
* **Optional Enhancements:**

  * Add a FastAPI or Streamlit UI
  * Support multiple languages
  * Switch to a local LLM (e.g., Mistral GGUF) for full offline inference

---

## Summary

This RAG system provides a lean, explainable pipeline for QA over internal documents — leveraging open-source tools and minimal infrastructure.
Using LangChain and Hugging Face together balances ease-of-use with full local control.

