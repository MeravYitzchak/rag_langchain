import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIG
EMBEDDINGS_FILE = "embeddings.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["texts"], data["metadata"]

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index, filename):
    faiss.write_index(index, filename)

def main():
    print("Loading embeddings...")
    embeddings, texts, metadata = load_embeddings(EMBEDDINGS_FILE)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index to disk...")
    save_faiss_index(index, FAISS_INDEX_FILE)

    print("Done!")

if __name__ == "__main__":
    main()
