import json
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# CONFIGURATION
INPUT_FILE = "processed_documents.json"
OUTPUT_FILE = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_chunks(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def build_embeddings(chunks, model_name):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [chunk["text"] for chunk in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings, texts, chunks

def save_embeddings(embeddings, texts, chunks, output_file):
    data = {
        "embeddings": embeddings,
        "texts": texts,
        "metadata": chunks
    }
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {output_file}")

def main():
    chunks = load_chunks(INPUT_FILE)
    embeddings, texts, metadata = build_embeddings(chunks, MODEL_NAME)
    save_embeddings(embeddings, texts, metadata, OUTPUT_FILE)

if __name__ == "__main__":
    main()
