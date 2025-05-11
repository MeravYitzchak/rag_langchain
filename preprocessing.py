import os
import json
import re
from pathlib import Path
from typing import List

# CONFIGURATION
DOCS_PATH = "./documents"  # folder with .txt files
CHUNK_SIZE = 100  # approx. word count per chunk (adjust as needed)
OUTPUT_FILE = "processed_documents.json"

# Load documents from folder
def load_documents(folder_path: str) -> List[dict]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                content = f.read().strip()
                documents.append({
                    "title": filename.replace(".txt", ""),
                    "content": content
                })
    return documents

# Clean unnecessary whitespace
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Split the document into paragraphs
def split_document_into_chunks(text: str, chunk_size: int) -> List[str]:
    paragraphs = text.split("\n\n")  # Split by double line breaks (paragraphs)
    
    chunks = []
    current_chunk = ""
    
    # Combine paragraphs into chunks based on chunk size (in words)
    for paragraph in paragraphs:
        words_in_paragraph = paragraph.split()
        
        # If adding this paragraph does not exceed chunk size, append it
        if len(current_chunk.split()) + len(words_in_paragraph) <= chunk_size:
            current_chunk += "\n\n" + paragraph
        else:
            # Save the current chunk and start a new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
    
    # Append the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Preprocess documents by cleaning and chunking
def preprocess_documents(documents: List[dict], chunk_size: int) -> List[dict]:
    processed = []
    for doc in documents:
        cleaned = clean_text(doc["content"])  # Clean the text
        chunks = split_document_into_chunks(cleaned, chunk_size)  # Split into chunks
        for i, chunk in enumerate(chunks):
            processed.append({
                "document": doc["title"],
                "chunk_id": f"{doc['title']}_chunk_{i+1}",
                "text": chunk
            })
    return processed

# Save processed data to JSON file
def save_to_json(data: List[dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("Loading documents...")
    docs = load_documents(DOCS_PATH)
    print(f"Found {len(docs)} documents.")

    print("Preprocessing and chunking...")
    processed = preprocess_documents(docs, CHUNK_SIZE)
    print(f"Generated {len(processed)} chunks.")

    print(f"Saving to {OUTPUT_FILE}...")
    save_to_json(processed, OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    main()
