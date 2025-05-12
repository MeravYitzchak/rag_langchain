import os
import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir

    def load_documents(self):
        """טוען את כל המסמכים בתיקייה"""
        all_documents = []
        for file in os.listdir(self.docs_dir):
            if file.endswith(".txt"):
                path = os.path.join(self.docs_dir, file)
                loader = TextLoader(path, encoding="utf-8")
                documents = loader.load()
                all_documents.extend(documents)
        return all_documents


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents):
        """פיצול טקסטים למסמכים קטנים יותר"""
        return self.splitter.split_documents(documents)


class ChunksSaver:
    def __init__(self, output_file: str):
        self.output_file = output_file

    def save_chunks(self, chunks):
        """שומר את המקטעים בקובץ JSON"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump([d.dict() for d in chunks], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {self.output_file}")


class DocumentProcessor:
    def __init__(self, docs_dir: str, output_file: str):
        self.docs_dir = docs_dir
        self.output_file = output_file

        # Create objects of the loader, splitter, and saver
        self.loader = DocumentLoader(docs_dir)
        self.splitter = TextSplitter()
        self.saver = ChunksSaver(output_file)

    def process_documents(self):
        """מעבד את כל המסמכים"""
        # Step 1: Load documents
        documents = self.loader.load_documents()

        # Step 2: Split the documents into chunks
        chunks = self.splitter.split_documents(documents)

        # Step 3: Save the chunks to a file
        self.saver.save_chunks(chunks)


if __name__ == "__main__":
    # Set your directories and output file names
    DOCS_DIR = "./documents"
    OUTPUT_FILE = "chunks.json"

    # Process the documents
    processor = DocumentProcessor(DOCS_DIR, OUTPUT_FILE)
    processor.process_documents()
