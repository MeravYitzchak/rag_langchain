import json
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
import os


class FAISSIndexer:
    def __init__(self, input_file: str, index_dir: str, embedding_model: str):
        """
        Initializes the FAISSIndexer with necessary paths and embedding model.
        
        :param input_file: Path to the JSON file containing document chunks
        :param index_dir: Directory to store the FAISS index
        :param embedding_model: Name of the model used for embeddings (e.g., 'all-MiniLM-L6-v2')
        """
        self.input_file = input_file
        self.index_dir = index_dir
        self.embedding_model = embedding_model

    def load_documents(self) -> list:
        """
        Loads documents from the input JSON file.
        
        :return: List of Document objects
        """
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                return [Document(**item) for item in raw]
        except FileNotFoundError:
            print(f"Error: The file {self.input_file} was not found.")
            raise
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the input file.")
            raise

    def build_and_save_index(self, documents: list) -> None:
        """
        Builds a FAISS index from the given documents and saves it locally.
        
        :param documents: List of Document objects to be indexed
        """
        try:
            embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
            vectorstore = FAISS.from_documents(documents, embedding)
            vectorstore.save_local(self.index_dir)
            print(f"FAISS index saved to {self.index_dir}")
        except Exception as e:
            print(f"An error occurred while building or saving the FAISS index: {str(e)}")
            raise

    def run(self) -> None:
        """
        Runs the full process of loading documents, building the FAISS index, and saving it.
        """
        print("Loading documents...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents.")

        print("Building FAISS index...")
        self.build_and_save_index(documents)


if __name__ == "__main__":
    # Define constants
    INPUT_FILE = "chunks.json"
    INDEX_DIR = "faiss_index"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Ensure the output directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Create an instance of the FAISSIndexer and run the process
    indexer = FAISSIndexer(input_file=INPUT_FILE, index_dir=INDEX_DIR, embedding_model=EMBEDDING_MODEL)
    indexer.run()
