from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.schema import Document

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    # Load FAISS & Embedding
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(INDEX_DIR, embedding, allow_dangerous_deserialization=True)

    # QA Pipeline (מודל מבית HuggingFace)
    qa_pipe = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    llm = HuggingFacePipeline(pipeline=qa_pipe)

    # Build RAG chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

    while True:
        query = input("\n ask question (or exit): ")
        if query.lower() == "exit":
            break

        # שלב של חיפוש באמצעות ה-retriever לקבלת מסמכים רלוונטיים
        docs = vectorstore.similarity_search(query, k=3)

        # יצירת קונטקסט משילוב המסמכים שנמצאו
        combined_context = " ".join([doc.page_content for doc in docs])

        # שליחת הקונטקסט לשלב ה-question answering
        result = qa_pipe(question=query, context=combined_context)

        print("\n question:", result["answer"])
        print("answer :", docs[0].metadata)


if __name__ == "__main__":
    main()
