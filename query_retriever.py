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

    # QA Pipeline (from HuggingFace)
    qa_pipe = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    llm = HuggingFacePipeline(pipeline=qa_pipe)

    # Build RAG chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

    while True:
        query = input("\n ask question (or exit): ")
        if query.lower() == "exit":
            break

        # A search phase using the retriever to obtain relevant documents
        docs = vectorstore.similarity_search(query, k=3)

        # Creating context from the combination of found documents
        combined_context = " ".join([doc.page_content for doc in docs])

        # Sending the context to the question answering stage
        result = qa_pipe(question=query, context=combined_context)

        print("\n question:", result["answer"])
        print("answer :", docs[0].metadata)


if __name__ == "__main__":
    main()
