import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# הגדרת קבצים
EMBEDDINGS_FILE = "embeddings.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
MODEL_NAME = "all-MiniLM-L6-v2"

# להכריח שימוש רק ב-PyTorch (למנוע שגיאות TensorFlow)
os.environ["USE_TF"] = "0"

# טעינת המערכת: FAISS + טקסטים + מודל אמבדינג
def load_system():
    print("🔄 Loading FAISS index and metadata...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    texts = data["texts"]
    metadata = data["metadata"]
    model = SentenceTransformer(MODEL_NAME)
    return index, texts, metadata, model

# שלב אחזור קטעים רלוונטיים
def retrieve(query, index, texts, metadata, model, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append({
            "text": texts[i],
            "meta": metadata[i]
        })
    return results

# הרצת המערכת עם ג'נרציה
def main():
    index, texts, metadata, model = load_system()

    # יצירת pipeline של שאלות-תשובות
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    while True:
        query = input("\n🔍 Ask something (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # שלב אחזור
        results = retrieve(query, index, texts, metadata, model)

        print("\n📚 Top relevant chunks:")
        for r in results:
            print(f"- [{r['meta']['document']}] {r['text'][:200]}...\n")

        # שלב ג'נרציה
        #combined_context = " ".join([r["text"] for r in results])
        combined_context = results[0]["text"]

        try:
            answer = qa_pipeline(question=query, context=combined_context)
            print("\n💬 Answer:", answer["answer"])
        except Exception as e:
            print("\n⚠️ Error generating answer:", str(e))

if __name__ == "__main__":
    main()
