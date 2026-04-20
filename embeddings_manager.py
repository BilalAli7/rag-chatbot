import os
import shutil
import time
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class EmbeddingsManager:
    def __init__(self, persist_directory: str = "vectorstore"):
        self.persist_directory = persist_directory
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
            cache_folder="model_cache"
        )
        print("✅ Embedding model loaded!")

    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        if not chunks:
            print("❌ No chunks to create vectorstore!")
            return None
        self.delete_vectorstore()
        os.makedirs(self.persist_directory, exist_ok=True)
        print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(self.persist_directory)
        print("✅ Vector store created!")
        return vectorstore

    def load_vectorstore(self) -> FAISS:
        if not os.path.exists(self.persist_directory):
            return None
        try:
            print("📂 Loading existing vector store...")
            vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vector store loaded!")
            return vectorstore
        except Exception as e:
            print(f"❌ Failed to load vectorstore: {e}")
            return None

    def delete_vectorstore(self):
        if not os.path.exists(self.persist_directory):
            return
        for attempt in range(5):
            try:
                shutil.rmtree(self.persist_directory)
                print("🗑️ Vector store deleted!")
                return
            except:
                time.sleep(1)
