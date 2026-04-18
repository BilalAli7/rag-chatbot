import os
import shutil
import time
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma


class EmbeddingsManager:
    def __init__(self, persist_directory: str = "vectorstore"):
        self.persist_directory = persist_directory
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
        )
        print("✅ Embedding model loaded!")

    def _get_client(self):
        return chromadb.PersistentClient(path=self.persist_directory)

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        if not chunks:
            print("❌ No chunks to create vectorstore!")
            return None

        # Always delete old one before creating new
        self.delete_vectorstore()
        os.makedirs(self.persist_directory, exist_ok=True)

        print(f"\n🔄 Creating embeddings for {len(chunks)} chunks...")
        client = self._get_client()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=client,
            collection_name="documents"
        )
        print("✅ Vector store created!")
        return vectorstore

    def load_vectorstore(self) -> Chroma:
        if not os.path.exists(self.persist_directory):
            return None
        try:
            print("📂 Loading existing vector store...")
            client = self._get_client()
            vectorstore = Chroma(
                client=client,
                collection_name="documents",
                embedding_function=self.embeddings
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