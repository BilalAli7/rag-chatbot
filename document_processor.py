import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, docs_folder: str = "documents"):
        self.docs_folder = docs_folder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,    
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        documents = []
        os.makedirs(self.docs_folder, exist_ok=True)

        files = os.listdir(self.docs_folder)
        if not files:
            print("⚠️ No documents found!")
            return documents

        print(f"\n📚 Loading documents...")

        for filename in files:
            file_path = os.path.join(self.docs_folder, filename)

            try:
                if filename.lower().endswith('.pdf'):
                    docs = self._load_pdf(file_path)
                    documents.extend(docs)
                    print(f"✅ Loaded PDF: {filename} ({len(docs)} pages)")

                elif filename.lower().endswith('.txt'):
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✅ Loaded TXT: {filename}")

                else:
                    print(f"⏭️ Skipped: {filename}")

            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")

        print(f"\n📊 Total pages/docs loaded: {len(documents)}")
        return documents

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Try multiple PDF loaders until one works."""
        docs = []

        # Method 1: PyPDFLoader
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if docs and any(d.page_content.strip() for d in docs):
                return [d for d in docs if d.page_content.strip()]
        except Exception as e:
            print(f"⚠️ PyPDFLoader failed: {e}")

        # Method 2: pdfplumber via manual extraction
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(Document(
                            page_content=text,
                            metadata={"source": file_path, "page": i}
                        ))
            if pages:
                return pages
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}")

        # Method 3: PyMuPDF (fitz)
        try:
            import fitz
            pages = []
            pdf = fitz.open(file_path)
            for i, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    pages.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i}
                    ))
            pdf.close()
            if pages:
                return pages
        except Exception as e:
            print(f"⚠️ PyMuPDF failed: {e}")

        print(f"❌ All PDF loaders failed for {file_path}")
        return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def process_documents(self) -> List[Document]:
        docs = self.load_documents()
        return self.split_documents(docs)