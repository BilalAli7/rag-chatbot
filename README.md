# 🤖 RAG Chatbot — Chat With Your Documents

> *Because your documents deserve better than a keyword search.*

---

## 🧠 Why RAG? The Problem With Traditional AI

Large Language Models are extraordinarily powerful — but they have a fundamental limitation: **they only know what they were trained on**. Ask a general-purpose LLM about your company's internal policy document, your research paper, or your custom dataset, and it will either hallucinate an answer or simply say it doesn't know.

This is where **Retrieval-Augmented Generation (RAG)** changes everything.

RAG is an architectural pattern that supercharges LLMs by grounding them in **your private knowledge**. Instead of relying solely on parametric memory baked into model weights, RAG dynamically retrieves the most relevant context from your documents at query time — and feeds it to the LLM as real, accurate information. The result: **precise, document-grounded answers with zero hallucination on your data**.

Think of it as giving your AI a **photographic memory of exactly the documents you care about**.

---

## ⚙️ How It Works — The Architecture

```
Your Documents (PDF/TXT)
        ↓
  Document Loader
        ↓
  Text Chunking (1000 tokens, 50 overlap)
        ↓
  Embedding Model (all-MiniLM-L6-v2)
        ↓
  FAISS Vector Store (saved locally)
        ↓
  User Query → Semantic Search → Top-K Relevant Chunks
        ↓
  Groq LLM (Llama 3.1) + Context → Final Answer
        ↓
  Answer + Source Citations
```

The pipeline consists of two phases:

### 📥 Ingestion Phase (One-Time Processing)
1. **Document Loading** — PDFs and TXT files are loaded using multiple fallback loaders (`PyPDF`, `pdfplumber`, `PyMuPDF`) to handle any format
2. **Text Chunking** — Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve semantic context across boundaries
3. **Embedding** — Each chunk is converted into a high-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model
4. **Vector Storage** — Embeddings are persisted locally using **FAISS** (Facebook AI Similarity Search) for blazing-fast retrieval

### 💬 Query Phase (Every Conversation)
1. **Semantic Search** — The user's question is embedded and compared against all stored vectors using cosine similarity
2. **Context Retrieval** — The top-K most relevant chunks are fetched from FAISS
3. **Augmented Generation** — The retrieved context + conversation history + question are passed to **Groq's Llama 3.1** via a carefully crafted prompt
4. **Grounded Response** — The LLM generates an answer strictly based on the retrieved context, with source citations shown

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive web UI |
| **LLM** | Groq + Llama 3.1 8B | Ultra-fast AI inference |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Document vectorization |
| **Vector Store** | FAISS (faiss-cpu) | Semantic similarity search |
| **Document Parsing** | PyPDF, pdfplumber, PyMuPDF | Multi-fallback PDF extraction |
| **Orchestration** | LangChain | RAG pipeline management |
| **Memory** | LangChain Message History | Conversation context tracking |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/BilalAli7/rag-chatbot.git
cd rag-chatbot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Run the App

```bash
streamlit run app.py
```

---

## 📖 How to Use

**Step 1 — Upload Your Documents**
Use the sidebar file uploader to upload PDF or TXT files. Files are saved instantly to the `documents/` folder.

**Step 2 — Process Documents**
Click **🔄 Process Documents**. The app will:
- Extract text from all uploaded files
- Split into semantic chunks
- Generate and store embeddings in FAISS

**Step 3 — Start Chatting**
Ask any question about your documents in natural language. The chatbot will:
- Retrieve the most relevant passages
- Generate a grounded, accurate answer
- Show source citations so you can verify

---

## ✨ Key Features

- ✅ **100% Free** — Groq's free tier + open-source embeddings = zero cost
- ✅ **Multi-format Support** — PDF and TXT with multi-fallback parsing
- ✅ **Conversation Memory** — Maintains chat history for follow-up questions
- ✅ **Source Citations** — Every answer shows the source chunks it was based on
- ✅ **Local Embeddings** — Your documents never leave your machine for vectorization
- ✅ **Fast Inference** — Groq delivers ~500 tokens/second on Llama 3.1

---

## 🌐 Deployment

This app is deployed on **Streamlit Community Cloud** (free tier).

Live demo: [your-app-url.streamlit.app](https://your-app-url.streamlit.app)

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                  # Main Streamlit application
├── chat_engine.py          # LLM + RAG query pipeline
├── document_processor.py   # Document loading and chunking
├── embeddings_manager.py   # FAISS vector store management
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── documents/              # Upload your files here
└── vectorstore/            # Auto-generated FAISS index
```

---

<p align="center">Built with ❤️ using LangChain, Groq & Streamlit</p>
