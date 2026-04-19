import streamlit as st
import os
from document_processor import DocumentProcessor
from embeddings_manager import EmbeddingsManager
from chat_engine import ChatEngine

st.set_page_config(
    page_title="RAG Chatbot with Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Chat with Your Documents using Groq</p>', unsafe_allow_html=True)

docs_folder = "documents"
os.makedirs(docs_folder, exist_ok=True)

# ✅ Cache the embeddings manager so model loads only once
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_manager():
    return EmbeddingsManager()

# ✅ Cache the vectorstore
@st.cache_resource(show_spinner="Loading vector store...")
def load_vectorstore(_em):
    return _em.load_vectorstore()

with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(docs_folder, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) saved!")

    existing_files = [f for f in os.listdir(docs_folder) if f.endswith(('.pdf', '.txt'))]
    if existing_files:
        st.success(f"✅ {len(existing_files)} document(s) ready")
        with st.expander("View Documents"):
            for file in existing_files:
                col1, col2 = st.columns([3, 1])
                col1.write(f"📄 {file}")
                if col2.button("🗑️", key=f"del_{file}"):
                    os.remove(os.path.join(docs_folder, file))
                    st.rerun()
    else:
        st.warning("⚠️ No documents found. Upload files above.")

    st.markdown("---")

    if st.button("🔄 Process Documents", use_container_width=True):
        disk_files = [f for f in os.listdir(docs_folder) if f.endswith(('.pdf', '.txt'))]
        if not disk_files:
            st.error("❌ No documents to process! Upload files first.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    processor = DocumentProcessor()
                    chunks = processor.process_documents()

                    if not chunks:
                        st.error("❌ Documents found but no content could be extracted!")
                    else:
                        em = get_embeddings_manager()
                        em.delete_vectorstore()
                        vectorstore = em.create_vectorstore(chunks)

                        if vectorstore:
                            st.success(f"✅ Processed {len(chunks)} chunks!")
                            st.balloons()
                            # Clear caches so vectorstore reloads
                            load_vectorstore.clear()
                            if 'chat_engine' in st.session_state:
                                del st.session_state.chat_engine
                        else:
                            st.error("❌ Failed to create vectorstore")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if 'chat_engine' in st.session_state:
            st.session_state.chat_engine.clear_history()
        st.success("Chat cleared!")
        st.rerun()

    st.markdown("---")

    with st.expander("ℹ️ Model Info"):
        st.write("**Model:** Llama 3.1")
        st.write("**Provider:** Groq")
        st.write("**Cost:** 100% FREE")
        st.write("**Privacy:** Local embeddings")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chat engine with caching
if "chat_engine" not in st.session_state:
    try:
        em = get_embeddings_manager()
        vectorstore = load_vectorstore(em)
        if vectorstore:
            st.session_state.chat_engine = ChatEngine(vectorstore)
            st.success("✅ AI ready!")
        else:
            st.info("📄 Upload and process documents to begin.")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(doc.page_content[:300] + "...")
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    if "chat_engine" not in st.session_state:
        st.error("⚠️ Please process documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chat_engine.ask(prompt)
                answer = response["answer"]
                sources = response["sources"]
                st.markdown(answer)
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc.page_content[:300] + "...")
                            st.markdown("---")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Powered by Groq 🤖 | Built with LangChain & Streamlit</p>",
    unsafe_allow_html=True
)