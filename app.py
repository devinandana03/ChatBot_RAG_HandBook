import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()

# --- Page Config & Styling ---
st.set_page_config(page_title="AI Handbook Pro", layout="wide", page_icon="📚")
st.title("📚 AI Handbook Pro: Multi-Doc Intelligence")
st.markdown("---")

# --- Helper Functions ---
@st.cache_resource
def get_embeddings():
    """Loads the local embedding model once."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_multiple_pdfs(pdf_files):
    """Extracts text and metadata from all uploaded PDFs."""
    docs = []
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(
                    page_content=text, 
                    metadata={"source": pdf.name, "page": i + 1}
                ))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, get_embeddings())
    return vector_store

# --- Sidebar: Document & Memory Management ---
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_files = st.file_uploader("Upload AI Handbooks", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 Index Documents"):
        if uploaded_files:
            with st.spinner("Analyzing knowledge base..."):
                st.session_state.vector_store = process_multiple_pdfs(uploaded_files)
                # Initialize memory with output_key to match RetrievalQA result
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True,
                    output_key="result"
                )
                st.success(f"Indexed {len(uploaded_files)} files!")
        else:
            st.error("Please upload PDFs first.")

    st.divider()
    
    # CREATIVE FEATURE: Smart Summary Button
    st.subheader("💡 Creative Tools")
    if st.button("📝 Quick Summary"):
        if "vector_store" in st.session_state:
            with st.spinner("Generating briefing..."):
                # Pull 5 most relevant chunks regarding 'overview'
                docs = st.session_state.vector_store.similarity_search("General overview and main goals", k=5)
                context = " ".join([d.page_content for d in docs])
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
                summary = llm.invoke(f"Based on these handbook excerpts, provide a 5-bullet summary of the main points: {context}")
                st.info(summary.content)
        else:
            st.warning("Index files first!")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.rerun()

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your handbooks..."):
    if "vector_store" not in st.session_state:
        st.error("Please upload and index documents in the sidebar first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant response
        with st.chat_message("assistant"):
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile", 
                temperature=0,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            # Grounding prompt from separate file
            from prompts import RAG_PROMPT
            qa_prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["context", "question"])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_prompt}
            )
            
            with st.spinner("Scanning documents..."):
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                sources = result["source_documents"]

            st.markdown(answer)
            
            # CREATIVE FEATURE: Source Citation UI
            with st.expander("📍 See Page Citations"):
                unique_sources = set()
                for doc in sources:
                    source_str = f"**{doc.metadata['source']}** (Page {doc.metadata['page']})"
                    if source_str not in unique_sources:
                        st.write(source_str)
                        unique_sources.add(source_str)

            st.session_state.messages.append({"role": "assistant", "content": answer})