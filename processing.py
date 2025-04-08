import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import os
import tempfile
import shutil
import hashlib
from pathlib import Path
import pytesseract
from PIL import Image
from utils import clean_text
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Custom local embedding class (no API needed)
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

@st.cache_resource(show_spinner=False)
def load_and_process_documents(uploaded_files, groq_api_key, dirs):
    with st.spinner("Processing documents..."):
        all_chunks = []
        tmp_dir = dirs["temp_dir"]

        try:
            file_hashes = [hashlib.md5(file.getbuffer()).hexdigest() for file in uploaded_files]
            file_names = [file.name for file in uploaded_files]
            for file in uploaded_files:
                filepath = os.path.join(dirs["uploads_dir"], file.name)
                file_hash = hashlib.md5(file.getbuffer()).hexdigest()
                if file_hash not in st.session_state.processed_docs:
                    with open(filepath, "wb") as f:
                        f.write(file.getbuffer())
                    
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(filepath)
                        docs = loader.load()
                    elif file.name.endswith((".txt", ".csv")):
                        loader = TextLoader(filepath) if file.name.endswith(".txt") else CSVLoader(filepath)
                        docs = loader.load()
                    elif file.name.endswith((".png", ".jpg", ".jpeg")):
                        # Enhanced OCR with pre-processing
                        image = Image.open(filepath)
                        # Convert to grayscale and apply noise reduction (optional)
                        image = image.convert('L')  # Grayscale
                        # Optional: Apply thresholding or noise reduction
                        text = pytesseract.image_to_string(image, config='--psm 6')  # PSM 6 for better accuracy on single uniform blocks
                        # Clean the extracted text
                        cleaned_text = clean_text(text)
                        docs = [type('Document', (), {'page_content': cleaned_text, 'metadata': {'source': file.name, 'type': 'image'}})]  # Mark as image
                    else:
                        continue

                    for doc in docs:
                        doc.page_content = clean_text(doc.page_content)
                        doc.metadata['source'] = file.name
                        if file.name.endswith((".png", ".jpg", ".jpeg")):
                            doc.metadata['type'] = 'image'  # Tag documents from images

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    chunks = splitter.split_documents(docs)
                    st.session_state.processed_docs[file_hash] = chunks

                all_chunks.extend(st.session_state.processed_docs[file_hash])

            # Use local embeddings
            embeddings = LocalSentenceTransformerEmbeddings("all-MiniLM-L6-v2")

            faiss_index_path = dirs["cache_dir"] / "faiss_index"
            cache_valid = os.path.exists(faiss_index_path) and st.session_state.get("last_file_hashes") == file_hashes

            if cache_valid:
                faiss_store = FAISS.load_local(str(faiss_index_path), embeddings, allow_dangerous_deserialization=True)
            else:
                faiss_store = FAISS.from_documents(all_chunks, embeddings)
                faiss_store.save_local(str(faiss_index_path))
                st.session_state.last_file_hashes = file_hashes

            bm25_retriever = BM25Retriever.from_documents(all_chunks)
            bm25_retriever.k = 3

            hybrid_retriever = EnsembleRetriever(
                retrievers=[faiss_store.as_retriever(search_kwargs={"k": 3}), bm25_retriever],
                weights=[0.5, 0.5]
            )

            llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=hybrid_retriever,
                memory=memory
            )

            return llm, qa_chain, hybrid_retriever, file_names

        finally:
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                st.warning(f"Could not clean up temp dir: {e}")

def refine_question(base_question, llm, file_context=None):
    """Refine the question into a single, clear query for a cohesive answer."""
    context_str = f" based on the content of the following files: {file_context}" if file_context else ""
    # Indicate that some documents might be images
    document_types = " including text extracted from images and other documents" if any(fn.endswith((".png", ".jpg", ".jpeg")) for fn in (file_context or [])) else ""
    prompt = f"Refine this question into a single, clear, and concise query to be answered in one cohesive response, correcting any spelling mistakes or short forms and ensuring it relates to the documents{context_str}{document_types}. If no relevant content exists in the documents, indicate that and provide a general answer if possible: {base_question}"
    refined_response = llm.invoke(prompt)
    return refined_response.content.strip()
