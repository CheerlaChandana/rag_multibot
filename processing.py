import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import BM25Retriever
import os
import tempfile
import shutil
import hashlib
import fitz  # PyMuPDF for PDF image extraction
from pathlib import Path
from utils import clean_text

@st.cache_resource(show_spinner=False)
def load_and_process_documents(uploaded_files, groq_api_key, dirs, process_image):
    """Process documents including embedded images and return LLM components."""
    with st.spinner("Processing documents..."):
        all_chunks = []
        tmp_dir = tempfile.mkdtemp(dir=dirs["temp_dir"])

        try:
            file_hashes = [hashlib.md5(file.getbuffer()).hexdigest() for file in uploaded_files]
            file_names = [file.name for file in uploaded_files]

            for file in uploaded_files:
                filepath = os.path.join(dirs["uploads_dir"], file.name)
                file_hash = hashlib.md5(file.getbuffer()).hexdigest()

                if file_hash not in st.session_state.processed_docs:
                    with open(filepath, "wb") as f:
                        f.write(file.getbuffer())

                    # Load document text
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(filepath)
                        docs = loader.load()

                        # Extract images from PDF using PyMuPDF
                        pdf_doc = fitz.open(filepath)
                        image_texts = []
                        for page_num in range(len(pdf_doc)):
                            page = pdf_doc[page_num]
                            images = page.get_images(full=True)
                            for img_index, img in enumerate(images):
                                xref = img[0]
                                base_image = pdf_doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                temp_image_path = os.path.join(tmp_dir, f"page_{page_num}_img_{img_index}.png")
                                with open(temp_image_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                image_text = process_image(Path(temp_image_path), dirs)
                                image_texts.append(image_text)
                                os.remove(temp_image_path)
                        pdf_doc.close()

                        # Append image text to document content
                        for doc, img_text in zip(docs, image_texts):
                            doc.page_content += f"\n\n[Image Text]: {img_text}"

                    elif file.name.endswith(".txt"):
                        loader = TextLoader(filepath)
                        docs = loader.load()
                    elif file.name.endswith(".csv"):
                        loader = CSVLoader(filepath)
                        docs = loader.load()
                    else:
                        continue

                    for doc in docs:
                        doc.page_content = clean_text(doc.page_content)
                        doc.metadata["source"] = file.name

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    chunks = splitter.split_documents(docs)
                    st.session_state.processed_docs[file_hash] = chunks

                all_chunks.extend(st.session_state.processed_docs[file_hash])

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
                weights=[0.5, 0.5],
            )

            llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=hybrid_retriever,
                memory=memory,
            )

            return llm, qa_chain, hybrid_retriever, file_names
        finally:
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                st.warning(f"Could not clean up temp dir: {e}")


def refine_question(base_question, llm, file_context=None):
    """Refine the question into a single, clear query for a cohesive answer."""
    context_str = f" based on the content of the following files and images: {file_context}" if file_context else ""
    prompt = (
        f"Refine this question into a single, clear, and concise query to be answered in one cohesive response, "
        f"correcting any spelling mistakes or short forms and ensuring it relates to the documents and images{context_str}. "
        f"If no relevant content exists in the documents or images, indicate that and provide a general answer if possible: {base_question}"
    )
    refined_response = llm.invoke(prompt)
    return refined_response.content.strip()