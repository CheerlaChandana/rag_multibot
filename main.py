import streamlit as st
from ui import render_ui
from processing import load_and_process_documents, refine_question
from utils import (
    clean_text,
    store_feedback,
    generate_wordcloud,
    export_chat_to_pdf,
    process_image,
)

if __name__ == "__main__":
    st.set_page_config(page_title="📄 Advanced Multi-Doc Chat with Groq", layout="wide")
    render_ui(
        load_and_process_documents,
        refine_question,
        clean_text,
        store_feedback,
        generate_wordcloud,
        export_chat_to_pdf,
        process_image,
    )