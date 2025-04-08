import streamlit as st
from ui import render_ui
from processing import load_and_process_documents, refine_question
from utils import clean_text, store_feedback, generate_wordcloud, export_chat_to_pdf
import os

if __name__ == "__main__":
    st.set_page_config(page_title="📄 Advanced Multi-Doc Chat with Grok", layout="wide")
    os.environ['GROQ_API_KEY']="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"
    render_ui(load_and_process_documents, refine_question, clean_text, store_feedback, generate_wordcloud, export_chat_to_pdf)
