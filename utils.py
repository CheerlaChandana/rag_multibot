import re
import json
from datetime import datetime
import os
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st

def clean_text(text):
    return re.sub(r'[\u0000-\u001F\u007F-\u009F\uD800-\uDFFF]', '', text)

def store_feedback(question, answer, feedback, base_dir):
    feedback_dir = base_dir / "feedback"
    feedback_dir.mkdir(exist_ok=True)
    feedback_file = feedback_dir / "feedback_log.json"
    
    feedback_data = {
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        if feedback_file.exists():
            with open(feedback_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(feedback_data)
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Could not save feedback: {e}")

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_str

def export_chat_to_pdf(base_dir):
    pdf_path = base_dir / f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for i in range(0, len(st.session_state.chat_history), 2):
        if i in st.session_state.chat_deleted:
            continue
        user_msg = st.session_state.chat_history[i]
        bot_msg = st.session_state.chat_history[i + 1] if i + 1 < len(st.session_state.chat_history) else None
        story.append(Paragraph(f"You ({user_msg[2]}): {user_msg[1]}", styles["Normal"]))
        if bot_msg:
            story.append(Paragraph(f"Bot ({bot_msg[2]}): {bot_msg[1]}", styles["Normal"]))
        story.append(Paragraph("<br/>", styles["Normal"]))
    doc.build(story)
    return pdf_path
