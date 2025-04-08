import streamlit as st
from datetime import datetime
import os
from pathlib import Path
import shutil
import json
import tempfile

def setup_directories():
    """Set up working directories on D: drive with fallback to system temp."""
    try:
        if os.path.exists("D:"):
            base_dir = Path("D:/streamlit_docs")
        else:
            base_dir = Path(tempfile.gettempdir()) / "streamlit_docs"
            st.warning("D: drive not found - using system temp directory instead")

        temp_dir = base_dir / "temp_files"
        cache_dir = base_dir / "cache"
        uploads_dir = base_dir / "uploads"

        for directory in [base_dir, temp_dir, cache_dir, uploads_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        return {
            "base_dir": base_dir,
            "temp_dir": temp_dir,
            "cache_dir": cache_dir,
            "uploads_dir": uploads_dir,
        }
    except Exception as e:
        st.error(f"Could not set up directories: {e}")
        raise


def check_disk_space(dirs):
    """Check available space on working drive."""
    try:
        if os.path.exists("D:"):
            drive = "D:"
        else:
            drive = dirs["base_dir"].drive

        usage = shutil.disk_usage(str(drive))
        available_gb = usage.free / (1024**3)
        return available_gb
    except Exception as e:
        st.error(f"Could not check disk space: {e}")
        return None


def cleanup_old_files(directory):
    """Delete all files in directory."""
    deleted_count = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
                deleted_count += 1
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
                deleted_count += 1
        except Exception as e:
            st.warning(f"Could not delete {filename}: {e}")
    return deleted_count


def render_ui(
    load_and_process_documents,
    refine_question,
    clean_text,
    store_feedback,
    generate_wordcloud,
    export_chat_to_pdf,
    process_image,
):
    # CSS Styling
    st.markdown(
        """
        <style>
        :root {
            --primary-bg: #f0f4f8;
            --button-bg: #3b82f6;
            --button-hover: #2563eb;
            --user-msg-bg: #e0f2fe;
            --bot-msg-bg: #dcfce7;
            --text-color: #1f2937;
        }
        [data-baseweb="baseui"] {
            background-color: var(--primary-bg);
            color: var(--text-color);
        }
        .stButton > button {
            background-color: var(--button-bg);
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s ease;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: var(--button-hover);
        }
        .chat-message {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.6;
            color: #111827;
        }
        .user-message {
            background-color: var(--user-msg-bg);
            border-left: 6px solid #0284c7;
        }
        .bot-message {
            background-color: var(--bot-msg-bg);
            border-left: 6px solid #16a34a;
        }
        .stExpander {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
        .debug-message {
            background-color: #fef3c7;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 6px solid #f59e0b;
            font-family: 'Courier New', monospace;
            color: #78350f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤖 Advanced Chat with Documents and Images")
    st.markdown(
        "Explore your documents (including embedded images) and standalone images with an interactive AI-powered chat interface!",
        unsafe_allow_html=True,
    )

    # Authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        with st.container():
            st.subheader("🔒 Login")
            password = st.text_input("Enter Password:", type="password", key="password_input")
            if st.button("Login"):
                if password == "admin123":  # Replace with secure authentication
                    st.session_state.authenticated = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Incorrect password!")
        return

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_deleted" not in st.session_state:
        st.session_state.chat_deleted = set()
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "hybrid_retriever" not in st.session_state:
        st.session_state.hybrid_retriever = None
    if "file_names" not in st.session_state:
        st.session_state.file_names = []
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = {}
    if "last_file_hashes" not in st.session_state:
        st.session_state.last_file_hashes = None
    if "file_context" not in st.session_state:
        st.session_state.file_context = "all uploaded files and images"

    dirs = setup_directories()
    os.environ['GROQ_API_KEY'] = "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"
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Set your GROQ_API_KEY in .streamlit/secrets.toml or env variable.")
        return

    # Sidebar
    with st.sidebar:
        st.subheader("💾 Storage Management")
        if st.button("Check Disk Space"):
            space = check_disk_space(dirs)
            if space is not None:
                st.info(f"Available space: {space:.1f} GB")
        if st.button("Clean Up Old Files"):
            with st.spinner("Cleaning up..."):
                deleted_uploads = cleanup_old_files(dirs["uploads_dir"])
                deleted_temp = cleanup_old_files(dirs["temp_dir"])
                deleted_cache = cleanup_old_files(dirs["cache_dir"])
            st.success(f"Deleted {deleted_uploads + deleted_temp + deleted_cache} files")
        st.write(f"**Storage Location:** {dirs['base_dir']}")
        st.write(f"**Temp Files:** {dirs['temp_dir']}")
        st.write(f"**Uploads:** {dirs['uploads_dir']}")
        st.write(f"**Cache:** {dirs['cache_dir']}")

    # Main UI
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📤 Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDFs, TXTs, or CSVs",
                type=["pdf", "txt", "csv"],
                accept_multiple_files=True,
                key="file_uploader",
            )
            st.subheader("🖼️ Upload Images")
            uploaded_images = st.file_uploader(
                "Upload Images (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="image_uploader",
            )
        with col2:
            st.subheader("🛠️ Quick Actions")
            if st.button("Clear Cache", key="clear_cache"):
                try:
                    shutil.rmtree(dirs["cache_dir"])
                    dirs["cache_dir"].mkdir(exist_ok=True)
                    st.success("Cache cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")

        if uploaded_files and uploaded_files != st.session_state.last_uploaded_files:
            available_space = check_disk_space(dirs)
            if available_space and available_space < 1:
                st.error("Low disk space! Please clean up files before uploading new documents.")
                return
            st.session_state.llm, st.session_state.qa_chain, st.session_state.hybrid_retriever, st.session_state.file_names = load_and_process_documents(
                uploaded_files, groq_api_key, dirs, process_image
            )
            st.session_state.last_uploaded_files = uploaded_files
            st.success(f"Processed {len(uploaded_files)} documents with embedded images.")

        if uploaded_images:
            with st.spinner("Processing standalone images..."):
                for image_file in uploaded_images:
                    extracted_text = process_image(image_file, dirs)
                    st.session_state.chat_history.append(
                        (
                            "You",
                            f"Extracted text from standalone image {image_file.name}: {extracted_text}",
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    )
                    st.session_state.chat_history.append(
                        ("Bot", "Image text extracted successfully.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                st.success(f"Processed {len(uploaded_images)} standalone images.")

        if st.session_state.file_names or uploaded_images:
            st.subheader("💬 Chat with Your Documents and Images")
            selected_files = st.multiselect(
                "Select files for your question (leave empty for all):",
                st.session_state.file_names,
                key="file_selection",
            )
            user_question = st.text_input(
                "Ask a question:",
                value=st.session_state.user_question,
                key="user_input",
                placeholder="Type your question here...",
                on_change=lambda: st.session_state.update(user_question=st.session_state.user_input),
            )

            file_context = ", ".join(selected_files) if selected_files else "all uploaded files and images"
            st.session_state.file_context = file_context

            if st.button("🚀 Submit", key="submit_question"):
                if user_question and user_question.strip():
                    with st.spinner("Generating answer..."):
                        try:
                            refined_q = refine_question(user_question, st.session_state.llm, file_context)
                            result = st.session_state.qa_chain({"question": refined_q})
                            answer = result.get("answer", "").strip() or "No answer generated."
                            if "no relevant content" in answer.lower() or "not in the documents" in answer.lower():
                                st.warning("No answer found in documents. Searching external sources...")
                                external_prompt = f"Search X and the web for: {refined_q}"
                                external_answer = st.session_state.llm.invoke(external_prompt).content.strip()
                                answer = f"{answer}\n\n**External Search Result:** {external_answer}"
                            st.session_state.chat_history.append(
                                (
                                    "You",
                                    f"Question about {file_context}: {user_question}",
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                )
                            )
                            st.session_state.chat_history.append(
                                ("Bot", answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            )
                            st.session_state.last_answer = answer
                            st.session_state.user_question = ""
                            st.success("Answer generated!")
                            st.markdown(f"<div class='chat-message bot-message'>{answer}</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                            st.session_state.last_answer = f"Error: {str(e)}"

            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.write("Adjust retrieval parameters or model settings here.")
                k_value = st.slider("Number of documents to retrieve (k)", 1, 10, 3)
                if st.session_state.hybrid_retriever and hasattr(st.session_state.hybrid_retriever, "retrievers"):
                    for retriever in st.session_state.hybrid_retriever.retrievers:
                        if hasattr(retriever, "search_kwargs"):
                            retriever.search_kwargs["k"] = k_value
                        if hasattr(retriever, "k"):
                            retriever.k = k_value

            st.subheader("📜 Chat History")
            chat_container = st.container(height=300)
            with chat_container:
                if not st.session_state.chat_history:
                    st.write("No chat history yet.")
                else:
                    for i in range(0, len(st.session_state.chat_history), 2):
                        if i in st.session_state.chat_deleted:
                            continue
                        user_msg = st.session_state.chat_history[i]
                        bot_msg = st.session_state.chat_history[i + 1] if i + 1 < len(st.session_state.chat_history) else None
                        col1, col2 = st.columns([9, 1])
                        with col1:
                            st.markdown(
                                f"<div class='chat-message user-message'>{user_msg[1]}<br><small>{user_msg[2]}</small></div>",
                                unsafe_allow_html=True,
                            )
                            if bot_msg:
                                st.markdown(
                                    f"<div class='chat-message bot-message'>{bot_msg[1]}<br><small>{bot_msg[2]}</small></div>",
                                    unsafe_allow_html=True,
                                )
                        with col2:
                            if st.button("🗑️", key=f"del_{i}", help="Delete this conversation"):
                                st.session_state.chat_deleted.add(i)
                                if i + 1 < len(st.session_state.chat_history):
                                    st.session_state.chat_deleted.add(i + 1)
                    if st.session_state.chat_deleted:
                        st.session_state.chat_history = [
                            msg for i, msg in enumerate(st.session_state.chat_history) if i not in st.session_state.chat_deleted
                        ]
                        st.session_state.chat_deleted = set()

            if st.button("🧹 Clear All Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.session_state.chat_deleted = set()
                st.session_state.last_answer = None
                st.success("Chat history cleared!")
                st.rerun()

            if st.button("📄 Export Chat as PDF"):
                pdf_path = export_chat_to_pdf(dirs["base_dir"])
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f.read(), file_name=pdf_path.name, mime="application/pdf")

            if st.session_state.last_answer:
                with st.expander("⭐ Feedback & Insights", expanded=False):
                    st.write("**Was this answer helpful?**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Yes", key="feedback_yes"):
                            store_feedback(
                                f"Question about {st.session_state.file_context}: {user_question}",
                                st.session_state.last_answer,
                                "positive",
                                dirs["base_dir"],
                            )
                            st.success("Thanks for your feedback!")
                    with col2:
                        if st.button("👎 No", key="feedback_no"):
                            store_feedback(
                                f"Question about {st.session_state.file_context}: {user_question}",
                                st.session_state.last_answer,
                                "negative",
                                dirs["base_dir"],
                            )
                            st.success("Thanks for your feedback!")
                    st.subheader("📊 Answer Insights")
                    wordcloud_text = st.session_state.last_answer
                    img_str = generate_wordcloud(wordcloud_text)
                    st.image(f"data:image/png;base64,{img_str}", caption="Word Cloud of Last Answer")

            chat_history_path = dirs["base_dir"] / "chat_history.json"
            with open(chat_history_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.chat_history, f, indent=2, ensure_ascii=False)

            st.download_button(
                "⬇️ Download Chat History",
                data=json.dumps(st.session_state.chat_history, ensure_ascii=False),
                file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            st.info(
                f"Processed {len(st.session_state.file_names)} files. Storage location: {dirs['base_dir']}. Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )