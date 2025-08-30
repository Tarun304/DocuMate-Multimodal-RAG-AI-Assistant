import streamlit as st
import requests, tempfile, os, base64
import time
from typing import List
import uuid
from src.utils.logger import logger

API_ROOT = "http://localhost:8000/api"
st.set_page_config(page_title="Multimodal-RAG Chat",
                   page_icon="💬",
                   layout="wide")


# ---------- state helpers ----------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader-0"

# Thread ID management
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    logger.info(f"Generated new thread_id: {st.session_state.thread_id}")



st.markdown(
    """
    <style>
        /* centred title + subtitle */
        .app-header           { text-align:center; margin-bottom:1.2rem; }
        .app-header h1        { font-weight:800; margin-bottom:0.25rem; }
        .app-header p         { font-size:1.05rem; color:#aaaaaa; margin-top:0; }


        /* centred "Select PDF" label  */
        section[data-testid="stFileUploader"] > label {
            display:block; text-align:center; font-weight:600;
        }


        /* About card */
        .about-card           { background:#1e1e29;               /* dark box   */
                                border:1px solid #444;           
                                border-radius:8px; 
                                padding:14px 18px; 
                                margin-top:14px; }


        .about-card h5        { margin:0 0 .6rem 0; font-size:1rem; }
        .about-card ul        { margin:0 0 .6rem 20px; }
        .about-card li        { margin:4px 0; }
        .about-card small     { color:#bbbbbb; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────── HEADER ──────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>💬 DocuMate: Multimodal RAG AI Assistant</h1>
        <p>Chat with your PDFs — texts, tables &amp; images</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ───────────────── SIDEBAR ─────────────────
with st.sidebar:
    st.header("📄 Upload & Manage")


    # Upload + Process
    pdf = st.file_uploader("Select PDF",
                       type=["pdf"],
                       key=st.session_state.uploader_key)
    if pdf:
        st.success(f"Selected: {pdf.name}")
        if st.button("⚡ Process Document"):
            with st.spinner("Processing and Analysing the PDF…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf.getbuffer())
                    path = tmp.name
                try:
                    # Hit the '/build-index' endpoint
                    ok = requests.post(f"{API_ROOT}/build-index",
                                       json={"pdf_path": path}).ok
                    st.success("Analysis Complete!" if ok else "Processing failed.")
                finally:
                    os.remove(path)


    # Reset
    if st.button("♻️ Reset", help="Clear session and start over"):
        with st.spinner("Resetting…"):
            try:
                # Send current thread_id to reset endpoint
                requests.post(f"{API_ROOT}/reset", 
                            json={"thread_id": st.session_state.thread_id}, 
                            timeout=5)
            except requests.exceptions.RequestException:
                st.warning("Backend not reachable")
            
            # Generate new thread_id for fresh session
            st.session_state.thread_id = str(uuid.uuid4())
            logger.info(f"Reset: new thread_id: {st.session_state.thread_id}")
        
        st.session_state.clear()
        # Restore thread_id after clear
        st.session_state.thread_id = str(uuid.uuid4())
        # create a brand-new key so the old file disappears
        st.session_state.uploader_key = f"uploader-{os.urandom(4).hex()}"
        st.rerun()


    # About  (left-aligned bullet list in dark card)
    st.markdown(
        """
<div class="about-card">
  <h5>ℹ️ About</h5>
  <p>DocuMate lets you:</p>
  <ul>
    <li>📄 Ask questions over PDF documents</li>
    <li>📊 Understand tables and charts</li>
    <li>🖼️ Analyse images</li>
    <li>🧠 Remember conversation context</li>
  </ul>
  <small>Powered by <b>LangChain</b>, <b>LangGraph</b> &amp; <b>Streamlit</b> 🚀</small>
</div>
""",
        unsafe_allow_html=True,
    )
    
    # API health-check button 
    st.markdown("----")
    if st.button("🔄 Check API health", help="Ping the FastAPI backend"):
        try:
            resp = requests.get(f"{API_ROOT}/health", timeout=5).json()
            st.success(f"✅ {resp.get('message', 'Backend is healthy')}")
        except Exception as e:
            st.error(f"❌ Backend not reachable: {e}")


# ───────────── HELPER FUNCTIONS ─────────────
def typewriter_effect(text: str, placeholder, delay: float = 0.0001):
    """Display text with typewriter effect at maximum speed"""
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(delay)  # Maximum speed
    


def render_context(ctx):
    """Render context tabs with original formatting"""
    with st.expander("🔍 Retrieved context", expanded=False):
        
        txt_tab, tbl_tab, img_tab = st.tabs(["📝 Texts", "📑Tables", "🖼️ Images"])

        info_msg= "USED MEMORY"  # Use the same info_msg

        if ( ctx.get("texts") == [info_msg] and ctx.get("tables") == [info_msg] and ctx.get("images") == [info_msg]):
            with txt_tab:
                st.info("💡 Used conversational memory to generate this response.")
            with tbl_tab:
                st.info("💡 Used conversational memory to generate this response.")
            with img_tab:
                st.info("💡 Used conversational memory to generate this response.") 

        else: 
            #  texts
            with txt_tab:
                texts = ctx.get("texts", [])
                if texts:
                    for t in texts:
                        st.markdown(f"> {t}")
                else:
                    st.info("No text snippets returned for this answer.")

            # tables
            with tbl_tab:
                tables = ctx.get("tables", [])
                if tables:
                    for html in tables:
                        st.markdown(html, unsafe_allow_html=True)
                else:
                    st.info("No tables relevant to this response.")

            # images
            with img_tab:
                images = ctx.get("images", [])
                if images:
                    for b64 in images:
                        st.image(base64.b64decode(b64))
                else:
                    st.info("No images associated with this answer.")


# ───────────── CHAT INTERFACE ─────────────
if "history" not in st.session_state:
    st.session_state.history: List[dict] = []

query = st.chat_input("Ask anything about the document…")

if query:
    
    # Step 1 - Show user message immediately
    st.session_state.history.append({"role": "user", "content": query})
    
    # Render all existing history
    for turn in st.session_state.history:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(turn["content"])
                ctx = turn.get("context", {})
                if ctx:
                    render_context(ctx)
    
    # Step 2 - Show thinking and make API call
    thinking_placeholder = st.empty()
    with thinking_placeholder:
        with st.chat_message("assistant"):
            with st.spinner("🧠✨ Thinking…"):
                try:
                    resp = requests.post(
                        f"{API_ROOT}/ask-question",
                        json={
                            "question": query,
                            "thread_id": st.session_state.thread_id  # Send thread_id
                        },
                        timeout=120,
                    ).json()
                    answer = resp.get("answer", "No answer returned.")
                    context = resp.get("context", {})
                    # Update thread_id if backend provides it
                    if "thread_id" in resp:
                        st.session_state.thread_id = resp["thread_id"]
                except Exception as e:
                    answer, context = f"❌ Error: {e}", {}
    
    # Step 3 - Replace thinking with typewriter answer
    thinking_placeholder.empty()  # Remove the thinking indicator
    
    # Show the assistant response with typewriter effect
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        typewriter_effect(answer, answer_placeholder, delay=0.001) 
        
        # Show context after typewriter completes
        if context:
            render_context(context)
    
    # Add the assistant response to history after typewriter completes
    st.session_state.history.append(
        {"role": "assistant", "content": answer, "context": context}
    )
    
    # AUTO-SCROLL: Force refresh to scroll to bottom
    st.rerun()

else:
    # ──────────── RENDER EXISTING HISTORY (when no new query) ────────────
    for turn in st.session_state.history:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(turn["content"])
                ctx = turn.get("context", {})
                if ctx:
                    render_context(ctx)
