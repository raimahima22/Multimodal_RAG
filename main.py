import sys
import gc
import os
import json
import warnings
from datetime import datetime
import torch
import time
import logging

import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore", message=".*__path__.*image_processing_sam.*")
warnings.filterwarnings("ignore", category=UserWarning)

from src.utils import clear_page_cache
from src.agent import run_agent
from src.voice import get_voice_interface

# ====================== CONFIG ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HISTORY_FILE = "chat_history.json"

st.set_page_config(
    page_title="Benefits Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    .block-container {padding-top: 2rem;}
    .big-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {text-align: center; color: #475569; font-size: 1.1rem;}
    .stChatMessage {border-radius: 16px; padding: 16px;}
</style>
""", unsafe_allow_html=True)

# ====================== CACHED HEAVY RESOURCES ======================
@st.cache_resource(show_spinner="Loading AI Models (this may take 20-60s on first run)...")
def load_voice_interface():
    """Cache voice + agent dependencies"""
    logger.info("Loading voice interface...")
    return get_voice_interface(run_agent)

@st.cache_resource(show_spinner="Warming up RAG Agent...")
def warmup_system():
    """One-time warmup"""
    try:
        run_agent("warmup query")
        logger.info("✅ System warmed up")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

# ====================== UTILITIES ======================
def save_to_history(query: str, answer: str):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            pass

    history.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer[:600] + "..." if len(answer) > 600 else answer
    })

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[-50:], f, indent=2)
    except Exception as e:
        logger.error(f"History save error: {e}")


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_page_cache()

# ====================== MAIN APP ======================
def main():
    # Initialize cached resources (this runs once thanks to cache_resource)
    voice_interface = load_voice_interface()
    warmup_system()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/health-care.png", width=80)
        st.title("Benefits Assistant")
        st.markdown("**SBC + SPD AI Assistant**")

        st.divider()
        st.markdown("### Knowledge Base")
        st.info("**SBC**: Quick benefits & costs\n**SPD**: Rules & procedures")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption("LangGraph • ColQwen2.5 • Qdrant")

    # ====================== HEADER ======================
    st.markdown('<h1 class="big-title">Healthcare Benefits Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Accurate answers from your plan documents</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💬 Text Chat", "🎤 Voice Assistant"])

    # ====================== TEXT CHAT ======================
    with tab1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🩺"):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your benefits..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("Searching documents..."):
                    try:
                        response = run_agent(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        save_to_history(prompt, response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            aggressive_cleanup()

    # ====================== VOICE TAB ======================
    with tab2:
        st.markdown("### 🎙️ Voice Assistant")
        audio_file = st.audio_input("Record your question")

        if st.button("🔊 Process Voice Query", type="primary"):
            if audio_file:
                with st.spinner("Transcribing and generating response..."):
                    try:
                        audio_path, result_text = voice_interface.voice_pipeline(audio_file)
                        st.success("**You said:** " + result_text.split("\n\n")[0])
                        st.audio(audio_path, format="audio/wav", autoplay=True)

                        st.session_state.messages.append({"role": "user", "content": "🎤 Voice Query"})
                        st.session_state.messages.append({"role": "assistant", "content": result_text})
                        save_to_history("🎤 Voice Query", result_text)
                    except Exception as e:
                        st.error(f"Voice error: {e}")
            else:
                st.warning("Please record audio first.")

    st.caption("Confidential • Internal Use")

if __name__ == "__main__":
    main()