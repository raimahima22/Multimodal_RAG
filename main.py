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

# Suppress common warnings
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
    page_title="Healthcare Benefits Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    .block-container {padding-top: 2rem;}
    .title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1e40af;
        text-align: center;
        margin-bottom: 0.4rem;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    .stChatMessage {border-radius: 12px; padding: 14px 18px;}
    .loading-container {
        text-align: center;
        padding: 4rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== UTILITIES ======================
def save_to_history(query: str, source: str, answer: str):
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
        "source": source,
        "answer": answer,
    })

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[-100:], f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"History save failed: {e}")


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_page_cache()


# ====================== INITIALIZATION ======================
@st.cache_resource(show_spinner=False)
def initialize_system():
    """One-time heavy initialization with clear feedback"""
    status = st.empty()
    status.info("🚀 Initializing AI Models... This may take 30-90 seconds on first run.")
    
    try:
        # Warmup Agent
        status.info("🔄 Warming up LangGraph Agent...")
        run_agent("warmup query")
        
        # Load Voice Interface (contains Whisper + Piper)
        status.info("🎙 Loading Voice Models (STT + TTS)...")
        voice_interface = get_voice_interface(run_agent)
        
        status.success("✅ System Ready!")
        time.sleep(1)
        status.empty()
        
        return voice_interface
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()


# ====================== MAIN APP ======================
def main():
    # Force initialization before showing UI
    if "voice_interface" not in st.session_state:
        with st.spinner(""):
            st.session_state.voice_interface = initialize_system()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.title("Benefits Assistant")
        st.markdown("**SBC + SPD Plan Assistant**")
        
        st.divider()
        st.markdown("### Knowledge Base")
        st.info("""
        **SBC** — Quick benefits, deductibles, copays  
        **SPD** — Rules, eligibility, exclusions, procedures
        """)
        
        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.caption("LangGraph • ColQwen2.5 • Qdrant")

    # ====================== HEADER ======================
    st.markdown('<h1 class="title">Healthcare Benefits Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Instant answers from your plan documents</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Text Chat", "Voice Assistant"])

    # ====================== TEXT CHAT ======================
    with tab1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your benefits plan..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    try:
                        response = run_agent(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        save_to_history(prompt, "Agent (SBC/SPD)", response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            aggressive_cleanup()

    # ====================== VOICE TAB ======================
    with tab2:
        st.markdown("### Voice Assistant")
        st.markdown("Record your question for spoken + text response.")

        audio_file = st.audio_input("Record your question")

        if st.button("Send Voice Query", type="primary", use_container_width=True):
            if audio_file:
                with st.spinner("Transcribing • Thinking • Speaking..."):
                    try:
                        audio_path, result_text = st.session_state.voice_interface.voice_pipeline(audio_file)
                        st.success("**You said:** " + (result_text.split("\n\n")[0] if "\n\n" in result_text else result_text))
                        st.audio(audio_path, format="audio/wav", autoplay=True)

                        st.session_state.messages.append({"role": "user", "content": "🎤 Voice Input"})
                        st.session_state.messages.append({"role": "assistant", "content": result_text})
                        save_to_history("Voice Input", "Agent (SBC/SPD)", result_text)
                    except Exception as e:
                        st.error(f"Voice error: {e}")
            else:
                st.warning("Please record audio first.")

    st.caption("Confidential • Internal Healthcare Benefits Tool")


if __name__ == "__main__":
    main()