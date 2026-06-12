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

# ====================== CONFIG & LOGGING ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HISTORY_FILE = "chat_history.json"

st.set_page_config(
    page_title="Benefits Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Healthcare Benefits Assistant v1.0"
    }
)

# ====================== PROFESSIONAL CSS ======================
st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    .block-container {padding-top: 2rem;}
    
    .big-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .stChatMessage {
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .stButton>button {
        border-radius: 8px;
    }
    
    .sidebar .stMarkdown h1, .sidebar .stMarkdown h2, .sidebar .stMarkdown h3 {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== CACHED RESOURCES ======================
@st.cache_resource(show_spinner="Initializing AI Models...")
def get_voice_interface_cached():
    return get_voice_interface(run_agent)

@st.cache_resource(show_spinner="Warming up Agent...")
def warmup_agent():
    try:
        run_agent("warmup query")
        logger.info("✅ Agent warmed up")
    except Exception as e:
        logger.warning(f"Warmup warning: {e}")

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
        "source": "LangGraph Agent",
        "answer": answer[:500] + "..." if len(answer) > 500 else answer
    })

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[-50:], f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"History save failed: {e}")


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_page_cache()

# ====================== MAIN APPLICATION ======================
def main():
    warmup_agent()
    voice_interface = get_voice_interface_cached()

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/health-care.png", width=80)
        st.title("Benefits Assistant")
        st.markdown("**Enterprise Healthcare Plan AI**")
        
        st.divider()
        
        st.markdown("### 📘 Knowledge Sources")
        st.info("""
        **SBC** — Summary of Benefits & Coverage  
        **SPD** — Summary Plan Description
        """, icon="📋")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("📥 Export Chat", use_container_width=True):
                if st.session_state.messages:
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(st.session_state.messages, indent=2),
                        file_name=f"benefits_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
        
        st.divider()
        st.caption("Powered by\n• LangGraph Agent\n• ColQwen2.5 Multimodal RAG\n• Qdrant Vector DB")

    # ====================== HEADER ======================
    st.markdown('<h1 class="big-title">Healthcare Benefits Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Instant, accurate answers from your SBC & SPD documents</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💬 Text Chat", "🎤 Voice Assistant"])

    # ====================== TEXT CHAT TAB ======================
    with tab1:
        for message in st.session_state.messages:
            avatar = "🧑‍💼" if message["role"] == "user" else "🩺"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your benefits plan..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("Searching documents and generating response..."):
                    start = time.time()
                    try:
                        response = run_agent(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        save_to_history(prompt, response)
                    except Exception as e:
                        error = f"**Error:** {str(e)}"
                        st.error(error)
                        st.session_state.messages.append({"role": "assistant", "content": error})

            # Keep only last 30 messages
            if len(st.session_state.messages) > 30:
                st.session_state.messages = st.session_state.messages[-30:]

            aggressive_cleanup()

    # ====================== VOICE TAB ======================
    with tab2:
        st.markdown("### Voice Interaction")
        st.markdown("Record your question for a natural spoken response.")

        col1, col2 = st.columns([3, 1])
        with col1:
            audio_file = st.audio_input("Record your question", label_visibility="collapsed")

        with col2:
            if st.button("🔊 Send & Speak", type="primary", use_container_width=True):
                if audio_file is not None:
                    with st.spinner("Transcribing • Thinking • Speaking..."):
                        try:
                            audio_path, result_text = voice_interface.voice_pipeline(audio_file)
                            
                            # Display transcription
                            transcribed = result_text.split("\n\n")[0] if "\n\n" in result_text else result_text
                            st.success(f"**You said:** {transcribed}")
                            
                            # Play response
                            st.audio(audio_path, format="audio/wav", autoplay=True)
                            
                            # Save to chat
                            st.session_state.messages.append({"role": "user", "content": "🎤 Voice Input"})
                            st.session_state.messages.append({"role": "assistant", "content": result_text})
                            save_to_history("🎤 Voice Input", result_text)
                            
                        except Exception as e:
                            st.error(f"Voice processing failed: {str(e)}")
                else:
                    st.warning("Please record an audio query first.")

        st.info("Voice uses Faster-Whisper (STT) + Piper TTS", icon="🎙️")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
        "Confidential • For Authorized Personnel Only • v1.0</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()