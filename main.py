# import sys
# import gc
# import os
# import json
# from datetime import datetime
# import torch
# import time
# import logging

# import streamlit as st

# from src.utils import clear_page_cache
# from src.agent import run_agent
# from src.voice import get_voice_interface

# HISTORY_FILE = "chat_history.json"

# # ====================== LOGGING ======================
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def save_to_history(query: str, source_input: str, answer: str):
#     """Save conversation history"""
#     history_data = []
#     if os.path.exists(HISTORY_FILE):
#         try:
#             with open(HISTORY_FILE, "r", encoding="utf-8") as f:
#                 history_data = json.load(f)
#         except Exception:
#             history_data = []
   
#     history_data.append({
#         "timestamp": datetime.now().isoformat(),
#         "query": query,
#         "source": source_input,
#         "answer": answer,
#     })
   
#     with open(HISTORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(history_data, f, indent=2, ensure_ascii=False)


# def aggressive_cleanup():
#     """Clean memory after operations"""
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     clear_page_cache()


# st.set_page_config(
#     page_title="Healthcare Benefits Assistant",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ====================== CUSTOM CSS ======================
# st.markdown("""
# <style>
#     .main {background-color: #f8fafc;}
#     .block-container {padding-top: 2.5rem;}
    
#     .title {
#         font-size: 2.6rem;
#         font-weight: 700;
#         color: #1e40af;
#         text-align: center;
#         margin-bottom: 0.4rem;
#     }
#     .subtitle {
#         text-align: center;
#         color: #475569;
#         font-size: 1.15rem;
#         margin-bottom: 2rem;
#     }
    
#     .stChatMessage {
#         border-radius: 12px;
#         padding: 14px 18px;
#     }
# </style>
# """, unsafe_allow_html=True)

# def main(force_reindex=False):
#     logger.info("Starting Healthcare Benefits Assistant")
    
#     # Warmup
#     print("Warming up Agent and retrieval models...")
#     try:
#         _ = run_agent("warmup query for initialization")
#         logger.info("Agent warmup completed")
#     except Exception as e:
#         logger.warning(f"Warmup warning: {e}")
    
#     voice_interface = get_voice_interface(run_agent)

#     # Session State
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # ====================== SIDEBAR ======================
#     with st.sidebar:
#         st.title("Benefits Assistant")
#         st.markdown("**Summary of Benefits & Coverage**")
        
#         st.divider()
        
#         st.markdown("### Knowledge Base")
#         st.markdown("""
#         **SBC** — Quick benefit details, deductibles, copays, coverage limits  
#         **SPD** — Detailed plan rules, eligibility, exclusions, procedures
#         """)
        
#         st.divider()
        
#         if st.button("Clear Chat History", use_container_width=True):
#             st.session_state.messages = []
#             st.rerun()
        
#         st.caption("Powered by LangGraph Agent • ColQwen2.5 • Qdrant")

#     # ====================== MAIN HEADER ======================
#     st.markdown('<h1 class="title">Healthcare Benefits Assistant</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="subtitle">Ask questions about your plan documents</p>', unsafe_allow_html=True)

#     # ====================== TABS ======================
#     tab1, tab2 = st.tabs(["Text Chat", "Voice Assistant"])

#     # ====================== TEXT CHAT TAB ======================
#     with tab1:
#         # Display chat messages
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         # User input
#         if prompt := st.chat_input("Ask a question about your benefits plan..."):
#             # Add user message
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             # Assistant response
#             with st.chat_message("assistant"):
#                 with st.spinner("Searching documents and generating response..."):
#                     try:
#                         bot_response = run_agent(prompt)
#                         st.markdown(bot_response)
#                         st.session_state.messages.append({"role": "assistant", "content": bot_response})
#                         save_to_history(prompt, "Agent (SBC/SPD)", bot_response)
#                     except Exception as e:
#                         error_msg = f"Error while generating response: {str(e)}"
#                         st.error(error_msg)
#                         st.session_state.messages.append({"role": "assistant", "content": error_msg})

#             aggressive_cleanup()

#     # ====================== VOICE ASSISTANT TAB ======================
#     with tab2:
#         st.markdown("### Voice-Enabled Assistant")
#         st.markdown("Record your question to receive a spoken response.")

#         audio_file = st.audio_input("Record Your Question", label_visibility="visible")

#         if st.button("Send Voice Query", type="primary", use_container_width=True):
#             if audio_file is not None:
#                 with st.spinner("Processing voice input..."):
#                     try:
#                         audio_path, result_text = voice_interface.voice_pipeline(audio_file)
                        
#                         # Display transcription and response
#                         st.markdown("**Transcription:**")
#                         st.info(result_text.split("\n\n")[0] if "\n\n" in result_text else result_text)
                        
#                         st.markdown("**Agent Response:**")
#                         st.audio(audio_path, format="audio/wav", autoplay=True)
                        
#                         # Add to chat history
#                         st.session_state.messages.append({"role": "user", "content": "Voice Input"})
#                         st.session_state.messages.append({"role": "assistant", "content": result_text})
#                         save_to_history("Voice Input", "Agent (SBC/SPD)", result_text)
                        
#                     except Exception as e:
#                         st.error(f"Error in voice pipeline: {str(e)}")
#             else:
#                 st.warning("Please record an audio message.")

#     st.caption("Confidential • Healthcare Benefits Assistant")


# if __name__ == "__main__":
#     force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
#     main(force_reindex)

"""
main.py — Healthcare Benefits Assistant (Streamlit UI)

Run with:
    streamlit run main.py
    streamlit run main.py -- --reindex      # force re-index documents
"""

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

import streamlit as st
import torch

from src.agent import run_agent
from src.utils import clear_page_cache
from src.voice import get_voice_interface

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HISTORY_FILE = "chat_history.json"

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Benefits Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #f0f4f8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155 !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}

/* ── Main container ── */
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1100px !important;
}

/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 24px rgba(30,64,175,0.18);
}
.app-header h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #bfdbfe;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}
.header-badges {
    display: flex;
    gap: 8px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.15);
    color: #e0f2fe;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.02em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px 12px 0 0;
    padding: 4px 6px 0;
    border-bottom: 2px solid #e2e8f0;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 0.55rem 1.2rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    transition: color 0.15s;
}
.stTabs [aria-selected="true"] {
    color: #1e40af !important;
    background: #eff6ff !important;
    border-bottom: 2px solid #1e40af !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #ffffff;
    border-radius: 0 0 12px 12px;
    padding: 1.5rem 1.5rem 1rem;
    border: 1px solid #e2e8f0;
    border-top: none;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #eff6ff !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #cbd5e1 !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ── Latency card ── */
.latency-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
}
.latency-card h4 {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0 0 0.75rem 0;
}
.latency-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}
.latency-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    text-align: center;
}
.latency-item .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e40af;
}
.latency-item .lbl {
    font-size: 0.68rem;
    color: #94a3b8;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Voice section ── */
.voice-hint {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid #0284c7;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    font-size: 0.84rem;
    color: #0369a1;
    margin-bottom: 1rem;
}

/* ── Metric chips (sidebar) ── */
.stat-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 6px;
    font-size: 0.8rem;
}
.stat-chip .sc-label { color: #64748b; font-size: 0.7rem; }
.stat-chip .sc-value { color: #f1f5f9; font-weight: 600; }

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: #2563eb !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Utility functions ─────────────────────────────────────────────────────────

def save_to_history(query: str, source: str, answer: str) -> None:
    history: list = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "source": source,
            "answer": answer,
        }
    )
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_page_cache()


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def latency_card_html(stt: float, agent: float, tts: float, total: float) -> str:
    return f"""
<div class="latency-card">
  <h4>⏱ Pipeline Latency</h4>
  <div class="latency-grid">
    <div class="latency-item">
      <div class="val">{stt:.2f}s</div>
      <div class="lbl">STT</div>
    </div>
    <div class="latency-item">
      <div class="val">{agent:.2f}s</div>
      <div class="lbl">Agent</div>
    </div>
    <div class="latency-item">
      <div class="val">{tts:.2f}s</div>
      <div class="lbl">TTS</div>
    </div>
    <div class="latency-item">
      <div class="val">{total:.2f}s</div>
      <div class="lbl">Total</div>
    </div>
  </div>
</div>
"""

# ── Session-state initialisation ──────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "messages": [],
        "total_queries": 0,
        "voice_interface": None,
        "agent_ready": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🩺 Benefits Assistant")
        st.markdown("---")

        # Knowledge base description
        st.markdown("### 📚 Knowledge Base")
        st.markdown(
            """
<div class="stat-chip">
  <div class="sc-label">SBC — Summary of Benefits</div>
  <div class="sc-value">Deductibles · Copays · Coverage</div>
</div>
<div class="stat-chip">
  <div class="sc-label">SPD — Summary Plan Description</div>
  <div class="sc-value">Rules · Eligibility · Exclusions</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Session stats
        st.markdown("### 📊 Session")
        st.markdown(
            f"""
<div class="stat-chip">
  <div class="sc-label">Queries this session</div>
  <div class="sc-value">{st.session_state.total_queries}</div>
</div>
<div class="stat-chip">
  <div class="sc-label">Messages in chat</div>
  <div class="sc-value">{len(st.session_state.messages)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Actions
        st.markdown("### ⚙️ Actions")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        history = load_history()
        if history:
            if st.button("📥 Export History", use_container_width=True):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(history, indent=2, ensure_ascii=False),
                    file_name=f"benefits_history_{datetime.now():%Y%m%d_%H%M}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.markdown("---")

        # Sample questions
        st.markdown("### 💡 Sample Questions")
        samples = [
            "What is my annual deductible?",
            "Is physical therapy covered?",
            "How do I submit a claim?",
            "What is the out-of-pocket maximum?",
            "Are prescriptions covered under my plan?",
        ]
        for q in samples:
            if st.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
                st.session_state["_prefill"] = q
                st.rerun()

        st.markdown("---")
        st.caption("Powered by LangGraph · ColQwen2.5 · Qdrant")


# ── Header ────────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        """
<div class="app-header">
  <h1>🩺 Healthcare Benefits Assistant</h1>
  <p>Ask questions about your plan documents in text or voice — get instant, grounded answers.</p>
  <div class="header-badges">
    <span class="badge">LangGraph Agent</span>
    <span class="badge">ColQwen2.5 RAG</span>
    <span class="badge">SBC + SPD</span>
    <span class="badge">Voice-Enabled</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Text chat tab ─────────────────────────────────────────────────────────────

def render_text_tab() -> None:
    # Show existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pre-filled query from sidebar sample buttons
    prefill = st.session_state.pop("_prefill", None)

    prompt = st.chat_input(
        "Ask about deductibles, coverage, claims, eligibility…",
        key="chat_input",
    )
    if prefill and not prompt:
        prompt = prefill

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_queries += 1

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching plan documents…"):
                t0 = time.perf_counter()
                try:
                    response = run_agent(prompt)
                    elapsed = time.perf_counter() - t0
                except Exception as exc:
                    response = f"An error occurred: {exc}"
                    elapsed = time.perf_counter() - t0

            st.markdown(response)

            # Subtle timing note
            st.caption(f"⚡ Response generated in {elapsed:.2f}s")

        st.session_state.messages.append({"role": "assistant", "content": response})
        save_to_history(prompt, "Agent (SBC/SPD)", response)
        cleanup()
        st.rerun()


# ── Voice tab ─────────────────────────────────────────────────────────────────

def render_voice_tab() -> None:
    st.markdown(
        """
<div class="voice-hint">
  🎙 Record your question below, then click <strong>Send Voice Query</strong>.
  You'll receive a spoken response along with the transcription and latency breakdown.
</div>
""",
        unsafe_allow_html=True,
    )

    col_rec, col_btn = st.columns([3, 1], gap="medium")

    with col_rec:
        audio_file = st.audio_input(
            "Record your question",
            label_visibility="collapsed",
            key="voice_input",
        )

    with col_btn:
        st.markdown("<div style='height:2.1rem'></div>", unsafe_allow_html=True)
        send = st.button("Send Voice Query", type="primary", use_container_width=True)

    if send:
        if audio_file is None:
            st.warning("Please record an audio message first.")
            return

        # Lazy-init voice interface
        if st.session_state.voice_interface is None:
            with st.spinner("Loading voice models (first run only)…"):
                st.session_state.voice_interface = get_voice_interface(run_agent)

        vi = st.session_state.voice_interface

        with st.spinner("Processing… STT → Agent → TTS"):
            result = vi.voice_pipeline(audio_file)

        if result["error"]:
            st.error(f"Voice pipeline error: {result['error']}")
            return

        # ── Results layout ──
        st.markdown("---")

        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("####  Transcription")
            st.info(result["query"] or "_(no speech detected)_")

            st.markdown("####  Agent Answer")
            st.markdown(result["answer"])

        with col_right:
            st.markdown("####  Audio Response")
            if result["audio_path"] and os.path.exists(result["audio_path"]):
                with open(result["audio_path"], "rb") as af:
                    st.audio(af.read(), format="audio/wav", autoplay=True)
            else:
                st.warning("Audio file not found.")

            st.markdown(
                latency_card_html(
                    result["stt_latency"],
                    result["agent_latency"],
                    result["tts_latency"],
                    result["total_latency"],
                ),
                unsafe_allow_html=True,
            )

        # Add to chat history
        st.session_state.messages.append(
            {"role": "user", "content": f"🎙 _{result['query']}_"}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.session_state.total_queries += 1
        save_to_history(result["query"], "Voice (SBC/SPD)", result["answer"])
        cleanup()


# ── History tab ───────────────────────────────────────────────────────────────

def render_history_tab() -> None:
    history = load_history()

    if not history:
        st.info("No conversation history yet. Ask a question to get started.")
        return

    st.markdown(f"**{len(history)} conversations recorded.**")

    # Reverse so newest first
    for entry in reversed(history[-50:]):
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        src = entry.get("source", "")
        with st.expander(f"{ts}  ·  {entry['query'][:80]}", expanded=False):
            st.markdown(f"**Source:** `{src}`")
            st.markdown(f"**Question:** {entry['query']}")
            st.markdown("**Answer:**")
            st.markdown(entry["answer"])

    if st.button("🗑 Clear All History", type="primary"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("History cleared.")
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    render_sidebar()
    render_header()

    # Warm up agent once per process (not per rerun)
    if not st.session_state.agent_ready:
        with st.spinner("Initialising agent and retrieval models…"):
            try:
                _ = run_agent("warmup")
                st.session_state.agent_ready = True
                logger.info("Agent warmup complete.")
            except Exception as exc:
                logger.warning(f"Warmup warning: {exc}")
                st.session_state.agent_ready = True  # don't retry forever

    # Tabs
    tab_text, tab_voice, tab_history = st.tabs(
        ["  Text Chat", "  Voice Assistant", "  History"]
    )

    with tab_text:
        render_text_tab()

    with tab_voice:
        render_voice_tab()

    with tab_history:
        render_history_tab()

    st.markdown(
        "<p style='text-align:center;color:#94a3b8;font-size:0.75rem;margin-top:2rem'>"
        "Confidential · Healthcare Benefits Assistant · Answers based solely on your plan documents"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()