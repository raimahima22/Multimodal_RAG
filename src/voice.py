# import time
# import torch
# import numpy as np
# import sounddevice as sd
# import gradio as gr
# import os

# from faster_whisper import WhisperModel
# from piper import PiperVoice  # NEW (instead of TTS)


# # MODEL_PATH = os.path.join(
# #     "/content/drive/MyDrive/piper_models",
# #     "en_US-amy-medium.onnx"
# # )
# MODEL_PATH = os.path.join(
#     "models",
#     "en_US-amy-medium.onnx"
# )

# class VoiceInterface:
#     def __init__(self, agent_func):
#         self.agent_func = agent_func

#         # self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device = "cpu"

#         # --------------------
#         # STT (Whisper)
#         # --------------------
#         print("Loading faster-whisper...")
#         self.stt_model = WhisperModel(
#             "base.en",
#             device=self.device,
#             # compute_type="float16" if self.device == "cuda" else "int8"
#             compute_type="int8"
#         )

#         # --------------------
#         # TTS (Piper)
#         # --------------------
#         print("Loading Piper TTS...")

#         # Path to downloaded model (YOU MUST SET THIS)
#         self.voice = PiperVoice.load(
#             model_path=MODEL_PATH
#         )

#         self.sample_rate = 22050

#     # --------------------
#     # STT
#     # --------------------
#     def transcribe_audio(self, audio_path: str) -> str:
#         start = time.time()

#         segments, _ = self.stt_model.transcribe(
#             audio_path,
#             beam_size=5,
#             language="en",
#             vad_filter=True
#         )

#         text = " ".join(seg.text for seg in segments).strip()

#         print(f"STT Latency: {time.time() - start:.2f}s")

#         return text

#     # --------------------
#     # TTS (Piper)
#     # --------------------
#     def speak(self, text: str) -> str:
#         start = time.time()

#         output_path = "agent_response.wav"

#         with open(output_path, "wb") as f:
#             self.voice.synthesize(text, f)

#         print(f"TTS Latency: {time.time() - start:.2f}s")

#         return output_path

#     # --------------------
#     # PIPELINE
#     # --------------------
#     # def voice_pipeline(self, audio):
#     #     """Full voice → agent → voice response (with latency tracking)"""
    
#     #     if audio is None:
#     #         return None, "No audio received. Please record again."

#     #     try:
#     #         total_start = time.time()

#     #         # Voice pipeline (STT + LLM + TTS)
       
#     #         pipeline_start = time.time()
#     #         audio_path, result_text = voice_interface.voice_pipeline(audio)
#     #         pipeline_latency = time.time() - pipeline_start

#     #         total_latency = time.time() - total_start

#     #         # Append latency info to output text
#     #         result_text += (
#     #             f"\n\n **Pipeline Latency:** {pipeline_latency:.2f}s"
#     #             f"\n **Total Latency:** {total_latency:.2f}s"
#     #         )

#     #         return audio_path, result_text

#     #     except Exception as e:
#     #         return None, f"Error in voice pipeline: {str(e)}"

#     def voice_pipeline(self, audio):
#         """STT → Agent → TTS pipeline"""

#         if audio is None:
#             return None, "No audio received. Please record again."

#         query = self.transcribe_audio(audio)

#         if not query:
#             return None, "Could not understand audio."

#         print(f"User said: {query}")

#         start = time.time()
#         answer = self.agent_func(query)
#         print(f"Agent Latency: {time.time() - start:.2f}s")

#         audio_path = self.speak(answer)

#         return audio_path, answer

#         gc.collect()
#         torch.cuda.empty_cache()


# voice_interface = None


# def get_voice_interface(agent_func):
#     global voice_interface
#     if voice_interface is None:
#         voice_interface = VoiceInterface(agent_func)
#     return voice_interface

"""
voice.py — STT + TTS interface for the Healthcare Benefits Assistant.

Pipeline:
    Audio (file / Streamlit UploadedFile)
        ↓
    faster-whisper  → transcribed text
        ↓
    LangGraph Agent → answer text
        ↓
    Piper TTS       → WAV audio file
"""

import gc
import os
import tempfile
import time

import torch
from faster_whisper import WhisperModel
from piper import PiperVoice

# MODEL_PATH = os.path.join("models", "en_US-amy-medium.onnx")
MODEL_PATH = os.path.join(
    "/content/drive/MyDrive/piper_models",
    "en_US-amy-medium.onnx"
)


class VoiceInterface:
    """
    End-to-end voice pipeline: STT → Agent → TTS.

    Latency for each stage is tracked and returned to the caller
    so the UI can display benchmarks.
    """

    def __init__(self, agent_func):
        self.agent_func = agent_func
        self.device = "cpu"  # keep deterministic; flip to "cuda" if available

        # ── STT ──────────────────────────────────────────────────────────
        print("Loading faster-whisper (base.en) …")
        self.stt_model = WhisperModel(
            "base.en",
            device=self.device,
            compute_type="int8",
        )

        # ── TTS ──────────────────────────────────────────────────────────
        print("Loading Piper TTS …")
        self.voice = PiperVoice.load(model_path=MODEL_PATH)
        self.sample_rate = 22050

        print("VoiceInterface ready.")

    # ── STT ──────────────────────────────────────────────────────────────

    def transcribe_audio(self, audio_source) -> tuple[str, float]:
        """
        Transcribe audio to text.

        Args:
            audio_source:
                Either a file-path string or a Streamlit ``UploadedFile``
                (anything with a ``.read()`` method).

        Returns:
            (transcribed_text, stt_latency_seconds)
        """
        start = time.perf_counter()

        # Streamlit audio_input returns an UploadedFile; write it to a
        # temporary WAV so faster-whisper can open it by path.
        if hasattr(audio_source, "read"):
            suffix = getattr(audio_source, "name", "audio.wav")
            suffix = os.path.splitext(suffix)[-1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_source.read())
                audio_path = tmp.name
            owns_tmp = True
        else:
            audio_path = audio_source
            owns_tmp = False

        try:
            segments, _ = self.stt_model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()
        finally:
            if owns_tmp:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

        latency = time.perf_counter() - start
        print(f"STT latency: {latency:.2f}s  |  text: {text!r}")
        return text, latency

    # ── TTS ──────────────────────────────────────────────────────────────

    def speak(self, text: str) -> tuple[str, float]:
        """
        Synthesise speech from text.

        Returns:
            (output_wav_path, tts_latency_seconds)
        """
        start = time.perf_counter()
        output_path = "agent_response.wav"
        with open(output_path, "wb") as f:
            self.voice.synthesize(text, f)
        latency = time.perf_counter() - start
        print(f"TTS latency: {latency:.2f}s")
        return output_path, latency

    # ── Full pipeline ─────────────────────────────────────────────────────

    def voice_pipeline(self, audio_source) -> dict:
        """
        Run the full STT → Agent → TTS pipeline.

        Args:
            audio_source: file path or Streamlit UploadedFile.

        Returns:
            dict with keys:
                ``query``         — transcribed user text  
                ``answer``        — agent answer text  
                ``audio_path``    — path to synthesised WAV  
                ``stt_latency``   — seconds  
                ``agent_latency`` — seconds  
                ``tts_latency``   — seconds  
                ``total_latency`` — seconds  
                ``error``         — error message or None  
        """
        result = {
            "query": "",
            "answer": "",
            "audio_path": None,
            "stt_latency": 0.0,
            "agent_latency": 0.0,
            "tts_latency": 0.0,
            "total_latency": 0.0,
            "error": None,
        }

        if audio_source is None:
            result["error"] = "No audio received. Please record again."
            return result

        total_start = time.perf_counter()

        try:
            # 1. STT
            query, stt_lat = self.transcribe_audio(audio_source)
            result["stt_latency"] = stt_lat

            if not query:
                result["error"] = "Could not understand audio. Please try again."
                return result

            result["query"] = query
            print(f"User said: {query}")

            # 2. Agent
            agent_start = time.perf_counter()
            answer = self.agent_func(query)
            result["agent_latency"] = time.perf_counter() - agent_start
            result["answer"] = answer

            # 3. TTS
            audio_path, tts_lat = self.speak(answer)
            result["audio_path"] = audio_path
            result["tts_latency"] = tts_lat

        except Exception as exc:
            result["error"] = str(exc)
        finally:
            result["total_latency"] = time.perf_counter() - total_start
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────

_voice_interface: VoiceInterface | None = None


def get_voice_interface(agent_func) -> VoiceInterface:
    global _voice_interface
    if _voice_interface is None:
        _voice_interface = VoiceInterface(agent_func)
    return _voice_interface