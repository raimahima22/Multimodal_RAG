import time
import torch
import numpy as np
import sounddevice as sd
import gradio as gr
import os

from faster_whisper import WhisperModel
from piper import PiperVoice  # NEW (instead of TTS)


MODEL_PATH = os.path.join(
    "/content/drive/MyDrive/piper_models",
    "en_US-lessac-medium.onnx"
)

class VoiceInterface:
    def __init__(self, agent_func):
        self.agent_func = agent_func

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --------------------
        # STT (Whisper)
        # --------------------
        print("Loading faster-whisper...")
        self.stt_model = WhisperModel(
            "small.en",
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8"
        )

        # --------------------
        # TTS (Piper)
        # --------------------
        print("Loading Piper TTS...")

        # Path to downloaded model (YOU MUST SET THIS)
        self.voice = PiperVoice.load(
            model_path=MODEL_PATH
        )

        self.sample_rate = 22050

    # --------------------
    # STT
    # --------------------
    def transcribe_audio(self, audio_path: str) -> str:
        start = time.time()

        segments, _ = self.stt_model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            vad_filter=True
        )

        text = " ".join(seg.text for seg in segments).strip()

        print(f"STT Latency: {time.time() - start:.2f}s")

        return text

    # --------------------
    # TTS (Piper)
    # --------------------
    def speak(self, text: str) -> str:
        start = time.time()

        output_path = "agent_response.wav"

        with open(output_path, "wb") as f:
            self.voice.synthesize(text, f)

        print(f"TTS Latency: {time.time() - start:.2f}s")

        return output_path

    # --------------------
    # PIPELINE
    # --------------------
    def voice_pipeline(audio):
        """Full voice → agent → voice response (with latency tracking)"""
    
        if audio is None:
            return None, "No audio received. Please record again."

        try:
            total_start = time.time()

            # Voice pipeline (STT + LLM + TTS)
       
            pipeline_start = time.time()
            audio_path, result_text = voice_interface.voice_pipeline(audio)
            pipeline_latency = time.time() - pipeline_start

            total_latency = time.time() - total_start

            # Append latency info to output text
            result_text += (
                f"\n\n **Pipeline Latency:** {pipeline_latency:.2f}s"
                f"\n **Total Latency:** {total_latency:.2f}s"
            )

            return audio_path, result_text

        except Exception as e:
            return None, f"Error in voice pipeline: {str(e)}"


voice_interface = None


def get_voice_interface(agent_func):
    global voice_interface
    if voice_interface is None:
        voice_interface = VoiceInterface(agent_func)
    return voice_interface