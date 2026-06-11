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
    def voice_pipeline(self, audio):
        if audio is None:
            return None, "No audio recorded."

        query = self.transcribe_audio(audio)

        if not query:
            return None, "Could not understand audio."

        print(f"User said: {query}")

        start = time.time()
        answer = self.agent_func(query)
        print(f"Agent Latency: {time.time() - start:.2f}s")

        audio_path = self.speak(answer)

        return audio_path, f"You said: {query}\n\nAnswer: {answer}"


voice_interface = None


def get_voice_interface(agent_func):
    global voice_interface
    if voice_interface is None:
        voice_interface = VoiceInterface(agent_func)
    return voice_interface