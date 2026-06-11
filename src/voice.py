# src/voice.py
import time
from faster_whisper import WhisperModel
from TTS.api import TTS
import numpy as np
import sounddevice as sd
import gradio as gr
import torch

class VoiceInterface:
    def __init__(self, agent_func):
        self.agent_func = agent_func  # run_agent from src.agent
        
        # STT: faster-whisper (very fast on Colab GPU)
        print("Loading faster-whisper model...")
        self.stt_model = WhisperModel(
            "small.en",          # or "base.en" for faster / "medium.en" for better accuracy
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        
        # TTS: Coqui TTS (XTTS is best but heavy → fallback to lighter)
        print("Loading TTS model...")
        try:
            self.tts_model = TTS("tts_models/en/vctk/vits").to("cuda" if torch.cuda.is_available() else "cpu")
        except:
            print("Falling back to simpler TTS...")
            self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to("cpu")
        
        self.sample_rate = 16000

    def transcribe_audio(self, audio_path: str) -> str:
        """STT: Audio file → Text"""
        start = time.time()
        segments, _ = self.stt_model.transcribe(
            audio_path, 
            beam_size=5,
            language="en",
            vad_filter=True
        )
        text = " ".join(segment.text for segment in segments).strip()
        latency = time.time() - start
        print(f"STT Latency: {latency:.2f}s")
        return text

    def speak(self, text: str) -> str:
        """TTS: Text → Audio file"""
        start = time.time()
        output_path = "agent_response.wav"
        
        self.tts_model.tts_to_file(
            text=text,
            speaker="p225",          # change if using different voice
            file_path=output_path,
            speed=1.0
        )
        latency = time.time() - start
        print(f"TTS Latency: {latency:.2f}s")
        return output_path

    def voice_pipeline(self, audio):
        """Full pipeline: Voice → Agent → Voice"""
        if audio is None:
            return None, "No audio recorded."
        
        # 1. STT
        query = self.transcribe_audio(audio)
        if not query:
            return None, "Could not understand audio. Please try again."
        
        print(f"User said: {query}")
        
        # 2. Agent
        agent_start = time.time()
        answer = self.agent_func(query)
        agent_latency = time.time() - agent_start
        print(f"Agent Latency: {agent_latency:.2f}s")
        
        # 3. TTS
        audio_path = self.speak(answer)
        
        total_latency = time.time() - agent_start + (time.time() - agent_start)  # rough
        print(f"Total Voice Round-trip ≈ {total_latency:.2f}s")
        
        return audio_path, f"**You said:** {query}\n\n**Answer:** {answer}"

# Global instance (lazy)
voice_interface = None

def get_voice_interface(agent_func):
    global voice_interface
    if voice_interface is None:
        voice_interface = VoiceInterface(agent_func)
    return voice_interface