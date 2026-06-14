# src/voice.py
import time
import os
import uuid
import numpy as np
import soundfile as sf
import torch
import gc

from faster_whisper import WhisperModel
from piper import PiperVoice

# MODEL_PATH = os.environ.get(
#     "PIPER_MODEL_PATH",
#     "models/en_US-amy-medium.onnx"         
# )


MODEL_PATH = os.path.join(
    "/content/drive/MyDrive/piper_models",
    "en_US-amy-medium.onnx"
)


class VoiceInterface:
    """
    Speech-to-text and text-to-speech wrapper.

    STT: faster-whisper (base.en)
    TTS: Piper (en_US-amy-medium)

    All three latency timings (stt_time, tts_time) are measured and returned
    so the UI can display them alongside the agent's own timing.
    """

    def __init__(self, agent_func):
        self.agent_func = agent_func
        self.device = "cpu"   # whisper runs fine on CPU; change to "cuda" if available

        # ── STT ──────────────────────────────────────────────
        print("[VOICE] Loading faster-whisper (base.en)...")
        self.stt_model = WhisperModel(
            "base.en",
            device=self.device,
            compute_type="int8",
        )

        # ── TTS ──────────────────────────────────────────────
        print(f"[VOICE] Loading Piper TTS from {MODEL_PATH}...")
        self.voice = PiperVoice.load(model_path=MODEL_PATH)
        self.sample_rate = 22050
        print("[VOICE] Models ready.")

    # ── STT ──────────────────────────────────────────────────────────────────

    def transcribe_audio(self, audio_path: str) -> tuple[str, float]:
        """
        Transcribe audio file to text.

        Returns:
            (transcribed_text, stt_latency_seconds)
        """
        t0 = time.time()
        segments, _ = self.stt_model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments).strip()
        stt_time = round(time.time() - t0, 2)
        print(f"[STT] '{text[:60]}...' | {stt_time}s")
        return text, stt_time

    # ── TTS ──────────────────────────────────────────────────────────────────

    def speak_stream(self, text: str):
        """
        Yield (sample_rate, audio_float32_chunk) for Gradio streaming audio.
        Does NOT measure total TTS time since it's a generator — timing is
        handled by the caller.
        """
        for chunk in self.voice.synthesize(text):
            yield (chunk.sample_rate, chunk.audio_float_array)

    def speak_to_file(self, text: str) -> tuple[str | None, float]:
        """
        Synthesize speech to a WAV file.

        Returns:
            (output_path_or_None, tts_latency_seconds)
        """
        t0 = time.time()
        try:
            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/tts_{uuid.uuid4().hex}.wav"

            chunks = []
            sample_rate = self.sample_rate
            for chunk in self.voice.synthesize(text):
                sample_rate = chunk.sample_rate
                chunks.append(chunk.audio_float_array)

            if not chunks:
                print("[TTS] No audio chunks generated.")
                return None, round(time.time() - t0, 2)

            audio = np.concatenate(chunks)
            sf.write(output_path, audio, sample_rate)
            tts_time = round(time.time() - t0, 2)
            print(f"[TTS] Written to {output_path} | {tts_time}s")
            return output_path, tts_time

        except Exception as e:
            print(f"[TTS] Error: {e}")
            return None, round(time.time() - t0, 2)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Singleton ─────────────────────────────────────────────────────────────────

_voice_interface = None


def get_voice_interface(agent_func) -> VoiceInterface:
    global _voice_interface
    if _voice_interface is None:
        _voice_interface = VoiceInterface(agent_func)
    return _voice_interface