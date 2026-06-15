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


# MODEL_PATH = os.path.join(
#     "/content/drive/MyDrive/piper_models",
#     "en_US-amy-medium.onnx"
# )

MODEL_PATH = os.path.join(
    "/content/piper_models",
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
        """
        Initialize STT and TTS models.

        Args:
            agent_func (callable): Function that processes transcribed text
                                   and returns agent response.
        """

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # compute = "float16" if self.device == "cuda" else "int8"

        self.agent_func = agent_func
        self.device = "cpu"   # whisper runs fine on CPU
    


        # STT (speech-to-text model initialization)
        print("[VOICE] Loading faster-whisper (base.en)...")
        self.stt_model = WhisperModel(
            "base.en",
            device=self.device,
            compute_type="int8",
        )

        # TTS (text-to-speech model initialization)
        print(f"[VOICE] Loading Piper TTS from {MODEL_PATH}...")
        self.voice = PiperVoice.load(model_path=MODEL_PATH)
        self.sample_rate = 22050

         #    the ONNX runtime is JIT-compiled before the first real query.
        print("[VOICE] Pre-warming Piper TTS...")
        for _ in self.voice.synthesize("ready"):
            pass

        print("[VOICE] Models ready.")

    # STT (speech-to-text)
    def transcribe_audio(self, audio_path: str) -> tuple[str, float]:
        """
        Pipeline:
            audio file → Whisper model → text transcription

        Args:
            audio_path (str): Path to the input audio file

        Returns:
            tuple:
                - transcribed_text (str): Final merged transcription
                - stt_latency_seconds (float): Time taken for transcription
        """
        t0 = time.time()
        segments, _ = self.stt_model.transcribe(
            audio_path,
            beam_size=5, # beam search for better accuracy
            language="en", # force English decoding
            vad_filter=True, # removes silence/non-speech segments
        )
        text = " ".join(seg.text for seg in segments).strip()
        stt_time = round(time.time() - t0, 2)
        print(f"[STT] '{text[:60]}...' | {stt_time}s")
        return text, stt_time

    # TTS (text-to-speech)

    def speak_stream(self, text: str):
        """
        Stream synthesized speech audio chunks.

        This method yields audio in real-time for UI frameworks like Gradio.

        Args:
            text (str): Input text to synthesize

        Yields:
            tuple:
                (sample_rate, audio_chunk_np_array)
        """
        for chunk in self.voice.synthesize(text):
            yield (chunk.sample_rate, chunk.audio_float_array)


    def speak_to_file(self, text: str) -> tuple[str | None, float]:
        """
        Generate speech audio and save it as a WAV file.

        Pipeline:
            text → Piper TTS → audio chunks → concatenation → .wav file

        Args:
            text (str): Text to synthesize

        Returns:
            tuple:
                - output_path (str | None): Path to saved WAV file
                - tts_latency_seconds (float): Time taken for synthesis
        """
        t0 = time.time()
        try:
            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/tts_{uuid.uuid4().hex}.wav"

            chunks = []
            sample_rate = self.sample_rate

            # Generate audio chunks from Piper
            for chunk in self.voice.synthesize(text):
                sample_rate = chunk.sample_rate
                chunks.append(chunk.audio_float_array)

            if not chunks:
                print("[TTS] No audio chunks generated.")
                return None, round(time.time() - t0, 2)

            # combine chunks into final waveform
            audio = np.concatenate(chunks)

            # save WAV file
            sf.write(output_path, audio, sample_rate)
            tts_time = round(time.time() - t0, 2)
            print(f"[TTS] Written to {output_path} | {tts_time}s")
            return output_path, tts_time

        except Exception as e:
            print(f"[TTS] Error: {e}")
            return None, round(time.time() - t0, 2)

    # Cleanup 

    def cleanup(self):
        """
        Free up system and GPU resources.

        - Runs Python garbage collection
        - Clears CUDA cache if GPU is used
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Singleton pattern (prevents reloading models multiple times)

_voice_interface = None


def get_voice_interface(agent_func) -> VoiceInterface:
    """
    Singleton accessor for VoiceInterface.

    Ensures that STT and TTS models are loaded only once
    across the application lifecycle.

    Args:
        agent_func (callable): Agent function for downstream pipeline

    Returns:
        VoiceInterface: Initialized shared instance
    """
    global _voice_interface
    if _voice_interface is None:
        _voice_interface = VoiceInterface(agent_func)
    return _voice_interface