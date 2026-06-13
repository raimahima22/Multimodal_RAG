import time
import torch
import numpy as np
import sounddevice as sd
import gradio as gr
import os

from faster_whisper import WhisperModel
from piper import PiperVoice  # NEW (instead of TTS)
import soundfile as sf

MODEL_PATH = os.path.join(
    "/content/drive/MyDrive/piper_models",
    "en_US-amy-medium.onnx"
)
# MODEL_PATH = os.path.join(
#     "models",
#     "en_US-amy-medium.onnx"
# )

class VoiceInterface:
    def __init__(self, agent_func):
        self.agent_func = agent_func

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        # --------------------
        # STT (Whisper)
        # --------------------
        print("Loading faster-whisper...")
        self.stt_model = WhisperModel(
            "base.en",
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8"
            # compute_type="int8"
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

   

    def speak_to_file(self, text: str):
        try:
            import os
            import uuid
            import soundfile as sf

            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/tts_{uuid.uuid4().hex}.wav"

            result = self.voice.synthesize(text)
            first_item = next(result)
            print(type(first_item))
            print(first_item)

            chunks = []
            sample_rate = self.sample_rate

            for chunk in result:
                sample_rate = chunk.sample_rate
                chunks.append(chunk.audio_float_array)
            
            if len(chunks) == 0:
                print("No audio chunks generated.")
                return None
            
            audio = np.concatenate(chunks)
            sf.write(output_path, audio, sample_rate)
            return output_path

            

        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def voice_pipeline(self, audio):
        """STT → Agent → TTS pipeline"""

        if audio is None:
            return None, "No audio received. Please record again."

        query = self.transcribe_audio(audio)

        if not query:
            return None, "Could not understand audio."

        print(f"User said: {query}")

        start = time.time()
        answer = self.agent_func(query)
        print(f"Agent Latency: {time.time() - start:.2f}s")

        audio_stream = self.speak_stream(answer)

        return audio_stream, answer

        gc.collect()
        torch.cuda.empty_cache()
    

    def speak_stream(self, text: str):
        """
        Yield audio chunks suitable for Gradio streaming.
        """
        for chunk in self.voice.synthesize(text):
            # chunk.audio_float_array is float32 numpy array
            yield (chunk.sample_rate, chunk.audio_float_array)


voice_interface = None


def get_voice_interface(agent_func):
    global voice_interface
    if voice_interface is None:
        voice_interface = VoiceInterface(agent_func)
    return voice_interface