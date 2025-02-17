import os
import torch
import pyaudio
import numpy as np
import threading
import queue
import sounddevice as sd
from faster_whisper import WhisperModel
from langchain_groq import ChatGroq
from chat_groq_audio import ChatManager
from TTS.api import TTS
import torch.serialization
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add required configs to safe globals for Coqui TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig])

# Patch torch.load to avoid 'weights_only' issues
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class VoiceChatAssistant:
    def __init__(self, model_size="base", device="cuda", compute_type="float16", model_name="mixtral-8x7b-32768"):
        # Initialize Whisper Model
        print("Initializing Whisper model...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=min(os.cpu_count(), 4)
        )

        # Initialize Chat Manager
        self.chat_manager = ChatManager(model_name=model_name)

        # Initialize TTS Model (Tacotron2)
        print("Initializing TTS model...")
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

        # Audio settings
        self.CHUNK = 1600
        self.RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paFloat32

        # Silence detection settings
        self.SILENCE_THRESHOLD = 0.02
        self.SILENCE_CHUNKS = int(self.RATE * 0.5 / self.CHUNK)
        self.MIN_AUDIO_LENGTH = int(self.RATE * 2 / self.CHUNK)

        # Initialize queues and flags
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.is_processing = False

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Add a new event for synchronization
        self.tts_done = threading.Event()
        self.tts_done.set()  # Initially set to True

    def get_ai_response(self, text):
        """Get response from Chat Manager and play it using TTS"""
        try:
            response = self.chat_manager.get_response(text)
            print(f"\n🤖 Assistant: {response}")
            self.speak_response(response)  # Speak response using TTS
        except Exception as e:
            print(f"Error getting AI response: {e}")
        finally:
            # Make sure processing flag is cleared
            self.is_processing = False
            # Make sure TTS done event is set
            self.tts_done.set()

    def speak_response(self, text, sample_rate=24000):
        """Generate speech and play it live, handling short sentences properly."""
        try:
            print("🔊 Speaking response...")

            # Split text into sentences while preserving punctuation
            parts = []
            current_part = ""
            
            # First, normalize all sentence endings to use periods
            text = text.strip()
            # Keep original punctuation by not replacing them
            sentences = []
            current = ""
            
            for char in text:
                current += char
                if char in '.!?':
                    sentences.append(current.strip())
                    current = ""
            if current:  # Add any remaining text
                sentences.append(current.strip())
            
            # Combine short sentences while preserving punctuation
            for sentence in sentences:
                if not sentence:
                    continue
                
                # If current_part is empty or combining won't exceed length limit
                if not current_part or len(current_part) + len(sentence) < 100:
                    current_part += " " + sentence
                else:
                    if current_part:
                        parts.append(current_part.strip())
                    current_part = sentence
            
            if current_part:
                parts.append(current_part.strip())

            # Process each part
            for part in parts:
                try:
                    print(f"🔊 Processing: {part[:50]}...")
                    
                    # Ensure the part ends with punctuation
                    if not part[-1] in '.!?':
                        part += '.'
                    
                    # Generate speech
                    wav = self.tts.tts(text=part)
                    wav_np = np.array(wav, dtype=np.float32)
                    
                    # Normalize audio
                    wav_np = wav_np / np.max(np.abs(wav_np))
                    
                    # Add small pause between parts
                    pause = np.zeros(int(sample_rate * 0.3))  # 0.3 second pause
                    wav_np = np.concatenate([wav_np, pause])
                    
                    # Play with blocking
                    sd.play(wav_np, samplerate=sample_rate, blocking=True)
                    sd.wait()  # Ensure playback is complete
                    
                except Exception as e:
                    print(f"Error processing part: {e}")
                    continue

            print("✅ Playback complete.")
            
        except Exception as e:
            print(f"Error in TTS playback: {e}")
        finally:
            # Ensure flags are reset properly
            self.is_processing = False
            self.tts_done.set()

    def start(self):
        """Start the voice chat"""
        self.is_running = True

        # Start audio stream
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        print("\n🎤 Listening...")

        # Start worker threads
        self.record_thread = threading.Thread(target=self._record_audio)
        self.transcribe_thread = threading.Thread(target=self._process_audio)

        self.record_thread.start()
        self.transcribe_thread.start()

    def stop(self):
        """Stop the voice chat"""
        print("\nStopping voice chat assistant...")
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _record_audio(self):
        """Record audio from microphone"""
        while self.is_running:
            try:
                # Only record if not processing and TTS is done
                if self.is_processing or not self.tts_done.is_set():
                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)
                    continue

                audio_chunk = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
                self.audio_queue.put(audio_data)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def _is_silent(self, audio_chunk):
        """Check if the audio chunk is silent"""
        return np.mean(np.abs(audio_chunk)) < self.SILENCE_THRESHOLD

    def _process_audio(self):
        """Process and transcribe audio chunks"""
        audio_buffer = []
        silence_count = 0

        while self.is_running:
            try:
                # Check if we can process audio
                if self.is_processing:
                    time.sleep(0.1)
                    continue

                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if self._is_silent(audio_chunk):
                    silence_count += 1
                else:
                    silence_count = 0
                    audio_buffer.extend(audio_chunk)

                # Process buffer when silence is detected or buffer is too large
                if (silence_count >= self.SILENCE_CHUNKS and len(audio_buffer) >= self.MIN_AUDIO_LENGTH):
                    self.is_processing = True
                    
                    audio_data = np.array(audio_buffer)
                    
                    # Transcribe audio
                    segments, _ = self.model.transcribe(
                        audio_data,
                        language="en",
                        vad_filter=True,
                        beam_size=1
                    )

                    # Get transcription
                    transcript = " ".join([segment.text for segment in segments]).strip()
                    if transcript:
                        print(f"\n🗣️  You: {transcript}")
                        self.get_ai_response(transcript)

                    # Clear buffer
                    audio_buffer = []

            except Exception as e:
                print(f"Error processing audio: {e}")
            finally:
                # Always ensure flags are reset
                self.is_processing = False
                self.tts_done.set()

def main():
    """Start the real-time voice assistant"""
    assistant = VoiceChatAssistant(model_name="mixtral-8x7b-32768")

    print("Starting voice chat assistant... Speak into your microphone")
    print("Press Ctrl+C to stop")

    try:
        assistant.start()
        while True:
            try:
                input()  # Keeps the loop running
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        pass
    finally:
        assistant.stop()

if __name__ == "__main__":
    main()
