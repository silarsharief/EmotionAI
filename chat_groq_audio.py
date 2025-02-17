import os
import torch
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from TTS.api import TTS
import torch.serialization
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = "hackathon"

# Add required configs to safe globals for Coqui
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

### 🎤 **Chat Manager (Handles LLM Response)**
class ChatManager:
    def __init__(self, model_name="mixtral-8x7b-32768"):
        """Initialize the chat manager"""
        print("Initializing Chat Model...")
        try:
            self.llm = ChatGroq(model=model_name)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a conversation AI bot. Your job is to have a casual conversation with the user as a friend. Express emotions and match the user's tone. Be dramatic if needed. Each reply should and MUST be short and consise and sassy. You basic english punctuation that is easy for a tts model to comprehend. use full words only. no emojis"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            self.conversation_history = []  # Store chat history
            self.max_history = 7  # Keep last 7 messages
            print(f"Successfully connected to the model: {model_name}")
        except Exception as e:
            print(f"Error initializing ChatGroq: {e}")
            raise e

    def add_to_history(self, human_message, ai_message):
        """Store conversation history and limit size"""
        self.conversation_history.append((human_message, ai_message))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_response(self, text):
        """Get response from LLM"""
        try:
            formatted_history = []
            for human_msg, ai_msg in self.conversation_history:
                formatted_history.append(HumanMessage(content=human_msg))
                formatted_history.append(AIMessage(content=ai_msg))

            chain = self.prompt | self.llm
            response = chain.invoke({
                "history": formatted_history,
                "input": text
            })
            
            # Store conversation history
            self.add_to_history(text, response.content)

            return response.content
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return None

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

### 🔊 **TTS Function (Plays Response Directly)**
def generate_and_play_text(text, speaker_wav="reference_speaker.wav", language="en", sample_rate=24000):
    """Generate speech and play it live (No file saving)"""
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph").to(device)

    # Ensure reference speaker file exists
    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"Speaker audio file '{speaker_wav}' not found. Provide a valid .wav file.")

    # Generate raw waveform (No saving, just playback)
    wav = tts.tts(text=text)

    # Convert to NumPy array
    wav_np = np.array(wav, dtype=np.float32)

    # Play generated speech in real-time
    print("🔊 Playing generated response...")
    sd.play(wav_np, samplerate=sample_rate)
    sd.wait()  # Ensure playback completes
    print("✅ Playback complete.")

### 🎙️ **Main Loop for Real-Time Chat**
if __name__ == "__main__":
    chat_manager = ChatManager()  # Initialize Chat LLM

    print("🎤 AI Chatbot with Live TTS 🎤")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break
        
        # Get AI response
        ai_response = chat_manager.get_response(user_input)
        print(f"AI: {ai_response}")

        # Convert AI response to speech and play it
        generate_and_play_text(ai_response)
