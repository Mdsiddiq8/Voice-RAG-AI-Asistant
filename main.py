import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import pygame
import speech_recognition as sr

# RAG Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def speak(text):
    """Converts text to speech using OpenAI TTS and plays it aloud."""
    print(f"\n🤖 AI: {text}")

    audio_file = "temp_speech.mp3"

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(audio_file)

    pygame.mixer.quit()
    pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)

    try:
        sound = pygame.mixer.Sound(audio_file)
        channel = sound.play()
        while channel.get_busy():
            time.sleep(0.1)
    finally:
        pygame.mixer.quit()
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except PermissionError:
                pass

def listen():
    """Captures microphone audio and transcribes it into text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️ Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        audio = recognizer.listen(source)

    try:
        print("⏳ Transcribing speech...")
        text = recognizer.recognize_google(audio)
        print(f"👤 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("⚠️ Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"⚠️ Speech Recognition service error: {e}")
        return None

class HybridRAG:
    """Handles PDF loading, text chunking, Vector Search, and BM25 Search."""

    def __init__(self, pdf_path):
        print(f"\n📄 Loading and indexing PDF: {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = text_splitter.split_documents(docs)

        # 1. Vector Store Setup (Dense Search)
        embeddings = OpenAIEmbeddings()
        self.vector_db = Chroma.from_documents(self.chunks, embeddings)

        # 2. BM25 Setup (Sparse Keyword Search)
        corpus = [doc.page_content.split(" ") for doc in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=2):
        """Combines Vector Search and BM25 Keyword Search (Hybrid RAG)."""
        # Dense Vector Search
        vector_results = self.vector_db.similarity_search(query, k=top_k)

        # Sparse BM25 Keyword Search
        tokenized_query = query.split(" ")
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)

        # Combine unique contexts
        combined_context = []
        seen_texts = set()

        for doc in vector_results + bm25_docs:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                combined_context.append(doc.page_content)

        return "\n\n".join(combined_context)

def answer_query(rag_engine, user_query):
    """Retrieves context using Hybrid RAG and calls OpenAI GPT to answer."""
    context = rag_engine.search(user_query)

    system_prompt = (
        "You are a helpful voice assistant answering questions based on the provided document. "
        "Keep your answers concise, clear, and natural for speech output (1-3 sentences).\n\n"
        f"Context from document:\n{context}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    pdf_filename = "sample.pdf"

    # Check if sample.pdf exists
    if not os.path.exists(pdf_filename):
        print(f"❌ Error: Please place a file named '{pdf_filename}' in your Voice_rag folder.")
        exit()

    # Step A: Greet User
    speak("Hi, how can I assist you today? I have loaded your document and am ready to answer your questions.")

    # Step B: Initialize Hybrid RAG
    rag = HybridRAG(pdf_filename)
    speak("Document indexed successfully. Ask me anything about it!")

    # Step C: Live Voice Q&A Loop
    while True:
        user_speech = listen()
        if not user_speech:
            continue

        # Check for exit command
        if "exit" in user_speech.lower() or "stop" in user_speech.lower() or "bye" in user_speech.lower():
            speak("Goodbye! Have a great day.")
            break

        # Get RAG Answer
        answer = answer_query(rag, user_speech)

        # Print on screen and read aloud
        speak(answer)