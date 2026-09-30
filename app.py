import os
import io
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# Safe pydub import for volume boosting
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ==========================================
# 1. Page Configuration & Light Purple Theme
# ==========================================
st.set_page_config(
    page_title="Hybrid Voice RAG AI Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background */
    .stApp { 
        background: #f3e8ff !important; 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Styles */
    .main-title { 
        font-size: 2.6rem; 
        font-weight: 800; 
        color: #9333ea !important; 
        text-align: center; 
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .sub-title { 
        text-align: center; 
        color: #6b21a8 !important; 
        font-size: 1.1rem; 
        font-weight: 600; 
        margin-bottom: 2rem; 
    }
    
    /* Sidebar Layout */
    [data-testid="stSidebar"] { 
        background-color: #fcfaff !important; 
        border-right: 1px solid #e9d5ff !important; 
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { 
        color: #4c1d95 !important; 
        font-size: 1.25rem !important;
        font-weight: 700 !important; 
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { 
        color: #581c87 !important; 
        font-weight: 600 !important; 
    }
    
    /* Chat Message Cards */
    [data-testid="stChatMessage"] { 
        background-color: #ffffff !important; 
        border: 1px solid #f0abfc !important; 
        border-radius: 20px !important; 
        padding: 1.25rem 1.5rem !important; 
        margin-bottom: 1.2rem !important; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom Sidebar Action Button */
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%) !important; 
        color: #ffffff !important; 
        font-weight: 700 !important; 
        border: none !important; 
        border-radius: 12px !important; 
        padding: 0.7rem 1.2rem !important; 
        box-shadow: 0 2px 4px rgba(147, 51, 234, 0.2);
    }
    
    /* Audio Player Styling */
    audio { 
        border-radius: 30px; 
        width: 100%; 
        margin-top: 10px; 
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
    st.stop()

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=api_key)

client = get_openai_client()

# ==========================================
# 2. Multi-Format Hybrid RAG Implementation
# ==========================================
class HybridRAG:
    def __init__(self, uploaded_files, chunk_size=1000, chunk_overlap=150):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from rank_bm25 import BM25Okapi

        all_docs = []
        self.indexed_filenames = []

        for uploaded_file in uploaded_files:
            ext = uploaded_file.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                file_docs = self._load_file(tmp_file_path, ext)
                all_docs.extend(file_docs)
                self.indexed_filenames.append(uploaded_file.name)
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = text_splitter.split_documents(all_docs)

        if len(self.chunks) > 0:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            self.vector_db = Chroma.from_documents(self.chunks, embeddings)

            corpus = [doc.page_content.split(" ") for doc in self.chunks]
            self.bm25 = BM25Okapi(corpus)

    def _load_file(self, file_path, ext):
        if ext == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(file_path).load()
        elif ext in ["docx", "doc"]:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(file_path).load()
        elif ext == "csv":
            from langchain_community.document_loaders import CSVLoader
            return CSVLoader(file_path).load()
        elif ext == "txt":
            from langchain_community.document_loaders import TextLoader
            return TextLoader(file_path, encoding="utf-8").load()
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def search(self, query, top_k=8):
        meta_keywords = ["skill", "resume", "experience", "profile", "qualification", "analyze", "summarize", "about", "overview", "steps", "list"]
        if any(word in query.lower() for word in meta_keywords):
            return "\n\n".join([doc.page_content for doc in self.chunks[:10]])

        vector_results = self.vector_db.similarity_search(query, k=top_k)
        tokenized_query = query.split(" ")
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)

        combined_context = []
        seen_texts = set()

        for doc in vector_results + bm25_docs:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                combined_context.append(doc.page_content)

        return "\n\n".join(combined_context)

    def generate_summary(self):
        full_text = "\n".join([doc.page_content for doc in self.chunks[:10]])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Provide a clear, structured summary of these uploaded documents in English. "
                        "Preserve step-by-step formats, bullet points, and key lists exactly as presented. "
                        "Always conclude with a reminder to upload documents if further analysis is needed."
                    )
                },
                {"role": "user", "content": f"Summarize:\n\n{full_text}"}
            ]
        )
        return response.choices[0].message.content

# ==========================================
# 3. Speech Processing & Dynamic Language Pipeline
# ==========================================
def transcribe_audio_file(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio_path = tmp_audio.name

    try:
        with open(tmp_audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

def generate_openai_speech(text, amplify_db=10):
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="onyx",  # Clear, loud, bold voice
        speed=0.92,
        input=text
    )
    audio_data = response.content

    if PYDUB_AVAILABLE:
        try:
            sound = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            loud_sound = sound + amplify_db
            output_buffer = io.BytesIO()
            loud_sound.export(output_buffer, format="mp3")
            return output_buffer.getvalue()
        except Exception:
            return audio_data

    return audio_data

def stream_multilingual_response(user_query, rag_engine=None):
    base_instructions = (
        "CORE OPERATIONAL RULES:\n"
        "1. EXACT PDF FORMATTING & STRUCTURE: When answering questions from an uploaded document or PDF, reproduce the information EXACTLY as it is structured in the source. If the PDF content is in step-by-step format, bullet points, lists, or distinct sections, present it in the EXACT SAME FORMAT. Do NOT merge items into a single continuous paragraph. Preserve layout, sequence, and list structures.\n"
        "2. DEFAULT ENGLISH PRIORITY: Your primary and default language for text and general responses MUST ALWAYS BE ENGLISH.\n"
        "3. ADAPTIVE MULTILINGUAL SWITCHING: Pay close attention to the user's input language. If the user speaks or asks in Tamil, Telugu, Hindi, Malayalam, Kannada, or Urdu, immediately switch to that requested language and respond warmly, clearly, and naturally in it, maintaining the exact required structure/formatting.\n"
        "4. MANDATORY DOCUMENT REMINDER: Always end your response by reminding the user to upload a PDF or document if they haven't yet (or to upload additional files), spoken in the current response language (e.g., in English: 'Please upload your PDF or document so I can analyze it and answer from it.').\n"
        "5. AUDIO CLEANLINESS: Avoid unnecessary visual ASCII ornamentation that disrupts speech output while preserving clear numbered lists or bullet lines."
    )

    if rag_engine is None:
        system_prompt = (
            f"{base_instructions}\n\n"
            "GENERAL INTERACTION MODE:\n"
            "No document is currently loaded. Answer general queries directly in the active response language, preserve list/step structures if requested, and ALWAYS append the mandatory document upload reminder at the very end."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
    else:
        context = rag_engine.search(user_query)
        system_prompt = (
            f"{base_instructions}\n\n"
            "STRICT RAG & DOCUMENT GROUNDING MODE:\n"
            "1. Base all your answers strictly on the uploaded document context below.\n"
            "2. Reproduce any step-by-step instructions, bullet points, or sections in the exact structure they appear in the source.\n"
            "3. Do NOT invent external details not present in the text.\n"
            "4. Always conclude your answer with the mandatory document upload reminder in the active response language.\n\n"
            f"Document Context:\n{context}"
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# ==========================================
# 4. State Management & Initial Greeting
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Mandatory loud greeting text
    initial_greeting = "Hi, I am the Hybrid Voice RAG AI assistant. How can I help you today? Please upload your document to get started."
    try:
        greeting_audio = generate_openai_speech(initial_greeting, amplify_db=12)
    except Exception:
        greeting_audio = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": initial_greeting,
        "audio": greeting_audio
    })

if "rag_engine" not in st.session_state:
    st.session_state["rag_engine"] = None

# ==========================================
# 5. UI Layout
# ==========================================
st.markdown('<div class="main-title">🎙️ Hybrid Voice RAG AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multilingual Document Intelligence</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📄 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "docx", "doc", "csv", "txt"],
        accept_multiple_files=True
    )

    with st.expander("⚙️ Advanced Chunk Settings"):
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=300, value=150, step=10)

    if uploaded_files:
        current_filenames = [f.name for f in uploaded_files]
        if st.session_state.get("indexed_filenames") != current_filenames:
            with st.spinner("Processing & Indexing Documents..."):
                rag_instance = HybridRAG(uploaded_files, chunk_size, chunk_overlap)
                
                if len(rag_instance.chunks) == 0:
                    st.error("⚠️ No readable text found in uploaded file(s)!")
                    st.session_state["rag_engine"] = None
                else:
                    st.session_state["rag_engine"] = rag_instance
                    st.session_state["indexed_filenames"] = current_filenames
                    st.success("✅ Documents indexed successfully!")

    if st.session_state.get("rag_engine") is not None:
        doc_count = len(st.session_state["rag_engine"].indexed_filenames)
        st.markdown(f"### 📑 Uploaded Documents ({doc_count})")
        for fname in st.session_state["rag_engine"].indexed_filenames:
            st.markdown(f"- 📄 `{fname}`")

    st.divider()
    st.markdown("### 🛠 Actions & Tools")

    if st.button("📑 Summarize Documents"):
        if st.session_state.get("rag_engine") is None:
            st.warning("Please upload at least one document first!")
        else:
            with st.spinner("Generating summary..."):
                summary = st.session_state["rag_engine"].generate_summary()
                st.info(f"Document Summary:\n\n{summary}")

# Render Chat Feed
for idx, msg in enumerate(st.session_state.messages):
    avatar_icon = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.write(msg["content"])
        if msg.get("audio"):
            autoplay = True if idx == 0 and len(st.session_state.messages) == 1 else False
            st.audio(msg["audio"], format="audio/mp3", autoplay=autoplay)

st.divider()
col1, col2 = st.columns([1, 4])

user_query = None

with col1:
    st.write("🎙️ **Voice Control:**")
    audio_data = mic_recorder(start_prompt="Record", stop_prompt="Stop", key="voice_recorder")
    if audio_data and "bytes" in audio_data:
        with st.spinner("Processing voice..."):
            user_query = transcribe_audio_file(audio_data["bytes"])

with col2:
    text_input = st.chat_input("Ask a question about your documents or start talking...")
    if text_input:
        user_query = text_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query, "audio": None})
    with st.chat_message("user", avatar="👤"):
        st.write(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        rag = st.session_state.get("rag_engine")
        ans = st.write_stream(stream_multilingual_response(user_query, rag_engine=rag))
        
        with st.spinner("Generating audio..."):
            audio_bytes = generate_openai_speech(ans)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

    st.session_state.messages.append({"role": "assistant", "content": ans, "audio": audio_bytes})