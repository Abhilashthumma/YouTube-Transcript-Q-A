# app.py — YouTube Transcript Q&A App
# Stack: Streamlit + Ollama (FREE, local) + ChromaDB + LangChain

import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA

# ── 1. Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Q&A", page_icon="🎬")
st.title("🎬 YouTube Transcript Q&A")
st.caption("100% FREE — powered by Ollama running locally on your machine.")

# ── 2. Sidebar — let user pick which Ollama model to use ──────────────────────
st.sidebar.header("⚙️ Ollama Settings")
chat_model = st.sidebar.text_input(
    "Chat model", value="llama3.2",
    help="Run: ollama pull llama3.2"
)
embed_model = st.sidebar.text_input(
    "Embedding model", value="nomic-embed-text",
    help="Run: ollama pull nomic-embed-text"
)
st.sidebar.info("Make sure Ollama is running:\n```ollama serve```")

# ── 3. Helper — extract video ID from any YouTube URL format ───────────────────
def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# ── 4. Helper — fetch transcript text from YouTube ─────────────────────────────
def fetch_transcript(video_id: str) -> str:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    return " ".join(item["text"] for item in transcript_list)

# ── 5. Helper — build ChromaDB vector store using Ollama embeddings ────────────
def build_vectorstore(transcript_text: str) -> Chroma:
    # Split transcript into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(transcript_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Use Ollama for embeddings — runs locally, no cost
    embeddings = OllamaEmbeddings(model=embed_model)

    # Store in ChromaDB (in-memory)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    return vectorstore

# ── 6. Helper — answer question using local Ollama LLM ────────────────────────
def answer_question(vectorstore: Chroma, question: str) -> str:
    # Use Ollama chat model — runs locally, no cost
    llm = ChatOllama(model=chat_model, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
    )

    result = qa_chain.invoke({"query": question})
    return result["result"]

# ── 7. UI — URL input and transcript loader ────────────────────────────────────
url = st.text_input("🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Load Transcript", type="primary"):
    if not url.strip():
        st.warning("Please paste a YouTube URL first.")
    else:
        video_id = extract_video_id(url.strip())
        if not video_id:
            st.error("❌ Could not parse a video ID from that URL. Check the link and try again.")
        else:
            with st.spinner("Fetching transcript…"):
                try:
                    transcript_text = fetch_transcript(video_id)
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                    st.stop()
                except NoTranscriptFound:
                    st.error("❌ No English transcript found for this video.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Failed to fetch transcript: {e}")
                    st.stop()

            with st.spinner("Building vector store with Ollama embeddings… (first time may be slow)"):
                try:
                    vectorstore = build_vectorstore(transcript_text)
                except Exception as e:
                    st.error(f"❌ Ollama error: {e}\n\nMake sure Ollama is running (`ollama serve`) and the model is pulled.")
                    st.stop()

            st.session_state["vectorstore"] = vectorstore
            word_count = len(transcript_text.split())
            st.success(f"✅ Transcript loaded! ({word_count:,} words) Ask your questions below.")

# ── 8. UI — Q&A section ────────────────────────────────────────────────────────
if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("💬 Ask a question about the video")

    question = st.chat_input("e.g. What is the main topic of this video?")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking… (Ollama is working locally)"):
                try:
                    answer = answer_question(st.session_state["vectorstore"], question)
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error: {e}")