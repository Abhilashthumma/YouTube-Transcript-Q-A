
import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter   
from langchain_core.documents import Document                         
from langchain_ollama import OllamaEmbeddings, ChatOllama              
from langchain_chroma import Chroma                                   

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Q&A", page_icon="🎬")
st.title("🎬 YouTube Transcript Q&A")

# ── Sidebar — model selection ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Ollama Settings")
chat_model  = st.sidebar.text_input("Chat model",      value="llama3.2",        help="ollama pull llama3.2")
embed_model = st.sidebar.text_input("Embedding model", value="nomic-embed-text", help="ollama pull nomic-embed-text")
st.sidebar.info("Keep Ollama running:\n```\nollama serve\n```")

# ── Helper: extract YouTube video ID ──────────────────────────────────────────
def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

# ── Helper: fetch transcript from YouTube ─────────────────────────────────────
def fetch_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en"])
    return " ".join(snippet.text for snippet in transcript)

# ── Helper: build ChromaDB vector store ───────────────────────────────────────
def build_vectorstore(text: str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    embeddings = OllamaEmbeddings(model=embed_model)
    return Chroma.from_documents(docs, embedding=embeddings)

# ── Helper: answer question (no chain — direct retrieval + LLM call) ──────────
def answer_question(vectorstore: Chroma, question: str) -> str:
    # Step 1: find the 4 most relevant transcript chunks
    relevant_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # Step 2: build a plain prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the transcript context below.
If the answer is not in the context, say "I couldn't find that in the video transcript."

Context:
{context}

Question: {question}
Answer:"""

    # Step 3: call the local Ollama model directly
    llm = ChatOllama(model=chat_model, temperature=0)
    response = llm.invoke(prompt)
    return response.content

# ── UI: URL input ──────────────────────────────────────────────────────────────
url = st.text_input("🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Load Transcript", type="primary"):
    if not url.strip():
        st.warning("Please paste a YouTube URL first.")
    else:
        video_id = extract_video_id(url.strip())
        if not video_id:
            st.error("❌ Could not find a video ID in that URL. Please check and try again.")
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
                    st.error(f"❌ Transcript error: {e}")
                    st.stop()

            with st.spinner("Building vector store with Ollama… (may take ~30 sec)"):
                try:
                    vectorstore = build_vectorstore(transcript_text)
                except Exception as e:
                    st.error(f"❌ Ollama error: {e}\n\nIs Ollama running? Try: ollama serve")
                    st.stop()

            st.session_state["vectorstore"] = vectorstore
            st.success(f"✅ Ready! ({len(transcript_text.split()):,} words loaded) Ask your questions below.")

# ── UI: Q&A ───────────────────────────────────────────────────────────────────
if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("💬 Ask a question about the video")

    question = st.chat_input("e.g. What is the main topic of this video?")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Ollama is thinking…"):
                try:
                    st.write(answer_question(st.session_state["vectorstore"], question))
                except Exception as e:
                    st.error(f"❌ {e}")