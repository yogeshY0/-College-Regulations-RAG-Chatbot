import streamlit as st
import requests
from rag_engine import RAGPipeline

# ─────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="College RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');
    :root {
        --navy: #0f1b35;
        --gold: #c9a227;
        --gold-light: #f0d98a;
        --cream: #faf6ee;
        --text: #1a1a2e;
        --border: #e5ddd0;
        --source-bg: #f8f4ec;
    }
    .stApp { background: var(--cream); font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: var(--navy) !important; border-right: 3px solid var(--gold); }
    [data-testid="stSidebar"] * { color: #e8e0d0 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--gold) !important;
        font-family: 'Crimson Pro', serif !important;
    }
    .main-header {
        background: linear-gradient(135deg, var(--navy) 0%, #1a2e5a 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 2px solid var(--gold);
    }
    .main-header h1 {
        font-family: 'Crimson Pro', serif; font-size: 2.2rem;
        color: var(--gold-light) !important; margin: 0;
    }
    .main-header p { color: #a8b4c8; margin: 0.4rem 0 0 0; font-size: 0.95rem; }
    .main-header .badge {
        display: inline-block; background: rgba(201,162,39,0.2);
        border: 1px solid var(--gold); color: var(--gold-light);
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; margin-top: 0.5rem;
    }
    .user-bubble {
        background: var(--navy); color: #e8e0d0; padding: 1rem 1.3rem;
        border-radius: 14px 14px 4px 14px; margin: 0.5rem 0;
        max-width: 80%; margin-left: auto; font-size: 0.95rem;
    }
    .assistant-bubble {
        background: white; color: var(--text); padding: 1.2rem 1.5rem;
        border-radius: 14px 14px 14px 4px; margin: 0.5rem 0;
        max-width: 85%; font-size: 0.95rem;
        border-left: 4px solid var(--gold); line-height: 1.7;
    }
    .source-item {
        background: white; border: 1px solid var(--border);
        border-radius: 6px; padding: 0.5rem 0.8rem;
        margin: 0.3rem 0; font-size: 0.82rem;
    }
    .sim-score {
        display: inline-block; background: var(--navy); color: var(--gold-light);
        padding: 0.1rem 0.5rem; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; float: right;
    }
    .pipeline-step {
        background: rgba(201,162,39,0.1); border: 1px solid rgba(201,162,39,0.3);
        border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.3rem 0;
        font-size: 0.82rem; color: #e8e0d0;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD RAG PIPELINE
# ─────────────────────────────────────────────
@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

rag = load_rag_pipeline()


# ─────────────────────────────────────────────
#  OLLAMA HEALTH CHECK
# ─────────────────────────────────────────────
def check_ollama_health() -> tuple[bool, str]:
    """
    Check if Ollama is running and model is available.
    Returns: (is_healthy, message)
    """
    try:
        # Try to reach Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False, f"Ollama API error: {response.status_code}"

        data = response.json()
        models = data.get("models", [])

        if not models:
            return False, "❌ No models installed. Run: ollama pull llama3.2"

        # Check if llama3.2 is available
        model_names = [m.get("name", "").split(":")[0] for m in models]
        has_llama = any("llama" in name for name in model_names)

        if has_llama:
            return True, f"✅ Ollama ready with {len(models)} model(s)"
        else:
            return False, f"❌ llama3.2 not found. Models available: {', '.join(model_names)}. Run: ollama pull llama3.2"

    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to Ollama. Run: ollama serve"
    except Exception as e:
        return False, f"❌ Error checking Ollama: {str(e)}"


def call_ollama(augmented_prompt: str) -> str:
    """
    Sends the augmented prompt to Ollama running locally on your Mac.

    HOW THIS WORKS:
    ───────────────
    When you run 'ollama serve' in the terminal, it starts a small
    web server on your Mac at http://localhost:11434
    We send our prompt to that address and get the answer back.

    No internet needed. No API key. Completely free.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",  # Ollama local address
            json={
                "model":  "llama3.2",   # model you downloaded with: ollama pull llama3.2
                "prompt": augmented_prompt,
                "stream": False,        # wait for the full answer at once
                "options": {
                    "temperature": 0.3, # low = more factual (good for regulations)
                    "num_predict": 400  # max length of response
                }
            },
            timeout=120  # local models can be slow, wait up to 2 minutes
        )

        # Check if response is successful
        if response.status_code != 200:
            return f"⚠️ Ollama returned status {response.status_code}: {response.text[:200]}"

        data = response.json()

        # Extract response text
        answer = data.get("response", "").strip()

        if not answer:
            # Response was empty or missing
            return (
                "⚠️ Ollama returned empty response. "
                "Check that llama3.2 model is installed with: ollama pull llama3.2"
            )

        return answer

    except requests.exceptions.ConnectionError:
        # This error means ollama serve is not running
        return (
            "⚠️ Cannot connect to Ollama. "
            "Make sure Ollama is running by opening a terminal and executing: ollama serve"
        )
    except requests.exceptions.Timeout:
        return (
            "⚠️ Ollama request timed out (>120 seconds). "
            "The model may be taking too long. Try with a shorter prompt or restart Ollama."
        )
    except Exception as e:
        return f"⚠️ Ollama error: {type(e).__name__}: {str(e)}"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ College RAG Bot")
    st.markdown("---")
    st.markdown("### 🔧 RAG Architecture")

    steps = [
        ("1. 📄 Ingest",   "Load JSON regulations (15 QA pairs)"),
        ("2. 🔢 Embed",    "Sentence-Transformers → 384-dim vectors"),
        ("3. 🔍 Retrieve", "Cosine similarity → Top-3 docs"),
        ("4. 📝 Augment",  "Inject context into prompt"),
        ("5. 🤖 Generate", "Ollama llama3.2 → Grounded answer"),
    ]
    for title, desc in steps:
        st.markdown(
            f'<div class="pipeline-step"><strong>{title}</strong><br>'
            f'<span style="opacity:0.75;font-size:0.78rem">{desc}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.markdown(f"**Documents:** {len(rag.documents)}")
    st.markdown(f"**Embedding dim:** {rag.embeddings.shape[1]}")
    st.markdown(f"**Embedding model:** `all-MiniLM-L6-v2`")
    st.markdown(f"**LLM:** `llama3.2` via Ollama")

    # Add Ollama health check
    st.markdown("---")
    st.markdown("### 🔌 Ollama Status")
    is_healthy, health_msg = check_ollama_health()
    if is_healthy:
        st.markdown(f"<span style='color: #2ecc71;'>{health_msg}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color: #e74c3c;'>{health_msg}</span>", unsafe_allow_html=True)
        st.warning("⚠️ Ollama must be running for the chatbot to work!")
        st.code("ollama serve", language="bash")

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is the attendance requirement?",
        "How do I apply for a scholarship?",
        "What happens if I fail an exam?",
        "How can I borrow library books?",
        "When do I need to pay fees?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
            st.session_state.pending_question = q

    st.markdown("---")
    st.markdown("### 🛠️ Prompt Engineering")
    st.markdown("""
**Techniques used:**
- ✅ **Role Assignment** — LLM acts as academic advisor
- ✅ **Context Injection** — Retrieved docs pasted before query
- ✅ **Chain-of-Thought** — Instructions guide reasoning
- ✅ **Output Constraints** — Concise, cite sources, admit ignorance
    """)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_history = []
        st.rerun()


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 College Regulations Assistant</h1>
    <p>Ask anything about attendance, exams, fees, scholarships, library rules, and more.</p>
    <span class="badge">RAG + llama3.2 (Ollama) · all-MiniLM-L6-v2 embeddings · Cosine Similarity Retrieval</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_history" not in st.session_state:
    st.session_state.source_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ─────────────────────────────────────────────
#  SHOW EXISTING CHAT HISTORY
# ─────────────────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑‍🎓 {message["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">{message["content"]}</div>',
            unsafe_allow_html=True
        )
        # Show sources under each past assistant message
        turn = i // 2
        if turn < len(st.session_state.source_history):
            sources = st.session_state.source_history[turn]
            if sources:
                with st.expander(f"📚 Sources used ({len(sources)} documents)", expanded=False):
                    for j, src in enumerate(sources):
                        st.markdown(
                            f'<div class="source-item">'
                            f'<strong>Source {j+1}:</strong> {src["question"]}'
                            f'<span class="sim-score">{src["similarity_score"]:.3f}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )


# ─────────────────────────────────────────────
#  HANDLE NEW INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask about college regulations, policies, fees, exams...")

# Sidebar sample question clicked
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:

    # Show user message
    st.markdown(
        f'<div class="user-bubble">🧑‍🎓 {user_input}</div>',
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── Step 1+2+3: Retrieve relevant documents ──
    with st.spinner("🔍 Searching regulations database..."):
        augmented_prompt, retrieved_docs = rag.query(user_input, top_k=3)

    # ── Step 4: Generate answer using Ollama ──
    with st.spinner("🤖 llama3.2 is thinking... (may take 10-30 seconds)"):
        answer = call_ollama(augmented_prompt)

    # Show answer
    st.markdown(
        f'<div class="assistant-bubble">{answer}</div>',
        unsafe_allow_html=True
    )

    # Show which sources were used
    if retrieved_docs:
        with st.expander(f"📚 Sources used ({len(retrieved_docs)} documents)", expanded=True):
            for j, src in enumerate(retrieved_docs):
                st.markdown(
                    f'<div class="source-item">'
                    f'<strong>Source {j+1}:</strong> {src["question"]}'
                    f'<span class="sim-score">{src["similarity_score"]:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_history.append(retrieved_docs)