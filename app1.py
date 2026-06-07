import streamlit as st
import os
import tempfile
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Apple Liquid Glass Dark Mode ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Design tokens ── */
:root {
    /* Background layers */
    --bg-base:          #080810;
    --bg-orb-1:         rgba(100, 60, 255, 0.28);
    --bg-orb-2:         rgba(0, 180, 255, 0.18);
    --bg-orb-3:         rgba(255, 60, 140, 0.12);

    /* Glass surfaces */
    --glass-bg:         rgba(255, 255, 255, 0.055);
    --glass-bg-hover:   rgba(255, 255, 255, 0.085);
    --glass-bg-strong:  rgba(255, 255, 255, 0.09);
    --glass-border:     rgba(255, 255, 255, 0.12);
    --glass-border-hi:  rgba(255, 255, 255, 0.22);
    --glass-shine:      rgba(255, 255, 255, 0.06);

    /* Accent — iOS blue */
    --accent:           #3b82f6;
    --accent-soft:      rgba(59, 130, 246, 0.25);
    --accent-glow:      rgba(59, 130, 246, 0.4);
    --accent-2:         #8b5cf6;

    /* Semantic */
    --success:          #30d158;
    --success-soft:     rgba(48, 209, 88, 0.2);
    --danger:           #ff453a;
    --danger-soft:      rgba(255, 69, 58, 0.15);
    --warn:             #ffd60a;

    /* Text */
    --text-primary:     rgba(255, 255, 255, 0.92);
    --text-secondary:   rgba(255, 255, 255, 0.60);
    --text-tertiary:    rgba(255, 255, 255, 0.36);

    /* Geometry */
    --radius-xl:        22px;
    --radius-lg:        16px;
    --radius-md:        12px;
    --radius-sm:        8px;
    --radius-pill:      100px;

    /* Blur values */
    --blur-glass:       blur(24px) saturate(180%);
    --blur-heavy:       blur(40px) saturate(200%);
}

/* ══════════════════════════════════════════════
   GLOBAL BASE
══════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Ambient orb background */
body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 70% 55% at 15% 20%, var(--bg-orb-1), transparent),
        radial-gradient(ellipse 55% 45% at 85% 75%, var(--bg-orb-2), transparent),
        radial-gradient(ellipse 45% 40% at 50% 50%, var(--bg-orb-3), transparent),
        var(--bg-base);
    animation: ambientShift 18s ease-in-out infinite alternate;
}
@keyframes ambientShift {
    0%   { filter: hue-rotate(0deg)   brightness(1);    }
    50%  { filter: hue-rotate(12deg)  brightness(1.05); }
    100% { filter: hue-rotate(-8deg)  brightness(0.97); }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; position: relative; z-index: 1; }

/* ══════════════════════════════════════════════
   SCROLLBARS
══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.15);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.28); }

/* ══════════════════════════════════════════════
   SIDEBAR — liquid glass panel
══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: rgba(12, 12, 22, 0.75) !important;
    backdrop-filter: var(--blur-heavy) !important;
    -webkit-backdrop-filter: var(--blur-heavy) !important;
    border-right: 1px solid var(--glass-border) !important;
    position: relative; z-index: 10;
}
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(160deg, rgba(255,255,255,0.04) 0%, transparent 60%);
    border-radius: inherit;
}
section[data-testid="stSidebar"] > div {
    padding: 1.6rem 1.25rem !important;
    position: relative; z-index: 1;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    display: flex; align-items: center; gap: 11px;
    margin-bottom: 1.8rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--glass-border);
}
.sidebar-logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 19px;
    box-shadow: 0 4px 16px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.2);
}
.sidebar-logo-text {
    font-family: 'Inter', sans-serif; font-weight: 600;
    font-size: 1.05rem; letter-spacing: -0.4px; color: var(--text-primary);
}
.sidebar-logo-text span {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Sidebar section label ── */
.sidebar-section-title {
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text-tertiary); margin: 1.5rem 0 0.55rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--glass-bg) !important;
    backdrop-filter: var(--blur-glass) !important;
    -webkit-backdrop-filter: var(--blur-glass) !important;
    border: 1px dashed var(--glass-border-hi) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.4rem !important;
    transition: border-color 0.25s, background 0.25s !important;
}
[data-testid="stFileUploader"]:hover {
    background: var(--glass-bg-hover) !important;
    border-color: rgba(59,130,246,0.5) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input, .stTextInput input {
    background: var(--glass-bg) !important;
    backdrop-filter: var(--blur-glass) !important;
    -webkit-backdrop-filter: var(--blur-glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
    caret-color: var(--accent);
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-tertiary) !important; }
[data-testid="stTextInput"] input:focus {
    background: var(--glass-bg-strong) !important;
    border-color: rgba(59,130,246,0.6) !important;
    box-shadow: 0 0 0 3px var(--accent-soft), 0 0 20px rgba(59,130,246,0.15) !important;
    outline: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--glass-bg-strong) !important;
    backdrop-filter: var(--blur-glass) !important;
    -webkit-backdrop-filter: var(--blur-glass) !important;
    border: 1px solid var(--glass-border-hi) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 0.55rem 1.1rem !important;
    width: 100% !important;
    letter-spacing: -0.01em;
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.08) inset, 0 2px 8px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    background: var(--glass-bg-hover) !important;
    border-color: rgba(255,255,255,0.22) !important;
    transform: translateY(-1px) scale(1.01) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.1) inset, 0 6px 20px rgba(0,0,0,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset !important;
}

/* ── PDF chip ── */
.pdf-chip {
    display: flex; align-items: center; gap: 8px;
    background: var(--glass-bg);
    backdrop-filter: var(--blur-glass);
    -webkit-backdrop-filter: var(--blur-glass);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 0.42rem 0.75rem;
    font-size: 0.77rem; color: var(--text-secondary);
    margin-bottom: 0.38rem;
    transition: background 0.2s;
}
.pdf-chip:hover { background: var(--glass-bg-hover); }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--glass-border) !important;
    margin: 1rem 0 !important;
}

/* ══════════════════════════════════════════════
   TOP BAR
══════════════════════════════════════════════ */
.topbar {
    background: rgba(10, 10, 20, 0.6);
    backdrop-filter: var(--blur-heavy);
    -webkit-backdrop-filter: var(--blur-heavy);
    border-bottom: 1px solid var(--glass-border);
    padding: 0.9rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    position: relative; z-index: 5;
}
.topbar::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
}
.topbar-title {
    font-family: 'Inter', sans-serif; font-size: 0.95rem;
    font-weight: 600; letter-spacing: -0.3px; color: var(--text-primary);
}
.topbar-badge {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-pill);
    padding: 0.28rem 0.85rem;
    font-size: 0.72rem; font-weight: 500;
    color: var(--text-tertiary);
    letter-spacing: 0.01em;
}
.topbar-badge.active {
    border-color: rgba(48, 209, 88, 0.35);
    color: var(--success);
    background: var(--success-soft);
    box-shadow: 0 0 12px rgba(48,209,88,0.12);
}

/* ══════════════════════════════════════════════
   WELCOME SCREEN
══════════════════════════════════════════════ */
.welcome-screen {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center;
    gap: 0.85rem; padding: 4rem 2rem;
}
.welcome-icon {
    width: 72px; height: 72px;
    background: linear-gradient(145deg, rgba(59,130,246,0.3), rgba(139,92,246,0.25));
    backdrop-filter: var(--blur-glass);
    -webkit-backdrop-filter: var(--blur-glass);
    border: 1px solid var(--glass-border-hi);
    border-radius: 22px;
    display: flex; align-items: center; justify-content: center;
    font-size: 34px; margin-bottom: 0.6rem;
    box-shadow: 0 8px 32px rgba(59,130,246,0.2), 0 1px 0 rgba(255,255,255,0.1) inset;
    animation: iconFloat 4s ease-in-out infinite;
}
@keyframes iconFloat {
    0%, 100% { transform: translateY(0);    }
    50%       { transform: translateY(-5px); }
}
.welcome-title {
    font-family: 'Inter', sans-serif; font-size: 1.55rem;
    font-weight: 600; letter-spacing: -0.6px; color: var(--text-primary);
}
.welcome-subtitle {
    font-size: 0.9rem; color: var(--text-secondary);
    max-width: 360px; line-height: 1.65; font-weight: 400;
}
.welcome-steps {
    display: flex; gap: 0.75rem; margin-top: 1rem;
    flex-wrap: wrap; justify-content: center;
}
.welcome-step {
    background: var(--glass-bg);
    backdrop-filter: var(--blur-glass);
    -webkit-backdrop-filter: var(--blur-glass);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 0.75rem 1rem;
    font-size: 0.8rem; color: var(--text-secondary);
    display: flex; align-items: center; gap: 0.55rem; min-width: 145px;
    transition: background 0.2s, transform 0.2s;
}
.welcome-step:hover {
    background: var(--glass-bg-hover);
    transform: translateY(-2px);
}
.step-num {
    width: 20px; height: 20px;
    background: var(--accent-soft);
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 50%;
    color: var(--accent); font-size: 0.68rem; font-weight: 600;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}

/* ══════════════════════════════════════════════
   MESSAGE BUBBLES
══════════════════════════════════════════════ */
.message-row {
    display: flex; align-items: flex-end; gap: 0.65rem;
    max-width: 800px; animation: msgIn 0.3s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes msgIn {
    from { opacity: 0; transform: translateY(10px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)    scale(1);    }
}
.message-row.user { flex-direction: row-reverse; align-self: flex-end; }
.message-row.bot  { align-self: flex-start; }

.avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; margin-bottom: 2px;
}
.avatar.bot {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 2px 10px var(--accent-glow);
    border: 1px solid rgba(255,255,255,0.15);
}
.avatar.user {
    background: var(--glass-bg-strong);
    border: 1px solid var(--glass-border-hi);
}

/* Base bubble */
.bubble {
    padding: 0.78rem 1rem;
    line-height: 1.6; font-size: 0.9rem; font-weight: 400;
    max-width: 660px; word-wrap: break-word; word-break: break-word;
    position: relative;
}

/* Bot bubble — frosted glass */
.bubble.bot {
    background: var(--glass-bg);
    backdrop-filter: var(--blur-glass);
    -webkit-backdrop-filter: var(--blur-glass);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
    color: var(--text-primary);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25), inset 0 1px 0 var(--glass-shine);
}
.bubble.bot::before {
    content: '';
    position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, transparent 55%);
}

/* User bubble — tinted glass */
.bubble.user {
    background: linear-gradient(135deg,
        rgba(59,130,246,0.22),
        rgba(139,92,246,0.16));
    backdrop-filter: var(--blur-glass);
    -webkit-backdrop-filter: var(--blur-glass);
    border: 1px solid rgba(59,130,246,0.28);
    border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
    color: var(--text-primary);
    box-shadow: 0 4px 24px rgba(59,130,246,0.12), inset 0 1px 0 rgba(255,255,255,0.08);
}

.message-meta {
    font-size: 0.67rem; font-weight: 400;
    color: var(--text-tertiary); margin-top: 0.28rem;
    padding: 0 0.15rem; letter-spacing: 0.01em;
}
.message-row.user .message-meta { text-align: right; }

/* ── Source pills ── */
.source-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: var(--radius-pill);
    padding: 0.18rem 0.6rem;
    font-size: 0.69rem; font-weight: 500;
    color: rgba(59,130,246,0.85);
    margin: 0.2rem 0.12rem 0;
}
.source-header {
    font-size: 0.71rem; font-weight: 500;
    color: var(--text-tertiary); margin-top: 0.55rem;
    letter-spacing: 0.02em;
}

/* ── Not-found bubble ── */
.bubble.not-found {
    background: var(--danger-soft);
    backdrop-filter: var(--blur-glass);
    -webkit-backdrop-filter: var(--blur-glass);
    border: 1px solid rgba(255,69,58,0.25);
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
    box-shadow: 0 4px 24px rgba(255,69,58,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
}
.not-found-icon { font-size: 1.05rem; margin-bottom: 0.3rem; display: block; }
.not-found-title {
    font-weight: 600; font-size: 0.86rem;
    color: rgba(255,100,90,0.95); margin-bottom: 0.22rem;
    letter-spacing: -0.1px;
}
.not-found-body { font-size: 0.84rem; color: var(--text-secondary); line-height: 1.55; }

/* ══════════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--glass-bg) !important;
    backdrop-filter: var(--blur-glass) !important;
    -webkit-backdrop-filter: var(--blur-glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 0.79rem !important; font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text-primary) !important; }

/* ══════════════════════════════════════════════
   SPINNER
══════════════════════════════════════════════ */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ══════════════════════════════════════════════
   STATUS DOT
══════════════════════════════════════════════ */
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success);
    display: inline-block; margin-right: 5px;
    box-shadow: 0 0 8px var(--success);
    animation: livePulse 2.4s ease-in-out infinite;
}
@keyframes livePulse {
    0%, 100% { opacity: 1;   transform: scale(1);    }
    50%       { opacity: 0.5; transform: scale(0.85); }
}

/* ══════════════════════════════════════════════
   SUCCESS / INFO ALERTS
══════════════════════════════════════════════ */
[data-testid="stAlert"] {
    background: var(--glass-bg) !important;
    backdrop-filter: var(--blur-glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ──────────────────────────────────────────────────────────
for key, default in {
    "store": {},
    "chat_messages": [],
    "rag_chain": None,
    "retriever": None,           # FIX: store retriever for relevance pre-check
    "uploaded_names": [],
    "last_submitted": "",        # FIX: tracks the last question actually sent
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── LLM + Embeddings ────────────────────────────────────────────────────────────
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

# ── Helper: session history ─────────────────────────────────────────────────────
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ── Helper: build RAG chain ─────────────────────────────────────────────────────
def build_rag_chain(uploaded_files):
    documents, names = [], []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            docs = PyPDFLoader(tmp_path).load()
            documents.extend(docs)
            names.append(uploaded_file.name)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    splits = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # search_kwargs k=6 gives more candidates for relevance check
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Contextualise question using history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference context in the "
         "chat history, formulate a standalone question that can be understood without the chat "
         "history. Do NOT answer the question — only reformulate it if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # ── Strict system prompt — answers ONLY from document context ──────────
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a document assistant. Your ONLY knowledge source is the retrieved document "
         "context provided below. Follow these rules strictly:\n"
         "1. Answer ONLY using information explicitly present in the context.\n"
         "2. If the answer is not found in the context, respond with EXACTLY this token and nothing else: "
         "NOT_FOUND_IN_DOCUMENTS\n"
         "3. Do NOT use any external knowledge, assumptions, or general information.\n"
         "4. Do NOT repeat yourself. Give a single, clear, concise answer.\n"
         "5. Never fabricate or infer information beyond what is explicitly stated.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(llm, qa_prompt),
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ), retriever, names


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">💬</div>
        <div class="sidebar-logo-text">Doc<span>Chat</span> AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">📂 Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type="pdf",
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Process Documents"):
            with st.spinner("Indexing your documents…"):
                chain, retriever, names = build_rag_chain(uploaded_files)
                st.session_state.rag_chain = chain
                st.session_state.retriever = retriever
                st.session_state.uploaded_names = names
                st.session_state.chat_messages = []
                st.session_state.store = {}
                st.session_state.last_submitted = ""
            st.success(f"✅ {len(names)} document(s) ready!")

    if st.session_state.uploaded_names:
        st.markdown('<div class="sidebar-section-title">Loaded Files</div>', unsafe_allow_html=True)
        for name in st.session_state.uploaded_names:
            st.markdown(
                f'<div class="pdf-chip"><span>📄</span>{name}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sidebar-section-title">⚙️ Session</div>', unsafe_allow_html=True)
    session_id = st.text_input(
        "Session ID", value="default_session",
        label_visibility="collapsed", placeholder="Session ID…",
    )

    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_messages = []
        st.session_state.last_submitted = ""
        st.session_state.store.pop(session_id, None)
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.72rem;color:var(--text-muted);line-height:1.6;">'
        'Powered by <b>LLaMA 3.3 · 70B</b><br>Embeddings: all-MiniLM-L6-v2<br>Vector DB: ChromaDB'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
is_ready = st.session_state.rag_chain is not None

status_html = (
    '<span class="status-dot"></span>Ready to chat'
    if is_ready else
    'Upload &amp; process a PDF to begin'
)
st.markdown(f"""
<div class="topbar">
    <div class="topbar-title">DocChat AI</div>
    <div class="topbar-badge {'active' if is_ready else ''}">{status_html}</div>
</div>
""", unsafe_allow_html=True)

# ── Render existing messages ──
with st.container():
    if not st.session_state.chat_messages:
        st.markdown("""
        <div class="welcome-screen">
            <div class="welcome-icon">💬</div>
            <div class="welcome-title">Ask anything about your documents</div>
            <div class="welcome-subtitle">
                Upload PDFs from the sidebar, process them, then start asking questions.
            </div>
            <div class="welcome-steps">
                <div class="welcome-step"><span class="step-num">1</span> Upload PDFs</div>
                <div class="welcome-step"><span class="step-num">2</span> Click Process</div>
                <div class="welcome-step"><span class="step-num">3</span> Start chatting</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_messages:
            role    = msg["role"]
            content = msg["content"]
            sources = msg.get("sources", [])

            if role == "user":
                st.markdown(f"""
                <div class="message-row user">
                    <div>
                        <div class="bubble user">{content}</div>
                        <div class="message-meta">You</div>
                    </div>
                    <div class="avatar user">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                is_not_found = msg.get("not_found", False)

                if is_not_found:
                    st.markdown(f"""
                    <div class="message-row bot">
                        <div class="avatar bot">🤖</div>
                        <div>
                            <div class="bubble not-found">
                                <span class="not-found-icon">🔍</span>
                                <div class="not-found-title">Not found in your documents</div>
                                <div class="not-found-body">
                                    The topic <b>"{msg.get('question','')}"</b> doesn't appear to be covered
                                    in the uploaded files. Try rephrasing, or upload a document that includes this topic.
                                </div>
                            </div>
                            <div class="message-meta">DocChat AI</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    pills = "".join(
                        f'<span class="source-pill">📄 Chunk {i+1}</span>'
                        for i in range(min(len(sources), 4))
                    ) if sources else ""
                    sources_html = f'<div class="source-header">Sources: {pills}</div>' if pills else ""

                    st.markdown(f"""
                    <div class="message-row bot">
                        <div class="avatar bot">🤖</div>
                        <div>
                            <div class="bubble bot">{content}{sources_html}</div>
                            <div class="message-meta">DocChat AI</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if sources:
                        with st.expander(f"📚 View {len(sources)} source chunk(s)"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"**Chunk {i+1}**")
                                st.markdown(
                                    f"```\n{doc.page_content[:600]}"
                                    f"{'…' if len(doc.page_content) > 600 else ''}\n```"
                                )

# ── Input bar ──
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([10, 1])
    with col1:
        # FIX: use on_change + a dedicated key so we can detect a *new* submission
        user_input = st.text_input(
            "Message",
            key="user_input_field",
            placeholder="Ask a question about your documents…",
            label_visibility="collapsed",
            disabled=not is_ready,
        )
    with col2:
        send_clicked = st.button("➤", disabled=not is_ready, use_container_width=True)

# ── FIX: only process if this is genuinely a new, non-empty submission ──────────
is_new_submission = (
    is_ready
    and user_input                                        # not empty
    and user_input.strip()                               # not whitespace
    and user_input != st.session_state.last_submitted    # not already processed
    and send_clicked                                     # user explicitly pressed Send
)

if is_new_submission:
    question = user_input.strip()
    st.session_state.last_submitted = question
    st.session_state.chat_messages.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        # ── Layer 1: check if retriever finds anything relevant at all ──
        retrieved_docs = st.session_state.retriever.invoke(question)
        context_text = " ".join(d.page_content for d in retrieved_docs).strip()
        not_found = False

        if not context_text:
            # No chunks retrieved at all
            not_found = True
            answer = ""
            sources = []
        else:
            # ── Layer 2: call the LLM — catch the NOT_FOUND token ──
            response = st.session_state.rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )
            answer  = response.get("answer", "NOT_FOUND_IN_DOCUMENTS").strip()
            sources = response.get("context", [])

            if "NOT_FOUND_IN_DOCUMENTS" in answer or not answer:
                not_found = True
                answer = ""
                sources = []

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "not_found": not_found,
        "question": question,
    })
    st.rerun()