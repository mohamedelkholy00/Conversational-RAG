# Conversational-RAG
A Streamlit-based web application that lets you upload PDF documents and have multi-turn conversations about their content, powered by LangChain, Groq (LLaMA 3.3 70B), and ChromaDB.

## ✨ Features

- 📄 **Multi-PDF upload** — drag and drop one or more PDFs and index them instantly
- 🧠 **Conversational memory** — follow-up questions are understood in full context using chat history
- 🔒 **Strict grounding** — two-layer validation ensures answers come *only* from your documents, never from the model's general knowledge
- 🔍 **Not-found feedback** — a clear styled card appears when the topic isn't in your files
- 📚 **Source transparency** — every answer links back to the document chunks that produced it
- 🪟 **Apple Liquid Glass UI** — frosted glass surfaces, animated ambient orbs, and spring-motion bubbles in a cinematic dark mode
- 🗂️ **Session management** — run multiple independent conversation threads by session ID

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| RAG Framework | LangChain (Classic) |
| PDF Parsing | PyPDFLoader |
| Styling | Custom CSS — Apple Liquid Glass Dark |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docchat-ai.git
cd docchat-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📋 Requirements

Create a `requirements.txt` with:

```
streamlit
langchain-classic
langchain-core
langchain-community
langchain-chroma
langchain-groq
langchain-huggingface
langchain-text-splitters
chromadb
huggingface-hub
sentence-transformers
torchvision
pypdf
python-dotenv
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## 🧭 How to Use

1. **Upload PDFs** — use the sidebar file uploader to select one or more PDF files
2. **Process** — click **⚡ Process Documents** to chunk and index them into ChromaDB
3. **Set a Session ID** — optionally name your session to keep threads separate
4. **Ask questions** — type in the input bar and click **➤** to send
5. **View sources** — expand the source chunks under any answer to see exactly where it came from
6. **Clear** — click **🗑️ Clear Conversation** to reset the chat history for the current session

---

## 🔒 Answer Grounding — How It Works

DocChat AI uses two independent layers to prevent hallucination:

**Layer 1 — Retriever pre-check**
Before the LLM is called, the retriever queries ChromaDB. If no relevant chunks are found, the question is immediately flagged as not found and the LLM is never invoked.

**Layer 2 — LLM token check**
The system prompt instructs the model to respond with the exact token `NOT_FOUND_IN_DOCUMENTS` if it cannot find the answer in the provided context. The app detects this token and displays the styled "not found" card instead of showing the raw model output.

---

## 🏗️ Project Structure

```
docchat-ai/
├── app.py               # Main Streamlit application
├── .env                 # API keys (never commit this)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🎨 UI Design

The interface is built on an **Apple Liquid Glass** aesthetic:

- **Ambient orbs** — three animated radial gradients (purple, cyan, pink) breathe behind all surfaces
- **Frosted glass panels** — sidebar, bubbles, and inputs use `backdrop-filter: blur(24–40px) saturate(180–200%)` with semi-transparent fills
- **Spring-motion messages** — chat bubbles animate in with a `cubic-bezier(0.34, 1.56, 0.64, 1)` spring curve
- **Semantic glass tints** — bot bubbles are neutral frost, user bubbles are blue-tinted, not-found cards are red-tinted
- **Typography** — Inter with `-0.4px` letter-spacing and antialiasing, matching Apple's SF Pro feel

---

## ⚠️ Notes

- PDFs are processed in memory via temporary files and deleted immediately after loading — nothing is stored on disk
- The vector store is rebuilt each time you click **Process Documents**
- Chat history lives in Streamlit session state and resets on page refresh or when you click Clear
- Chunk size is set to 500 tokens with 50-token overlap — tune these in `build_rag_chain()` for your use case

---

## 📄 License

MIT — free to use, modify, and distribute.
