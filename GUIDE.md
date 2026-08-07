# Complete Beginner's Guide: RAG Document Chatbot
### Google Colab → GitHub → Streamlit Community Cloud

This guide walks through everything: what RAG means, how to build and test
the pipeline in Google Colab, and how to deploy the finished app using
GitHub and Streamlit Community Cloud. It assumes **zero prior experience**.

---

## 1. What is RAG, in plain language?

- **RAG (Retrieval-Augmented Generation):** instead of asking an AI model to
  answer purely from what it memorized during training, you first *retrieve*
  relevant snippets from your own documents, then *give those snippets to
  the model* so it answers using your real content.
- **Embedding:** a way of turning text into a list of numbers (a vector) that
  captures its *meaning*. Similar sentences end up with similar numbers.
- **Vector database:** a specialized storage system optimized for finding
  which vectors (embeddings) are most similar to a given query vector —
  extremely fast, even with thousands of chunks.
- **Document chunking:** splitting a long document into smaller overlapping
  pieces so each piece is small enough to embed and retrieve precisely,
  while overlap keeps ideas from being awkwardly cut in half.
- **Similarity search:** given a question's embedding, finding the stored
  chunk embeddings that are numerically closest to it (most similar meaning).
- **Retrieval:** the overall step of pulling the top-matching chunks out of
  the vector database to use as context for the answer.
- **Prompt grounding:** explicitly instructing the LLM to base its answer
  only on the retrieved text, not on its own general knowledge.
- **Hallucination:** when an LLM confidently states something that isn't
  true or isn't supported by the given context. Grounding + clear
  instructions reduce (but don't 100% eliminate) this risk.
- **Two different jobs:** the Hugging Face model's *only* job is turning
  text into numeric vectors for search. The Groq LLM's job is *reading*
  the retrieved text and *writing* a natural-language answer. They never
  do each other's job.
- **End-to-end flow:**

```
Uploaded Documents
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Hugging Face Embeddings
        ↓
FAISS Vector Database
        ↓
Relevant Chunk Retrieval
        ↓
Groq LLM
        ↓
Answer with Sources
```

---

## 2. FAISS vs. ChromaDB — and why FAISS was chosen

| Criteria | FAISS | ChromaDB |
|---|---|---|
| Install ease | Very easy (`faiss-cpu`, no native build issues) | Usually easy, but pulls more transitive dependencies |
| Ease of use | Simple, low-level API | Higher-level API, slightly more setup |
| CPU performance | Excellent | Good |
| Memory usage | Low | Slightly higher |
| Persistence | Manual (save/load index file) | Built-in local persistence |
| Dependency conflicts | Rare | More common on constrained cloud environments |
| Colab compatibility | Excellent | Good |
| Streamlit Cloud compatibility | Excellent, very reliable | Occasionally hits build issues on limited free tier |
| Beginner-friendliness | High | Medium |

**Decision: FAISS (`faiss-cpu`)** — it installs cleanly and reliably on both
Google Colab and Streamlit Community Cloud's constrained free environment,
has the fewest dependency conflicts, and its simple API is easy for a
beginner to understand. It also runs entirely locally/in-session, with no
paid service required.

---

## 3. Creating a Groq API Key

1. Go to https://console.groq.com and sign up (free).
2. Once logged in, open the **API Keys** section in the left menu.
3. Click **Create API Key**, give it a name, and copy the key immediately —
   Groq only shows it once.
4. Store it somewhere safe temporarily (like a password manager) — you will
   paste it into Colab Secrets and Streamlit Secrets, never into code.

---

## 4. Phase 1 — Google Colab: Build & Test the Pipeline

Create a new notebook at https://colab.research.google.com. Add the
following cells **in order**, running each one before moving to the next.

### Cell 1 — Install required libraries

```python
!pip install -q streamlit groq sentence-transformers faiss-cpu pypdf numpy
```
*No restart is needed after this install in a fresh Colab runtime.*

### Cell 2 — Import required libraries

```python
import os
import io
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from google.colab import files, userdata
```

### Cell 3 — Add the Groq API key securely

**Option A — Colab Secrets (recommended):**
Click the 🔑 key icon in the left sidebar of Colab, add a new secret named
`GROQ_API_KEY`, paste your key as the value, and toggle "Notebook access" on.

```python
GROQ_API_KEY = userdata.get("GROQ_API_KEY")
print("Key loaded:", GROQ_API_KEY is not None)  # never print the key itself
```

**Option B — temporary environment variable** (only for quick local testing,
re-enter each session, do not save the notebook with the key inside):

```python
import getpass
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
```

### Cell 4 — Define application settings

```python
GROQ_MODEL = "llama-3.3-70b-versatile"   # change here if a newer model is released
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
```

### Cell 5 — Load the Hugging Face embedding model

```python
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")
```
*The first run downloads ~80MB of model weights — this can take 30-60
seconds. Subsequent loads in the same session are instant.*

### Cell 6 — Upload sample PDF or TXT files

```python
uploaded = files.upload()  # opens a file picker in Colab
print("Uploaded:", list(uploaded.keys()))
```

### Cell 7 — Extract text from uploaded files

```python
def extract_pages(uploaded_dict):
    pages = []
    for file_name, file_bytes in uploaded_dict.items():
        ext = file_name.lower().split(".")[-1]
        if ext == "pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"text": text, "source": file_name, "page": i})
        elif ext == "txt":
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            if text:
                pages.append({"text": text, "source": file_name, "page": 1})
    return pages

all_pages = extract_pages(uploaded)
print(f"Extracted {len(all_pages)} page(s) of text.")
```

### Cell 8 — Split extracted text into chunks

```python
def chunk_text(text, chunk_size, chunk_overlap):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks

all_chunks = []
for p in all_pages:
    for i, c in enumerate(chunk_text(p["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
        all_chunks.append({"text": c, "source": p["source"], "page": p["page"], "chunk_id": i})

print(f"Created {len(all_chunks)} chunks.")
print(all_chunks[0] if all_chunks else "No chunks created.")
```

### Cell 9 — Create the vector database

```python
texts = [c["text"] for c in all_chunks]
embeddings = np.asarray(embed_model.encode(texts, show_progress_bar=True), dtype="float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print("FAISS index size:", index.ntotal)
```

### Cell 10 — Create the retrieval function

```python
def retrieve(query, top_k=TOP_K):
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, min(top_k, len(all_chunks)))
    return [all_chunks[i] for i in indices[0] if i != -1]
```

### Cell 11 — Create the Groq answer-generation function

```python
client = Groq(api_key=GROQ_API_KEY)

RAG_SYSTEM_PROMPT = """You are a careful, helpful document assistant.
Answer ONLY using the retrieved context. If the answer is not present,
say: "I could not find this information in the uploaded documents."
Do not invent facts. Mention sources where relevant."""

def ask(question):
    retrieved = retrieve(question)
    context = "\n\n".join(f"[{c['source']} p.{c['page']}] {c['text']}" for c in retrieved)
    prompt = f"Retrieved Context:\n{context}\n\nUser Question:\n{question}\n\nAnswer:"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content, retrieved
```

### Cell 12 — Test the RAG pipeline

```python
answer, sources = ask("What are the library hours during exam week?")
print("ANSWER:\n", answer)
print("\nSOURCES:")
for s in sources:
    print(f"- {s['source']} (page {s['page']})")
```

### Cell 13 — Optional: testing the full Streamlit app from Colab

Colab cannot display a live Streamlit interface directly inside a notebook
cell the way it shows text output. The practical approach:

1. Write your `app.py` to disk using `%%writefile app.py` in a cell.
2. Run `!streamlit run app.py &>/content/log.txt &` in the background.
3. Use a tunneling tool (e.g. `localtunnel` via `!npx localtunnel --port 8501`)
   to get a temporary public URL to click.

This is optional and mainly useful for a quick sanity check — the tunnel
URL is temporary and resets each Colab session. **The simpler and
recommended path** is to fully test the RAG functions (Cells 1-12) inside
Colab first, then deploy the polished `app.py` directly to Streamlit
Community Cloud for real testing (see Phase 3 below).

---

## 5. Local Computer Testing (optional, alongside Colab)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS
source venv/bin/activate
# Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml` locally:
```toml
GROQ_API_KEY = "your-api-key"
```
**Never upload this file to GitHub** — it is already excluded by `.gitignore`.

---

## 6. Phase 2 — Creating the GitHub Repository

1. Go to https://github.com and click **Sign up** (free account).
2. Once logged in, click the **+** icon (top right) → **New repository**.
3. Name it, e.g., `rag-streamlit-chatbot`.
4. Choose **Public** (required for the free Streamlit Community Cloud tier)
   or Private if you have a paid Streamlit plan.
5. Check **Add a README file**, then click **Create repository**.

### Option A — Create files directly on the GitHub website
1. Click **Add file → Create new file**.
2. Type the file name exactly: `app.py`.
3. Paste in the complete `app.py` code.
4. Scroll down, add a short commit message, click **Commit changes**.
5. Repeat for `requirements.txt`, `.gitignore`, and `.streamlit/config.toml`
   (typing `.streamlit/config.toml` as the file name automatically creates
   the folder).

### Option B — Upload files from your computer
1. Click **Add file → Upload files**.
2. Drag and drop `app.py`, `requirements.txt`, `.gitignore`, `README.md`,
   and the `.streamlit` folder (or its `config.toml` file).
3. Add a commit message and click **Commit changes**.

### Editing a file later
1. Open the file in your repository.
2. Click the pencil (✏️) icon.
3. Make your changes.
4. Scroll down, add a commit message, click **Commit changes**.

### Confirm the API key isn't in the repository
Open `app.py` and `requirements.txt` on GitHub and visually confirm no key
is present. Only `st.secrets` / `os.getenv` calls should appear — never a
real key string.

---

## 7. Phase 3 — Streamlit Community Cloud Deployment

1. Go to https://streamlit.io and click **Sign in** (or **Community Cloud**).
2. Choose **Continue with GitHub** and authorize Streamlit to access your
   repositories.
3. Click **Create app** (sometimes labeled **New app**).
4. Select **"Deploy a public app from GitHub"**.
5. Choose your repository (e.g. `yourname/rag-streamlit-chatbot`).
6. Choose the branch (usually `main`).
7. Set the main file path to `app.py`.
8. Optionally customize the app URL (subdomain) if the field is available.
9. Click **Advanced settings** before deploying.
10. In the **Secrets** box, paste:
    ```toml
    GROQ_API_KEY = "your-real-groq-api-key"
    ```
11. Click **Save**, then click **Deploy**.
12. Watch the build log while it installs `requirements.txt` — this can
    take a few minutes the first time.
13. Once deployed, open the app's URL.
14. Upload `sample_test_document.txt` (included in this project) and click
    **Process Documents**.
15. Ask a test question and confirm the **Sources** expander shows the
    correct file name and page.

### Managing the app afterward
- **View logs:** open the app → **Manage app** (bottom right) → **Logs**.
- **Reboot:** **Manage app → Reboot app** (clears session state/cache).
- **Redeploy after a GitHub update:** pushing a new commit to the connected
  branch triggers an automatic redeploy; you can also click **Reboot app**.
- **Delete/stop:** **Manage app → Delete app** (or **Down** to pause it).

---

## 8. Testing Requirements

Use `sample_test_document.txt` (included) and try these five questions:

1. **Direct answer, one chunk:** *"What are the library hours on weekends?"*
   → Expect: "10:00 AM to 6:00 PM," citing the handbook, Library Hours page.
2. **Multiple chunks:** *"What does it cost to attend, and what's the
   technology fee?"* → Expect tuition figures + the $250 fee, possibly
   citing two chunks from the Tuition section.
3. **Not in the document:** *"What is the university's football team
   record this year?"* → Expect: "I could not find this information in
   the uploaded documents."
4. **Follow-up question:** First ask *"What GPA is required for the Merit
   Scholarship?"*, then ask *"And what about to renew it?"* → Expect the
   renewal GPA (3.3), using conversation history to know "it" = the
   scholarship.
5. **Source citation check:** *"How many students attend BrightLeaf?"* →
   Expect ~8,200, with the source shown as the sample document, page 1.

---

## 9. Security & Privacy Notes

- Uploaded documents may contain sensitive information — only the
  retrieved chunks (not entire files) are sent to the Groq API per question.
- API keys must live only in Streamlit Secrets or environment variables —
  never in code, notebooks, or `README.md`.
- Public Streamlit Community Cloud apps are accessible to anyone with the
  URL — don't deploy sensitive company/personal documents on a public app
  without adding authentication.
- Application logs should never contain the API key or full private
  document text.
- This project is intended for learning/demonstration; production use
  would need authentication, persistent secure storage, and access controls.

## 10. Streamlit Community Cloud Limitations

- Limited free CPU/RAM; large documents or many concurrent users may be slow
- Apps "sleep" after inactivity and need to "wake up" (cold start delay)
- No permanent file storage — uploaded documents and the FAISS index reset
  on reboot/restart
- Session data (chat history) is lost when the app restarts
- First embedding-model load after a cold start adds a short delay
- File upload size is capped (see `.streamlit/config.toml`)
- Groq API has its own rate limits, independent of Streamlit

---

## 11. Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | A package used in code isn't in `requirements.txt` | Add the exact package name to `requirements.txt` and redeploy |
| Groq authentication error | Wrong/expired API key | Regenerate the key in the Groq console and update Secrets |
| Missing `GROQ_API_KEY` | Secret not set | Colab: add via Secrets panel. Streamlit: Manage app → Secrets |
| Deprecated Groq model | Model retired | Update `GROQ_MODEL` in `app.py` to a current model from Groq's docs |
| FAISS install failure | Rare platform mismatch | Ensure `faiss-cpu` (not `faiss-gpu`) is in `requirements.txt` |
| ChromaDB dependency conflict | Heavier dependency tree | Switch to FAISS (already used in this project) |
| Sentence-transformer download error | No internet access during build, or low memory | Retry deploy; ensure the Streamlit Cloud outbound network isn't blocked |
| Blank page on Streamlit Cloud | An uncaught exception on startup | Open **Manage app → Logs** to see the real error |
| App keeps rebooting | Out-of-memory from a large model/index | Reduce chunk count, use the small MiniLM model (already default) |
| Works in Colab, fails on Cloud | Version or path differences | Confirm `requirements.txt` versions match what you tested, avoid absolute local paths |
| `requirements.txt` build failure | An incompatible pinned version | Loosen the version pin (e.g. `>=x,<y`) and redeploy |
| Stuck on "Installing dependencies" | Large/incompatible dependency resolution | Check logs; remove unused/heavy packages |
| "Module not found" after deploy | Package missing from `requirements.txt` | Add it, commit, and it will auto-redeploy |
| PDF uploads but no text extracted | Scanned/image-only PDF | Expected — app shows a warning; OCR would be needed to support this |
| Chatbot invents answers | Prompt not strict enough / too much retrieved noise | Strengthen the system prompt wording, lower `TOP_K`, increase chunk quality |
| Poor retrieval quality | Chunking or `TOP_K` not tuned | Adjust chunk size/overlap, increase `TOP_K`, check source document quality |
| Sources missing | Metadata not attached to chunks | Confirm every chunk keeps `source`/`page` fields through chunking and search |
| Chat history disappears | App restarted / session ended | Expected behavior — session state doesn't persist across restarts |
| One user sees another's data | Global variables/caching holding user data | Only cache the model itself (`st.cache_resource`); keep documents/chat in `st.session_state` (already done in `app.py`) |

---

## 12. Suggested Future Improvements

- Add OCR (e.g. `pytesseract`) for scanned PDFs
- Persist the FAISS index to disk (or a hosted vector DB) between sessions
- Add simple authentication for private deployments
- Support DOCX, CSV, and HTML uploads
- Stream the Groq response token-by-token for a faster feel
- Add a feedback button ("was this answer helpful?") to track quality
