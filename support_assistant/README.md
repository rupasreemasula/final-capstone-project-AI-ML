# Module 3 — Support Assistant (`/support_assistant`)

A small, complete GenAI service for Zepto: a document corpus embedded and indexed in ChromaDB,
a LangGraph-orchestrated flow that routes each query and retrieves grounded context, a
Pydantic-validated structured output, and a FastAPI wrapper. **Graded baseline: `MOCK_LLM`
unset (or `=1`)** — the entire pipeline runs fully offline and deterministically, with no
signup, API key, or network call to any LLM provider. `MOCK_LLM=0` (+ a Groq API key) is an
optional, ungraded extension layered on top.

## How to run

```bash
cd support_assistant
python ingest.py                                  # Task 1: embed the 8 docs into ChromaDB (one-time; idempotent)
uvicorn main:app --host 0.0.0.0 --port 7860        # Task 6: serve POST /ask (MOCK_LLM defaults to mock mode)
```

(`main.py`'s FastAPI `lifespan` handler also calls `ingest()` automatically on startup, so
`python ingest.py` is a convenience/first-run step, not a hard prerequisite — running
`uvicorn` alone is enough.)

Try it:

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "How much does standard delivery cost?"}'
```

### Docker (required, graded baseline for containerization)

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

Serves the same `POST /ask` endpoint at `http://localhost:7860/ask`. `MOCK_LLM=1` is set as
the image's default `ENV`, so the container runs the graded mock baseline out of the box with
no secrets required.

**Verified:** the image has been built and run end to end with Docker (Docker Engine 29.8.0,
Linux containers). `docker build` completed successfully, including the build-time
`RUN python ingest.py` step, which reported *"Ingested 8 document(s) into ChromaDB collection
'zepto_policies' at /app/chroma_db"* inside the container. `docker run -p 7860:7860` then
started uvicorn (`Application startup complete. Uvicorn running on http://0.0.0.0:7860`) and
served `POST /ask` successfully for both routes — a policy question returning retrieved
`sources`, and a general question returning the canned no-retrieval response. Both responses
matched the transcripts recorded under Task 6 below exactly.

### Optional extension: real LLM (Groq)

```bash
export MOCK_LLM=0
export GROQ_API_KEY=<your free-tier key from console.groq.com>
uvicorn main:app --host 0.0.0.0 --port 7860
```

Entirely optional and ungraded — the required submission is graded with `MOCK_LLM` left at its
default, and is fully correct using only the mock baseline. If `GROQ_API_KEY` is missing or the
call fails, `classify_intent` falls back to the keyword heuristic and the answer-generation
nodes return a clearly marked `"ERROR: ..."` response after retrying (see Task 5 below) —
neither path crashes the request.

## Task-by-task summary

### Task 1 — Document corpus, chunking, embedding, storage

The 8 policy documents live in [`docs/`](docs/) as `doc_01.txt` … `doc_08.txt`, containing the
exact required text. [`ingest.py`](ingest.py) loads them, chunks them **one chunk per
document** (each doc is a single short paragraph, well within the "simple per-document chunk…
fine given their length" option the task allows), embeds each chunk locally with
`sentence-transformers`' `all-MiniLM-L6-v2` (no API key; the ~80MB model downloads once from
Hugging Face Hub and is cached locally afterward, the same pattern as `sns.load_dataset` in
Module 2), and stores the embeddings in a **persistent ChromaDB collection**
(`chroma_db/`, collection `zepto_policies`), **explicitly configured for cosine similarity**
(`metadata={"hnsw:space": "cosine"}` — ChromaDB defaults to L2 otherwise). `ingest()` is
idempotent: it's a no-op if the collection is already fully populated.

Verified: all 8 documents are embedded and queryable — a standalone retrieval sanity check
(`python ingest.py`) and further ad-hoc queries below all return the semantically correct
document.

### Task 2 — Structured prompt template

[`prompts.py`](prompts.py) defines `RAG_PROMPT_TEMPLATE` (used by `retrieve_and_answer`'s
optional real-LLM branch) and `DIRECT_PROMPT_TEMPLATE` (used by `direct_answer`'s optional
real-LLM branch). Both follow the **role → context → task → format → length** skeleton
verbatim as section headers, and both include an explicit **negative constraint** ("Do not
answer using information not present in the provided Context…") and a **few-shot example**
embedded directly in the template text. Full text is in `prompts.py` — not just described here,
but present as the actual strings the optional extension sends to the LLM.

### Task 3, 4 & 5 — LangGraph pipeline, routing, structured output

[`graph.py`](graph.py) defines a `TypedDict` state (`GraphState`: `query`, `intent`,
`retrieved`, `answer`, `sources`, `confidence`) and a `StateGraph` with exactly 3 nodes:

- **`classify_intent`** — mock mode (graded baseline): keyword heuristic over
  `["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card",
  "support hours"]`, no LLM call. Optional `MOCK_LLM=0`: calls the LLM (`llm.classify_intent_llm`),
  falling back to the same keyword heuristic if that call fails for any reason.
- **`retrieve_and_answer`** — always retrieves the top-3 chunks for real via
  `ingest.retrieve_top_chunks` (no API key needed in either mode). Mock mode: returns
  `f"Based on the retrieved context: {top_chunk_snippet}"` from the single top chunk. Optional
  `MOCK_LLM=0`: prompts the real LLM with `RAG_PROMPT_TEMPLATE`, grounded only in the retrieved
  chunks.
- **`direct_answer`** — mock mode: fixed canned string, no LLM call. Optional `MOCK_LLM=0`:
  prompts the LLM directly with `DIRECT_PROMPT_TEMPLATE`, no retrieval.

A **conditional edge** (`route_after_classify`) routes `classify_intent`'s output to
`retrieve_and_answer` (if `policy_question`) or `direct_answer` (if `general_question`) — this
routing logic itself never depends on `MOCK_LLM`, only each node's internal generation step
does.

**Structured output (Task 5):** every path returns a `schemas.AskResponse` Pydantic model
(`answer: str`, `sources: list[str]`, `confidence: float` in `[0, 1]`). In mock mode this is
populated deterministically from code (no LLM output exists to fail validation):
`sources` = the ids of the top-3 retrieved chunks for `policy_question`, `[]` for
`general_question`; `confidence` = a fixed `1.0`. The optional `MOCK_LLM=0` path's retry logic
lives in [`llm.py`](llm.py)'s `call_llm_and_validate`: on a JSON-parse or Pydantic validation
failure, it retries up to **2 additional times** with a corrective instruction appended to the
prompt, then — if still invalid, or if the LLM call itself fails (e.g. missing API key) —
returns a clearly marked `"ERROR: ..."` response instead of crashing.

This retry logic was tested directly (not just written and left untriggered): a simulated
flaky LLM that returns invalid JSON twice then valid JSON on the 3rd attempt was correctly
recovered; a simulated always-invalid LLM was correctly given up on after exactly 3 attempts
(1 initial + 2 retries), returning `sources` = the retrieval fallback and `confidence = 0.0`.

### Task 6 — FastAPI wrapper + example calls

[`main.py`](main.py) wraps the graph in a `POST /ask` endpoint (`AskRequest` → `AskResponse`),
run locally via `uvicorn main:app --host 0.0.0.0 --port 7860`. Both calls below were made with
a real running `uvicorn` server (not an in-process test client) and `MOCK_LLM` left at its
default:

**Call 1 — policy question (triggers `retrieve_and_answer`):**

```
POST /ask
{"query": "How much does standard delivery cost and is there a fee for small orders?"}
```
```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_03"],"confidence":1.0}
```

**Call 2 — general question (triggers `direct_answer`, no retrieval):**

```
POST /ask
{"query": "Can you recommend a good sci-fi movie to watch tonight?"}
```
```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

**Additional retrieval-accuracy spot checks** (top-1 retrieved doc per query, via
`ingest.retrieve_top_chunks`, confirming retrieval returns the *semantically correct* source
document, not just *a* document):

| Query | Top retrieved doc |
|---|---|
| "What is your refund policy for damaged items?" | `doc_02` (Returns & Refunds) |
| "How do I cancel my membership subscription?" | `doc_03` (Membership Tiers) |
| "Can I track my order in real time?" | `doc_04` (Order Tracking) |
| "What are your customer support hours?" | `doc_08` (Customer Support Hours) |
| "Do gift cards expire?" | `doc_07` (Gift Cards) |

All 5/5 correct.

### Task 7 — Dockerfile

[`Dockerfile`](Dockerfile): `python:3.12-slim` base, installs `support_assistant/requirements.txt`,
copies the app, runs `python ingest.py` at build time (so the container serves requests
immediately on start — also idempotent/self-healing via `main.py`'s startup hook regardless),
sets `MOCK_LLM=1` as the default environment, exposes port 7860, and starts
`uvicorn main:app --host 0.0.0.0 --port 7860`. Build/run commands are documented above under
"Docker". (No live Hugging Face Space deployment was attempted — that optional stretch was not
pursued.)

### Task 8 — RAG pipeline architecture

The pipeline has four stages, in this order: **ingestion → embedding → retrieval →
generation**.

**1. Ingestion.** The ingestion stage loads the 8 Zepto policy documents and breaks them into
chunks. In this project a simple one-chunk-per-document approach is enough, because the
documents are short. This prepares the policy text so it can be stored and searched. This is
handled by `ingest.py`.

**2. Embedding.** Each document chunk is converted into a numerical embedding using the
`all-MiniLM-L6-v2` model. These embeddings represent the meaning of the text, and they are
stored in the ChromaDB collection.

**3. Retrieval.** When a user asks a policy question, the query is also embedded and compared
with the stored document embeddings. The `retrieve_and_answer` node retrieves the top 3 most
similar chunks from ChromaDB using cosine similarity.

**4. Generation.** The retrieved information is then used to create the final answer. In
`MOCK_LLM` mode the system does not call a real LLM — instead it returns a fixed response built
from the most relevant retrieved chunk. If `MOCK_LLM=0`, the optional real-LLM path can
generate the answer using the retrieved context and the structured prompt.

Simple flow:

```
Policy documents → ingest.py → embeddings → ChromaDB → retrieve_and_answer → answer
```

**How the router decides where a query goes.** The router checks the intent of the user's
question. In the required mock mode, `classify_intent` uses a keyword-based rule: if the
question contains words such as *delivery, return, refund, membership, tracking, cancel, gift
card,* or *support hours*, it is classified as a `policy_question` and sent to
`retrieve_and_answer`. Otherwise it is classified as a `general_question` and sent to
`direct_answer`. The important point is that the routing itself does not depend on `MOCK_LLM`.

**Why retrieval works without an API key.** Retrieval does not need a paid or external LLM. The
project creates embeddings locally using `all-MiniLM-L6-v2` and searches them in the local
ChromaDB database, and both of these run locally. An API key is only needed for the optional
real-LLM generation when `MOCK_LLM=0`. So retrieval is local and needs no API key, while real
answer generation is optional and may need one. The required graded baseline uses the mock
generation and needs no API key at all.

**Why embeddings instead of keyword search.** Embeddings let the system find text based on
meaning, not just exact words. For example, a user might ask *"What will I be charged if my
order is small?"* — the policy talks about orders below INR 149 and a delivery fee, even though
it does not use the same words as the question. Embeddings allow the system to compare the
meaning of the query with the meaning of the policy text and retrieve the relevant information.
This is why the project uses embeddings plus ChromaDB for the RAG pipeline.

---

**Detailed component reference:**

```
 docs/doc_01.txt..doc_08.txt
          │  (ingestion: ingest.load_documents — 1 chunk per doc)
          ▼
 sentence-transformers all-MiniLM-L6-v2
          │  (embedding: ingest.get_embedding_model().encode(...))
          ▼
 ChromaDB collection "zepto_policies"  (chroma_db/, cosine similarity)
          │  (storage — populated once by ingest.ingest())
          ▼
      ┌───────────────────┐
      │  classify_intent   │  keyword heuristic (mock) / LLM (MOCK_LLM=0)
      └─────────┬─────────┘
                 │ conditional edge (route_after_classify)
       ┌─────────┴─────────┐
       ▼                   ▼
┌─────────────────┐  ┌───────────────┐
│retrieve_and_answer│  │ direct_answer │
│ - ingest.retrieve_ │  │ canned string │
│   top_chunks()     │  │ (mock) / LLM  │
│   (real retrieval, │  │ no retrieval  │
│   both modes)      │  └───────┬───────┘
│ - canned template   │          │
│   (mock) / LLM +     │          │
│   RAG_PROMPT_TEMPLATE│          │
│   (MOCK_LLM=0)        │          │
└─────────┬───────────┘          │
          └───────────┬──────────┘
                       ▼
        Pydantic AskResponse (answer, sources, confidence)
                       │
                       ▼
              FastAPI POST /ask  (main.py)
```

**Stage → component mapping:**

- **Ingestion** — `ingest.load_documents()` reads the 8 `docs/*.txt` files, one chunk per
  document.
- **Embedding** — `ingest.get_embedding_model()` (`all-MiniLM-L6-v2`, local, no API key) turns
  each chunk into a vector; `ingest.ingest()` writes them into the `zepto_policies` ChromaDB
  collection under `chroma_db/`.
- **Retrieval** — the `retrieve_and_answer` LangGraph node (`graph.py`) calls
  `ingest.retrieve_top_chunks(query, n_results=3)`, which embeds the incoming query with the
  same model and queries ChromaDB via cosine similarity. This step runs for real in **both**
  `MOCK_LLM` states — it needs no API key or network call either way.
- **Generation** — mock mode: `retrieve_and_answer` builds the canned
  `"Based on the retrieved context: ..."` string in code, and `direct_answer` returns a fixed
  canned string — both in `graph.py`, no LLM call. Optional `MOCK_LLM=0` state: `graph.py`
  builds a prompt from `prompts.RAG_PROMPT_TEMPLATE` (grounded, retrieval-based) or
  `prompts.DIRECT_PROMPT_TEMPLATE` (no retrieval) and calls `llm.call_llm_and_validate()`,
  which hits Groq's API and validates/retries the JSON output against `schemas.AskResponse`.

**What changes under `MOCK_LLM`:** ingestion, embedding, and retrieval are identical in both
states — only the **generation** stage (the final answer text/sources/confidence inside
`retrieve_and_answer` and `direct_answer`, and the classification decision inside
`classify_intent`) branches on the toggle. In the default mock state, generation is
deterministic code with no network calls; in the optional `MOCK_LLM=0` state, generation
becomes a real Groq API call using the Task 2 prompt templates, with schema-validation retries
and a graceful, clearly-marked failure path if the call can't succeed.

## Acceptance criteria checklist

- [x] All 8 corpus documents embedded and queryable from ChromaDB (verified: 5/5 retrieval spot checks correct)
- [x] Structured prompt template shows all 5 skeleton components + negative constraint + few-shot example, as actual text (`prompts.py`)
- [x] `classify_intent`'s keyword heuristic correctly routes a policy-style query to `policy_question` and an unrelated query to `general_question`, no LLM call in mock mode (Calls 1 & 2 above)
- [x] Graph has the 3 named nodes + working conditional edge, demonstrated by one example each of `retrieve_and_answer` and `direct_answer` routing
- [x] Retrieval for a policy question returns chunks from the correct source document (5/5 spot checks)
- [x] Mock-mode outputs follow the exact canned templates for both nodes, no network call
- [x] Pydantic schema (`answer`/`sources`/`confidence`) correctly populated in mock mode; retry-on-failure logic present **and tested** for the optional real-LLM path
- [x] FastAPI app runs locally via `uvicorn`; both example calls' raw JSON responses shown above
- [x] Dockerfile present, documented as buildable/runnable locally (build/run commands above) — required baseline; Hugging Face Spaces deployment not attempted (optional, not required)
- [x] README includes the full ingestion → embedding → retrieval → generation architecture description, naming each component/file/node and stating what changes under `MOCK_LLM`
