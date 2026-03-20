

import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH       = "data/qa_data.json"
EMBED_CACHE     = "embeddings/embeddings_cache.pkl"   # cache so we don't re-embed every run
MODEL_NAME      = "all-MiniLM-L6-v2"                 # fast 384-dim sentence-transformer


# ─────────────────────────────────────────────
#  STEP 1 — LOAD & PREPROCESS DOCUMENTS
# ─────────────────────────────────────────────
def load_documents(path: str = DATA_PATH) -> list[dict]:
    """
    Load domain-specific QA pairs from a JSON file.

    Each document is a dict with keys: id, question, answer.
    We concatenate question + answer into a single 'text' field
    because we want the embedding to capture the full meaning,
    not just the question surface form.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    documents = []
    for item in raw:
        doc = {
            "id":       item["id"],
            "question": item["question"],
            "answer":   item["answer"],
            # Combined text used for embedding — richer signal than question alone
            "text":     f"Question: {item['question']}\nAnswer: {item['answer']}"
        }
        documents.append(doc)

    return documents


# ─────────────────────────────────────────────
#  STEP 2 — GENERATE EMBEDDINGS
# ─────────────────────────────────────────────
def get_embeddings(documents: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Convert each document's text into a dense vector (embedding).

    WHY EMBEDDINGS?
    ───────────────
    Computers can't compare meaning in raw text, but they CAN compare
    numbers. A sentence-transformer maps text → a 384-dimensional vector
    such that semantically similar sentences land nearby in that space.

    Example:
      "Can I borrow a book?" → [0.12, -0.34, 0.87, ...]   ← 384 numbers
      "Library borrowing policy" → [0.11, -0.31, 0.90, ...] ← close!
      "Football rules" → [-0.45, 0.23, -0.12, ...]          ← far away

    We CACHE the embeddings to disk so we don't recompute on every app restart.
    """
    os.makedirs("embeddings", exist_ok=True)

    # ── Cache hit: load pre-computed embeddings ──
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE, "rb") as f:
            cached = pickle.load(f)
        # Validate cache matches current data (same number of docs)
        if len(cached) == len(documents):
            return cached

    # ── Cache miss: compute fresh embeddings ──
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,   # progress bar during initial load
        batch_size=16,            # process 16 texts at a time (memory efficient)
        normalize_embeddings=True # L2-normalize → cosine sim = dot product (faster)
    )

    # Save to cache
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


# ─────────────────────────────────────────────
#  STEP 3 — VECTOR SEARCH (RETRIEVAL)
# ─────────────────────────────────────────────
def retrieve_top_k(
    query: str,
    documents: list[dict],
    doc_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
    similarity_threshold: float = 0.2
) -> list[dict]:
    """
    Find the top-k most relevant documents for a user query.

    HOW COSINE SIMILARITY WORKS:
    ─────────────────────────────
    Imagine each embedding as an arrow pointing in some direction in 384D space.
    Cosine similarity = the cosine of the angle between two arrows.
      - Score = 1.0 → arrows point in EXACTLY the same direction (identical meaning)
      - Score = 0.0 → arrows are perpendicular (unrelated)
      - Score = -1.0 → opposite directions (opposite meaning)

    We pick the documents whose arrows point closest to the query arrow.

    Args:
        query              : The user's question (raw text)
        documents          : All loaded documents
        doc_embeddings     : Pre-computed embeddings matrix (N × 384)
        model              : SentenceTransformer instance
        top_k              : How many documents to return
        similarity_threshold: Minimum score to include a result

    Returns:
        List of dicts: {question, answer, similarity_score}
    """
    # Embed the query using the SAME model (crucial — must be same embedding space)
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )  # shape: (1, 384)

    # Compute cosine similarity between query and every document
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]  # shape: (N,)

    # Get indices sorted by similarity (highest first)
    ranked_indices = np.argsort(similarities)[::-1]

    # Collect top_k results above the threshold
    results = []
    for idx in ranked_indices[:top_k]:
        score = float(similarities[idx])
        if score >= similarity_threshold:
            results.append({
                "question":        documents[idx]["question"],
                "answer":          documents[idx]["answer"],
                "similarity_score": round(score, 4)
            })

    return results


# ─────────────────────────────────────────────
#  STEP 4 — BUILD AUGMENTED PROMPT
# ─────────────────────────────────────────────
def build_prompt(user_query: str, retrieved_docs: list[dict]) -> str:
    """
    Construct the final prompt for the LLM using FEW-SHOT + RAG technique.

    PROMPT ENGINEERING TECHNIQUES USED HERE:
    ──────────────────────────────────────────
    1. ROLE ASSIGNMENT    → "You are a college regulations assistant..."
                            Sets the LLM's persona and domain expertise.

    2. CONTEXT INJECTION  → Paste retrieved documents BEFORE the question.
                            This is the "Augmented" part of RAG.
                            LLMs read left-to-right, so context before the
                            question ensures it's considered when answering.

    3. CHAIN-OF-THOUGHT   → "Use ONLY the provided context... If not found, say so."
                            Guides the model to reason step-by-step and
                            NOT hallucinate when information is missing.

    4. FEW-SHOT FORMAT    → The "Context:" block implicitly shows the LLM
                            the expected input format with Q&A pairs.

    5. OUTPUT CONSTRAINTS → "Be concise", "cite which rule applies"
                            Shapes the output style for domain appropriateness.
    """
    if not retrieved_docs:
        # No relevant docs found — tell LLM to admit it
        context_block = "No relevant regulations found in the knowledge base."
    else:
        # Build a numbered context block from retrieved documents
        context_lines = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_lines.append(
                f"[Source {i} | Relevance: {doc['similarity_score']}]\n"
                f"Q: {doc['question']}\n"
                f"A: {doc['answer']}"
            )
        context_block = "\n\n".join(context_lines)

    # ── Assembled prompt using SYSTEM + CONTEXT + QUERY pattern ──
    prompt = f"""You are an academic regulations assistant for a college. Your job is to help students understand college rules, policies, and procedures.

INSTRUCTIONS:
- Answer ONLY based on the context provided below.
- If the answer is not in the context, clearly say: "I don't have information about that in our regulations database. Please contact the Academic Office."
- Be clear, concise, and student-friendly in your tone.
- Cite which source/rule you are referring to when applicable.
- Do NOT make up or guess any policies, fees, or deadlines.

RETRIEVED CONTEXT FROM COLLEGE REGULATIONS:
{context_block}

STUDENT QUESTION:
{user_query}

ANSWER:"""

    return prompt


# ─────────────────────────────────────────────
#  RAG PIPELINE (puts it all together)
# ─────────────────────────────────────────────
class RAGPipeline:
    """
    Wraps the full Retrieve-Augment pipeline into a single reusable object.

    Usage:
        rag = RAGPipeline()
        prompt, docs = rag.query("What is the attendance policy?")
        # then send `prompt` to any LLM API
    """

    def __init__(self):
        print("[RAG] Loading embedding model...")
        self.model     = SentenceTransformer(MODEL_NAME)
        self.documents = load_documents()
        print(f"[RAG] Loaded {len(self.documents)} documents from {DATA_PATH}")
        self.embeddings = get_embeddings(self.documents, self.model)
        print(f"[RAG] Embeddings ready — shape: {self.embeddings.shape}")

    def query(self, user_query: str, top_k: int = 3) -> tuple[str, list[dict]]:
        """
        Full RAG pipeline for a single user query.

        Returns:
            prompt      : Augmented prompt ready for the LLM
            retrieved   : The source documents used (for display/debugging)
        """
        # 1. Retrieve relevant documents
        retrieved = retrieve_top_k(
            query          = user_query,
            documents      = self.documents,
            doc_embeddings = self.embeddings,
            model          = self.model,
            top_k          = top_k
        )

        # 2. Build augmented prompt
        prompt = build_prompt(user_query, retrieved)

        return prompt, retrieved
