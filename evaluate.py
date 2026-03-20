
from rag_engine import RAGPipeline


# ─────────────────────────────────────────────
#  SAMPLE EVALUATION QUERIES
# ─────────────────────────────────────────────
# Each entry: (query_text, expected_topic_keywords)
EVAL_QUERIES = [
    (
        "What percentage of classes must I attend?",
        ["attendance", "75%", "eligible", "examination"]
    ),
    (
        "How much does re-evaluation cost?",
        ["re-evaluation", "500", "Form RE-01"]
    ),
    (
        "Tell me about scholarship eligibility",
        ["scholarship", "GPA", "3.5", "tuition"]
    ),
    (
        "What is the fine for returning library books late?",
        ["fine", "Rs. 5", "14 days", "library"]
    ),
    (
        "Can I pay my fees in installments?",
        ["installment", "Finance Office", "late fee"]
    ),
    (
        "What GPA do I need to pass a semester?",
        ["GPA", "2.0", "grading", "semester"]
    ),
    (
        "How many times can I retake a failed subject?",
        ["supplementary", "3 attempts", "fail", "retake"]
    ),
    (
        "Is an internship required to graduate?",
        ["internship", "mandatory", "6 weeks", "report"]
    ),
    # Edge case: out-of-domain question
    (
        "What is the weather like today?",
        []  # Should return low similarity / no good match
    ),
]


def evaluate_retrieval(rag: RAGPipeline):
    """
    Run each eval query through retrieval and print results.
    No LLM needed — just checks if the RIGHT docs come back.
    """
    print("=" * 70)
    print("  RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    total_queries      = len(EVAL_QUERIES)
    successful_hits    = 0
    total_top1_score   = 0.0

    for i, (query, expected_keywords) in enumerate(EVAL_QUERIES, 1):
        print(f"\n[Query {i}/{total_queries}]")
        print(f"  Q: {query}")

        # Run retrieval
        _, retrieved_docs = rag.query(query, top_k=3)

        if not retrieved_docs:
            print("  ❌ No documents retrieved (below threshold)")
            continue

        # Check top-1 result
        top_doc   = retrieved_docs[0]
        top_score = top_doc["similarity_score"]
        total_top1_score += top_score

        # Check if expected keywords appear in retrieved answers
        all_retrieved_text = " ".join(
            d["question"] + " " + d["answer"] for d in retrieved_docs
        ).lower()

        if expected_keywords:
            matched_keywords = [
                kw for kw in expected_keywords
                if kw.lower() in all_retrieved_text
            ]
            hit_rate = len(matched_keywords) / len(expected_keywords)
            successful_hits += 1 if hit_rate >= 0.5 else 0

            print(f"  Top-1 Match   : '{top_doc['question'][:60]}...'")
            print(f"  Similarity    : {top_score:.4f}")
            print(f"  Keyword Hits  : {len(matched_keywords)}/{len(expected_keywords)} "
                  f"({hit_rate*100:.0f}%) — {matched_keywords}")
        else:
            # Out-of-domain: expect LOW similarity
            ood_flag = "✅ Correctly LOW" if top_score < 0.4 else "⚠️ Unexpectedly HIGH"
            print(f"  [Out-of-Domain] Top-1 similarity: {top_score:.4f} → {ood_flag}")
            successful_hits += 1 if top_score < 0.4 else 0

        # Show all 3 retrieved docs
        print("  Retrieved docs:")
        for j, doc in enumerate(retrieved_docs, 1):
            print(f"    [{j}] score={doc['similarity_score']:.3f} | {doc['question'][:55]}...")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total queries tested   : {total_queries}")
    print(f"  Successful retrievals  : {successful_hits}/{total_queries} "
          f"({successful_hits/total_queries*100:.1f}%)")
    print(f"  Avg Top-1 Similarity   : {total_top1_score/total_queries:.4f}")
    print()


def show_full_prompt_example(rag: RAGPipeline):
    """
    Show one complete augmented prompt so you can see exactly
    what the LLM receives — great for understanding prompt engineering.
    """
    print("=" * 70)
    print("  AUGMENTED PROMPT EXAMPLE (what the LLM receives)")
    print("=" * 70)

    query = "What are the rules for hostel students?"
    augmented_prompt, docs = rag.query(query)

    print(f"\nOriginal Query: '{query}'")
    print(f"Retrieved {len(docs)} document(s)\n")
    print("─" * 70)
    print(augmented_prompt)
    print("─" * 70)


def show_embedding_info(rag: RAGPipeline):
    """
    Show embedding statistics — helps understand the vector space.
    """
    import numpy as np

    print("\n" + "=" * 70)
    print("  EMBEDDING STATISTICS")
    print("=" * 70)
    emb = rag.embeddings
    print(f"  Matrix shape       : {emb.shape}  (docs × dimensions)")
    print(f"  Embedding model    : all-MiniLM-L6-v2")
    print(f"  Dimension          : {emb.shape[1]}  (each doc → 384 numbers)")
    print(f"  Min value          : {emb.min():.4f}")
    print(f"  Max value          : {emb.max():.4f}")
    print(f"  Mean value         : {emb.mean():.4f}")
    print(f"  L2 norm (row 0)    : {np.linalg.norm(emb[0]):.4f}  (≈1.0 = normalized)")
    print()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔄 Loading RAG Pipeline for evaluation...\n")
    rag = RAGPipeline()

    show_embedding_info(rag)
    evaluate_retrieval(rag)
    show_full_prompt_example(rag)

    print("\n✅ Evaluation complete! Run 'streamlit run app.py' to use the chatbot.\n")
