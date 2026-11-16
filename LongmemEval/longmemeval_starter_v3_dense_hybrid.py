"""LongMemEval-S diagnostic runner: BM25, Dense Retrieval, DPR, coverage and Oracle.

Examples (Windows PowerShell; dataset beside this file):
    py longmemeval_starter_v3_75_fixed.py --self-test
    py -m pip install openai tiktoken sentence-transformers
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions bm25 dense --summary-only
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions dpr --summary-only
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions dense --run --max-calls 25 --results dense_none.jsonl
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions bm25 --effort none
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions bm25 --effort none --run --max-calls 25 --results bm25_none_v2.jsonl
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions bm25 --effort medium --run --max-calls 25 --results bm25_medium_v2.jsonl
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions hybrid --effort none --run --max-calls 25 --results hybrid_none_v2.jsonl
    py -X utf8 longmemeval_starter_v3_75_fixed.py --data longmemeval_s_cleaned.json --conditions oracle --ids REAL_FAILED_QID1 REAL_FAILED_QID2 --effort none

No --run => print FREE preview, do not contact the API. Dense/DPR downloads public pretrained weights on first use. --ids overrides the fixed 25 question IDs.
Use Oracle only on BM25 failures identified after manual grading. Annotation IDs and
reference answers are for offline evaluation only and never enter the LLM prompt.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

MODEL = "gpt-6-luna"
HISTORY_TOKEN_BUDGET = 8000
INPUT_TOKEN_GUARD = 9000
MAX_OUTPUT_TOKENS = 1024
BM25_K1 = 1.2
BM25_B = 0.25  # Preserve the effective length normalization of the initial study.
CONDITIONS = ("no_history", "recent", "bm25", "dense", "dpr", "hybrid", "oracle")
HYBRID_RRF_K = 60  # Reciprocal Rank Fusion parameter for hybrid retrieval
DENSE_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DPR_QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"
DPR_CONTEXT_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
# These are retriever-tokenizer tokens, not OpenAI/tiktoken tokens.
RETRIEVAL_CHUNK_TOKENS = 256
RETRIEVAL_CHUNK_OVERLAP = 32
RETRIEVAL_BATCH_SIZE = 16
PROMPT_VERSION = "starter-v3-dense"
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
FIXED_25_IDS = (
    "caf9ead2", "a06e4cfe", "25e5aa4f", "58bf7951", "86b68151",
    "18dcd5a5", "e48988bc", "c7cf7dfd", "6222b6eb", "c4f10528",
    "ce6d2d27", "a2f3aa27", "f685340e", "26bdc477", "3ba21379",
    "gpt4_d6585ce8", "gpt4_2312f94c", "4dfccbf7",
    "gpt4_8279ba03", "993da5e2",
    "bf659f65", "5a7937c8", "e25c3b8d", "c2ac3c61", "2b8f3739",
)


def tokenizer():
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("Install: py -m pip install openai tiktoken") from exc
    return tiktoken.get_encoding("o200k_base")


def words(text):
    return WORD_RE.findall(text.lower())


def load_data(path):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Expected the cleaned LongMemEval-S JSON to be a list")
    return data


def choose_questions(data, ids):
    index = {row["question_id"]: row for row in data}
    if len(ids) != len(set(ids)) or not 1 <= len(ids) <= 30:
        raise ValueError("Supply 1-30 distinct question IDs")
    missing = set(ids) - index.keys()
    if missing:
        raise ValueError(f"Unknown question IDs: {sorted(missing)}")
    return [index[qid] for qid in ids]


def sessions_for(row):
    dates = row["haystack_dates"]
    ids = row["haystack_session_ids"]
    sessions = row["haystack_sessions"]
    if not (len(dates) == len(ids) == len(sessions)):
        raise ValueError(f"Unaligned sessions for {row['question_id']}")
    result = []
    for i, (date, sid, turns) in enumerate(zip(dates, ids, sessions)):
        # The raw sid is kept ONLY in internal metadata and evaluation logs.
        # Both the model input and the BM25 document use only neutral text.
        lines = [f"{date}"]
        for turn in turns:
            if turn.get("role") not in ("user", "assistant"):
                raise ValueError(f"Unexpected role in {row['question_id']}")
            lines.append(f"{turn['role']}: {turn['content']}")
        result.append({"id": sid, "date": str(date), "ordinal": i,
                       "text": "\n".join(lines)})
    return sorted(result, key=lambda session: (session["date"], session["ordinal"]))


def bm25_order(question, sessions):
    query = set(words(question))
    docs = [words(session["text"]) for session in sessions]
    lengths = [len(doc) for doc in docs]
    avg = sum(lengths) / max(len(docs), 1)
    df = Counter(term for doc in docs for term in set(doc))

    def score(i):
        counts = Counter(docs[i])
        total = 0.0
        for term in query:
            count = counts[term]
            if count:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                # Standard BM25 normalization: (1 - b) + b * doc_length / avg_length
                norm = (1 - BM25_B) + BM25_B * lengths[i] / max(avg, 1)
                total += idf * count * (BM25_K1 + 1) / (count + BM25_K1 * norm)
        return total

    return sorted(range(len(sessions)), key=lambda i: (-score(i), i))



class DenseRetriever:
    """Pretrained bi-encoder retrieval; no fine-tuning and no OpenAI API calls.

    backend='dense': lightweight MiniLM semantic search (cosine similarity)
    backend='dpr': paired, pretrained DPR BERT encoders (dot product)
    Documents are split with the *retriever* tokenizer so no part of a long
    session disappears due to the model's 256/512-token input limit.
    """

    def __init__(self, backend, device="cpu", cache_dir=Path(".dense_cache")):
        if backend not in ("dense", "dpr"):
            raise ValueError(f"Unsupported dense backend: {backend}")
        self.backend = backend
        self.device = device
        self.cache_dir = Path(cache_dir)
        try:
            import numpy as np
        except ImportError as exc:
            raise SystemExit("Install dense dependencies: py -m pip install numpy") from exc
        self.np = np

        if backend == "dense":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise SystemExit("Install: py -m pip install sentence-transformers") from exc
            print(f"Loading pretrained retrieval model {DENSE_MODEL} on {device} ...", file=sys.stderr)
            self.model_name = DENSE_MODEL
            self.model = SentenceTransformer(DENSE_MODEL, device=device)
            self.doc_tokenizer = self.model.tokenizer
            self.max_length = min(RETRIEVAL_CHUNK_TOKENS, int(self.model.max_seq_length or 512) - 2)
        else:
            try:
                import torch
                from transformers import (DPRQuestionEncoder, DPRContextEncoder,
                                          DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer)
            except ImportError as exc:
                raise SystemExit("Install DPR: py -m pip install torch transformers") from exc
            print("Loading pretrained DPR question + context encoders ...", file=sys.stderr)
            self.torch = torch
            self.model_name = DPR_QUESTION_MODEL + " + " + DPR_CONTEXT_MODEL
            self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(DPR_QUESTION_MODEL)
            self.doc_tokenizer = DPRContextEncoderTokenizer.from_pretrained(DPR_CONTEXT_MODEL)
            self.question_model = DPRQuestionEncoder.from_pretrained(DPR_QUESTION_MODEL).to(device).eval()
            self.context_model = DPRContextEncoder.from_pretrained(DPR_CONTEXT_MODEL).to(device).eval()
            self.max_length = RETRIEVAL_CHUNK_TOKENS - 2

    def _make_chunks(self, sessions):
        passages, owners = [], []
        for i, session in enumerate(sessions):
            # Only sanitized history text is encoded; annotation IDs/answers are never used.
            # Read the full token sequence before chunking. verbose=False only
            # suppresses the expected overlong-input warning from the tokenizer;
            # no text is truncated here.
            ids = self.doc_tokenizer(
                session["text"], add_special_tokens=False,
                truncation=False, verbose=False,
            )["input_ids"]
            if not ids:
                ids = self.doc_tokenizer(
                    "(empty session)", add_special_tokens=False,
                    truncation=False, verbose=False,
                )["input_ids"]
            step = self.max_length - RETRIEVAL_CHUNK_OVERLAP
            if step <= 0:
                raise ValueError("Retrieval chunk overlap must be below chunk length")
            for start in range(0, len(ids), step):
                chunk_ids = ids[start:start + self.max_length]
                passages.append(self.doc_tokenizer.decode(
                    chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                owners.append(i)
                if start + self.max_length >= len(ids):
                    break
        return passages, self.np.asarray(owners, dtype=self.np.int32)

    def _encode_documents(self, passages):
        if self.backend == "dense":
            return self.model.encode_document(
                passages, batch_size=RETRIEVAL_BATCH_SIZE,
                normalize_embeddings=True, show_progress_bar=False,
                convert_to_numpy=True).astype("float32")
        chunks = []
        for start in range(0, len(passages), RETRIEVAL_BATCH_SIZE):
            batch = self.doc_tokenizer(
                passages[start:start + RETRIEVAL_BATCH_SIZE],
                padding=True, truncation=True,
                max_length=RETRIEVAL_CHUNK_TOKENS, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self.torch.inference_mode():
                embeddings = self.context_model(**batch).pooler_output
            chunks.append(embeddings.cpu().numpy().astype("float32"))
        return self.np.concatenate(chunks, axis=0)

    def _encode_question(self, question):
        if self.backend == "dense":
            return self.model.encode_query(
                question, normalize_embeddings=True,
                convert_to_numpy=True).astype("float32")
        batch = self.question_tokenizer(
            question, truncation=True, max_length=RETRIEVAL_CHUNK_TOKENS,
            return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with self.torch.inference_mode():
            embedding = self.question_model(**batch).pooler_output
        return embedding[0].cpu().numpy().astype("float32")

    def _document_vectors(self, sessions):
        # Cache embeddings on disk so rerunning a preview / paid run doesn't
        # re-encode the entire same haystack. Key never includes gold annotations.
        sha = hashlib.sha256()
        sha.update(self.model_name.encode("utf-8"))
        sha.update(f"|chunk={self.max_length}|overlap={RETRIEVAL_CHUNK_OVERLAP}".encode())
        for s in sessions:
            raw = s["text"].encode("utf-8")
            sha.update(len(raw).to_bytes(8, "big"))
            sha.update(raw)
        cache_path = self.cache_dir / f"{self.backend}_{sha.hexdigest()}.npz"
        if cache_path.exists():
            with self.np.load(cache_path, allow_pickle=False) as archive:
                vectors = archive["vectors"]
                owners = archive["owners"]
            if (vectors.ndim == 2 and owners.ndim == 1
                    and len(vectors) == len(owners)
                    and (len(owners) == 0 or int(owners.max()) < len(sessions))):
                return vectors, owners
            raise ValueError(f"Invalid dense cache {cache_path}; delete and retry")
        passages, owners = self._make_chunks(sessions)
        vectors = self._encode_documents(passages)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.np.savez_compressed(cache_path, vectors=vectors, owners=owners)
        return vectors, owners

    def order(self, question, sessions):
        if not sessions:
            return []
        vectors, owners = self._document_vectors(sessions)
        qvec = self._encode_question(question)
        chunk_scores = vectors @ qvec  # cosine for MiniLM; raw dot-product for DPR
        scores = [float("-inf")] * len(sessions)
        for owner, score in zip(owners, chunk_scores):
            scores[int(owner)] = max(scores[int(owner)], float(score))
        # Aggregate chunks back to sessions using maximum passage relevance;
        # retain exact existing session-level 8k-token selection for comparability.
        return sorted(range(len(sessions)), key=lambda i: (-scores[i], i))

def rrf_fuse_orders(bm25_indices, semantic_indices, k=60):
    scores = [0.0] * len(bm25_indices)

    for ranking in (bm25_indices, semantic_indices):
        for rank, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (k + rank)

    return sorted(
        range(len(scores)),
        key=lambda i: (-scores[i], i)
    )

def backend_for_condition(condition, hybrid_backend="dense"):
    """Map an experiment condition to a reusable semantic retriever."""
    if condition == "hybrid":
        return hybrid_backend
    if condition in ("dense", "dpr"):
        return condition
    return None


def get_retriever_for_condition(condition, hybrid_backend, retrievers, factory):
    """Initialize/cache one semantic retriever; also route Hybrid to it.

    The same backend object is reused when both a standalone semantic condition
    and a Hybrid condition are selected in the same invocation.
    """
    backend = backend_for_condition(condition, hybrid_backend)
    if backend is None:
        return None
    if backend not in retrievers:
        retrievers[backend] = factory(backend)
    return retrievers[backend]


def prompt_version_for_condition(condition, hybrid_backend="dense"):
    """Prevent accidental reuse when the model, prompt or RRF settings change."""
    backend = backend_for_condition(condition, hybrid_backend)
    if backend == "dense":
        model_id = DENSE_MODEL
    elif backend == "dpr":
        model_id = DPR_QUESTION_MODEL + "+" + DPR_CONTEXT_MODEL
    else:
        return PROMPT_VERSION

    version = (f"{PROMPT_VERSION}|{model_id}|"
               f"chunk={RETRIEVAL_CHUNK_TOKENS}|overlap={RETRIEVAL_CHUNK_OVERLAP}")
    if condition == "hybrid":
        version += f"|fusion=rrf|rrf_k={HYBRID_RRF_K}|weights=1+1"
    return version


def history_text(selected):
    return "\n\n".join(session["text"] for session in selected)


def history_tokens(selected, enc):
    return len(enc.encode(history_text(selected)))


def coverage_info(row, selected):
    gold = set(row.get("answer_session_ids", []))
    selected_ids = {item["id"] for item in selected}
    found = gold & selected_ids
    if not gold:
        label = "unknown"
    elif found == gold:
        label = "all"
    elif found:
        label = "some"
    else:
        label = "none"
    return {
        "evidence_coverage": label,
        "evidence_found": len(found),
        "evidence_total": len(gold),
        "evidence_recall": len(found) / len(gold) if gold else None,
    }


def select_sessions(row, condition, enc, retriever=None):
    if condition == "no_history":
        return []
    sessions = sessions_for(row)  # Chronological order.
    if condition == "oracle":
        gold_ids = set(row.get("answer_session_ids", []))
        if not gold_ids:
            raise ValueError(f"No annotated gold evidence for {row['question_id']}")
        selected = [session for session in sessions if session["id"] in gold_ids]
        found_ids = {s["id"] for s in selected}
        if found_ids != gold_ids:
            raise ValueError(f"Missing gold sessions for {row['question_id']}")
        # Diagnostic Oracle always retains *all* annotated support sessions.
        # An over-budget Oracle is allowed if the full prompt passes its own
        # input guard; it must be labeled as over-budget in the results.
        return selected

    if condition == "recent":
        order = list(reversed(range(len(sessions))))
    elif condition == "bm25":
        order = bm25_order(row["question"], sessions)
    elif condition in ("dense", "dpr"):
        if retriever is None or retriever.backend != condition:
            raise ValueError(f"Condition {condition} needs its matching DenseRetriever")
        order = retriever.order(row["question"], sessions)
    elif condition == "hybrid":
        if retriever is None or retriever.backend not in ("dense", "dpr"):
            raise ValueError("Hybrid requires a DenseRetriever (dense or dpr)")
        bm25_ranking = bm25_order(row["question"], sessions)
        semantic_ranking = retriever.order(row["question"], sessions)
        order = rrf_fuse_orders(bm25_ranking, semantic_ranking, k=HYBRID_RRF_K)
    else:
        raise ValueError(f"Unknown condition {condition}")

    # Pre-tokenize each candidate just once. Re-encoding the growing concatenated
    # history on *every* candidate can be very slow for long-memory haystacks.
    sizes = [len(enc.encode(session["text"])) for session in sessions]
    separator_tokens = len(enc.encode("\n\n"))
    selected_indices = []
    remaining = HISTORY_TOKEN_BUDGET
    for index in order:
        cost = sizes[index] + (separator_tokens if selected_indices else 0)
        if cost <= remaining:
            selected_indices.append(index)
            remaining -= cost
    # BPE token counts are not always additive across string boundaries.
    # Make the final concatenated history satisfy the strict budget.
    selected = [sessions[i] for i in sorted(selected_indices)]
    while selected and history_tokens(selected, enc) > HISTORY_TOKEN_BUDGET:
        # Drop lowest-priority selected session (not the latest chronologically).
        drop_index = next(i for i in reversed(order) if i in selected_indices)
        selected_indices.remove(drop_index)
        selected = [sessions[i] for i in sorted(selected_indices)]
    return selected


def make_prompt(row, selected):
    history = history_text(selected) if selected else "(No chat history provided)"
    return (
        "Answer the question using only the chat history below. "
        "If the history does not contain enough information, say 'I don't know'. "
        "Give a concise answer.\n\n"
        f"Question date: {row.get('question_date', 'unknown')}\n"
        f"Chat history:\n{history}\n\nQuestion: {row['question']}"
    )


def append_jsonl(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def export_grading_csv(data, results_path, csv_path):
    """Offline only: reference answers go to a separate manual grading sheet."""
    index = {row["question_id"]: row for row in data}
    records = read_jsonl(results_path)
    fields = ["question_id", "question_type", "condition", "effort",
              "question", "reference_answer", "hypothesis", "correct",
              "evidence_coverage", "evidence_found", "evidence_total",
              "evidence_recall", "status"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = index.get(record["question_id"])
            if row is None:
                raise ValueError(f"Result QID missing in dataset: {record['question_id']}")
            writer.writerow({
                "question_id": record["question_id"],
                "question_type": row.get("question_type", ""),
                "condition": record.get("condition", ""),
                "effort": record.get("effort", ""),
                "question": row["question"],
                "reference_answer": row["answer"],
                "hypothesis": record.get("hypothesis", ""),
                "correct": "",  # Human grader fills this; do not infer by string equality.
                "evidence_coverage": record.get("evidence_coverage", ""),
                "evidence_found": record.get("evidence_found", ""),
                "evidence_total": record.get("evidence_total", ""),
                "evidence_recall": record.get("evidence_recall", ""),
                "status": record.get("status", ""),
            })
    print(f"Created {csv_path}: {len(records)} rows (reference answers NOT sent to model)")


def key(qid, condition, model, prompt_version, effort):
    return (str(qid), condition, model, prompt_version, effort)


def key_for_record(r):
    return key(r["question_id"], r["condition"], r["model"],
               r["prompt_version"], r.get("effort"))


def self_test():
    class TestTokenizer:
        def encode(self, value):
            return words(value)

    enc = TestTokenizer()
    row = {
        "question_id": "local-test", "question_type": "knowledge-update",
        "question": "Where did Mira move in 2024?", "question_date": "2025-01-01",
        "answer": "Seattle", "answer_session_ids": ["answer_new_2"],
        "haystack_dates": ["2024-01-01", "2022-01-01"],
        "haystack_session_ids": ["answer_new_2", "sharegpt_old"],
        "haystack_sessions": [
            [{"role": "user", "content": "Mira moved to Seattle in 2024.", "has_answer": True}],
            [{"role": "user", "content": "Mira lived in Boston."}],
        ],
    }
    sessions = sessions_for(row)
    assert [s["id"] for s in sessions] == ["sharegpt_old", "answer_new_2"]
    assert all("answer_new_2" not in s["text"] and "sharegpt_old" not in s["text"]
               for s in sessions)
    for condition in ("no_history", "recent", "bm25", "oracle"):
        selected = select_sessions(row, condition, enc)
        prompt = make_prompt(row, selected)
        assert "answer_new_2" not in prompt
        assert "sharegpt_old" not in prompt
        assert "has_answer" not in prompt and "answer_session_ids" not in prompt
        assert ("Seattle" not in prompt) == (condition == "no_history")
        if condition == "oracle":
            assert [s["id"] for s in selected] == ["answer_new_2"]
            assert coverage_info(row, selected)["evidence_coverage"] == "all"
    class FakeRetriever:
        backend = "dense"

        def order(self, question, sessions):
            assert question == row["question"]
            assert all("answer_new_2" not in s["text"] for s in sessions)
            return [1, 0]

    dense_selected = select_sessions(row, "dense", enc, FakeRetriever())
    assert [s["id"] for s in dense_selected] == ["sharegpt_old", "answer_new_2"]
    # Regression test: the same retriever must be passed into Hybrid; no NoneType.
    built = []
    cache = {}
    def fake_factory(backend):
        built.append(backend)
        assert backend == "dense"
        return FakeRetriever()
    hybrid_retriever = get_retriever_for_condition(
        "hybrid", "dense", cache, fake_factory)
    assert hybrid_retriever is get_retriever_for_condition(
        "dense", "dense", cache, fake_factory)
    assert built == ["dense"]  # Don't load the same model twice.
    hybrid_selected = select_sessions(row, "hybrid", enc, hybrid_retriever)
    hybrid_prompt = make_prompt(row, hybrid_selected)
    assert "answer_new_2" not in hybrid_prompt and "sharegpt_old" not in hybrid_prompt
    assert "has_answer" not in hybrid_prompt
    assert rrf_fuse_orders([0, 1, 2], [1, 0, 2]) == [0, 1, 2]
    assert prompt_version_for_condition("dense") != PROMPT_VERSION
    assert prompt_version_for_condition("dpr") != prompt_version_for_condition("dense")
    assert prompt_version_for_condition("hybrid", "dense") != prompt_version_for_condition("dense")
    assert prompt_version_for_condition("hybrid", "dpr") != prompt_version_for_condition("hybrid", "dense")
    many = {**row, "answer_session_ids": ["answer_new_2", "sharegpt_old"]}
    assert coverage_info(many, [sessions[0]])["evidence_coverage"] == "some"
    assert coverage_info(row, [sessions[0]])["evidence_coverage"] == "none"
    assert coverage_info(many, sessions)["evidence_coverage"] == "all"
    assert key("q", "bm25", MODEL, PROMPT_VERSION, "none") != key(
        "q", "bm25", MODEL, PROMPT_VERSION, "medium"
    )
    print("PASS: sanitized prompts, BM25/Dense/DPR/Hybrid routing and RRF, chronology, Oracle, coverage, keys; NO API calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--ids", nargs="+", help="Explicit question IDs; overrides the default fixed 25")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["bm25"])
    parser.add_argument("--effort", choices=["none", "medium"], default="none")
    parser.add_argument("--hybrid-backend", choices=["dense", "dpr"], default="dense",
                            help="Semantic retriever to fuse with BM25 (default: MiniLM dense)")
    parser.add_argument("--retrieval-device", choices=["cpu", "cuda", "mps"], default="cpu",
                        help="Device for local dense/DPR encoding; CPU is safest")
    parser.add_argument("--dense-cache-dir", type=Path, default=Path(".dense_cache"),
                        help="Local cache of passage embeddings (not model responses)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Free preview with selected IDs and coverage; omit full prompt text")
    parser.add_argument("--results", type=Path, default=Path("results_v2.jsonl"))
    parser.add_argument("--max-calls", type=int, default=1,
                        help="Maximum number of NEW API calls allowed in this invocation")
    parser.add_argument("--run", action="store_true", help="Spend API credits; absent => FREE preview")
    parser.add_argument("--retry-uncertain", action="store_true",
                        help="Retry a logged attempt with no result ONLY after checking API usage")
    parser.add_argument("--self-test", action="store_true", help="Local tests; no data/API needed")
    parser.add_argument("--export-grading-csv", type=Path,
                        help="Offline: export a blank correctness sheet from --results")
    parser.add_argument("--oracle-max-input-tokens", type=int, default=9000,
                        help="Maximum estimated full-prompt tokens for diagnostic Oracle (default: 9000)")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.data or not args.data.is_file():
        parser.error("Give an existing --data path (or use --self-test)")
    if not 1 <= args.max_calls <= 1000:
        parser.error("--max-calls must be 1..1000")
    if not 1 <= args.oracle_max_input_tokens <= 100000:
        parser.error("--oracle-max-input-tokens must be 1..100000")
    if args.export_grading_csv:
        export_grading_csv(load_data(args.data), args.results, args.export_grading_csv)
        return
    enc = tokenizer()
    data = load_data(args.data)
    rows = choose_questions(data, args.ids if args.ids is not None else FIXED_25_IDS)
    print("Selected question IDs:", [row["question_id"] for row in rows])

    # Lazy-load each backend once, and reuse it for Hybrid when applicable.
    retrievers = {}
    def retriever_factory(backend):
        return DenseRetriever(backend, args.retrieval_device, args.dense_cache_dir)

    jobs = []
    skipped = []
    for row in rows:
        for condition in dict.fromkeys(args.conditions):
            retriever = get_retriever_for_condition(
                condition, args.hybrid_backend, retrievers, retriever_factory)
            selected = select_sessions(row, condition, enc, retriever)
            prompt = make_prompt(row, selected)
            estimate = len(enc.encode(prompt))
            guard = args.oracle_max_input_tokens if condition == "oracle" else INPUT_TOKEN_GUARD
            if estimate > guard:
                msg = f"{row['question_id']} / {condition}: input {estimate} exceeds configured guard {guard}"
                if condition == "oracle":
                    skipped.append(msg)
                    print(f"SKIP {msg}; adjust --oracle-max-input-tokens after checking model context/cost", file=sys.stderr)
                    continue
                raise ValueError(msg)
            jobs.append((row, condition, selected, prompt, estimate, coverage_info(row, selected)))

    if skipped:
        print(f"Skipped {len(skipped)} over-limit Oracle cases", file=sys.stderr)
    if not args.run:
        print(f"FREE PREVIEW: {len(rows)} questions, {len(jobs)} eligible calls; "
              f"model={MODEL} effort={args.effort}; NO API request")
        for row, condition, selected, prompt, estimate, info in jobs:
            selected_ids = [s["id"] for s in selected]
            print(f"\nID={row['question_id']} TYPE={row['question_type']} CONDITION={condition} "
                  f"EFFORT={args.effort}")
            print(f"Selected IDs (EVALUATION LOG ONLY): {selected_ids}")
            print(f"Coverage={info['evidence_coverage']} "
                  f"({info['evidence_found']}/{info['evidence_total']})")
            hist_n = history_tokens(selected, enc)
            if condition == "oracle" and hist_n > HISTORY_TOKEN_BUDGET:
                print(f"WARNING: Oracle exceeds BM25 history budget by {hist_n-HISTORY_TOKEN_BUDGET} tokens")
            if args.summary_only:
                print(f"History tokens={hist_n} input estimate={estimate}")
            else:
                print(f"History tokens={hist_n} "
                      f"input estimate={estimate}\n--- EXACT MODEL INPUT ---\n{prompt}\n--- END ---")
        print("\n--- RETRIEVAL DIAGNOSTIC SUMMARY (gold IDs used only for offline evaluation) ---")
        for condition in dict.fromkeys(args.conditions):
            infos = [info for _, c, _, _, _, info in jobs if c == condition]
            counts = Counter(x["evidence_coverage"] for x in infos)
            recalls = [x["evidence_recall"] for x in infos if x["evidence_recall"] is not None]
            mean_recall = (sum(recalls) / len(recalls)) if recalls else None
            print(f"{condition}: {len(infos)} questions | all={counts['all']} "
                  f"some={counts['some']} none={counts['none']} "
                  f"unknown={counts['unknown']} "
                  f"mean gold-session recall={mean_recall if mean_recall is not None else 'n/a'}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is missing; set it privately in the terminal")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install dependencies: py -m pip install openai tiktoken") from exc
    client = OpenAI(max_retries=0, timeout=60.0)
    attempts_path = args.results.with_name(args.results.stem + ".attempts.jsonl")
    completed = {key_for_record(x) for x in read_jsonl(args.results)}
    started = {key_for_record(x) for x in read_jsonl(attempts_path)}
    count = 0
    for row, condition, selected, prompt, estimate, info in jobs:
        prompt_version = prompt_version_for_condition(condition, args.hybrid_backend)
        retriever = retrievers.get(backend_for_condition(condition, args.hybrid_backend))
        job_key = key(row["question_id"], condition, MODEL, prompt_version, args.effort)
        if job_key in completed:
            print(f"SKIP saved: {row['question_id']} / {condition} / {args.effort}")
            continue
        if job_key in started and not args.retry_uncertain:
            print(f"SKIP uncertain attempt: {row['question_id']} / {condition}. "
                  "Review API usage before retrying.")
            continue
        if count >= args.max_calls:
            break
        record = {
            "question_id": row["question_id"], "question_type": row["question_type"],
            "condition": condition, "model": MODEL, "effort": args.effort,
            "prompt_version": prompt_version,
            "retriever_model": retriever.model_name if retriever is not None else None,
            "retriever_backend": retriever.backend if retriever is not None else None,
            "hybrid_rrf_k": HYBRID_RRF_K if condition == "hybrid" else None,
            "retrieval_chunk_tokens": (RETRIEVAL_CHUNK_TOKENS if retriever is not None else None),
            "retrieval_chunk_overlap": (RETRIEVAL_CHUNK_OVERLAP if retriever is not None else None),
            "history_budget": HISTORY_TOKEN_BUDGET, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "selected_session_ids": [s["id"] for s in selected],
            "history_tokens_estimate": history_tokens(selected, enc),
            "oracle_over_history_budget": (
                condition == "oracle" and history_tokens(selected, enc) > HISTORY_TOKEN_BUDGET
            ),
            "local_input_token_estimate": estimate,
            "input_guard": (args.oracle_max_input_tokens if condition == "oracle" else INPUT_TOKEN_GUARD),
            **info,
        }
        append_jsonl(attempts_path, record)
        started.add(job_key)
        count += 1
        try:
            response = client.responses.create(
                model=MODEL, reasoning={"effort": args.effort},
                max_output_tokens=MAX_OUTPUT_TOKENS, input=prompt,
            )
        except Exception as exc:
            print(f"STOP: {type(exc).__name__}: {exc}; attempt logged.", file=sys.stderr)
            break
        record.update({
            "status": response.status, "response_id": response.id,
            "hypothesis": response.output_text or "",
            "input_tokens": response.usage.input_tokens if response.usage else None,
            "output_tokens": response.usage.output_tokens if response.usage else None,
            "correct": None,  # Fill manually after comparison with reference answer.
        })
        append_jsonl(args.results, record)
        completed.add(job_key)
        print(f"SAVED {row['question_id']} / {condition} / {args.effort} | "
              f"{response.status} | {record['hypothesis']!r}")
        if response.status != "completed" or not record["hypothesis"].strip():
            print("STOP: incomplete/empty answer; review usage and result.", file=sys.stderr)
            break
        if record["input_tokens"] is not None and record["input_tokens"] > record["input_guard"]:
            print("STOP: actual input tokens exceeded configured guard.", file=sys.stderr)
            break
    print(f"New paid requests started: {count}. Results: {args.results}")


if __name__ == "__main__":
    main()
