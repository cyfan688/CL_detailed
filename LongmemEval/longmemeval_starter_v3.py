"""LongMemEval-S diagnostic runner: sanitized IDs, BM25, evidence coverage and Oracle.

Examples (Windows PowerShell; dataset beside this file):
    py longmemeval_starter_v3_checked.py --self-test
    py -X utf8 longmemeval_starter_v3_checked.py --data longmemeval_s_cleaned.json --conditions bm25 --effort none
    py -X utf8 longmemeval_starter_v3_checked.py --data longmemeval_s_cleaned.json --conditions bm25 --effort none --run --max-calls 25 --results bm25_none_v2.jsonl
    py -X utf8 longmemeval_starter_v3_checked.py --data longmemeval_s_cleaned.json --conditions bm25 --effort medium --run --max-calls 25 --results bm25_medium_v2.jsonl
    py -X utf8 longmemeval_starter_v3_checked.py --data longmemeval_s_cleaned.json --conditions oracle --ids REAL_FAILED_QID1 REAL_FAILED_QID2 --effort none

No --run => print FREE preview, do not contact the API. --ids overrides the fixed 25 question IDs.
Use Oracle only on BM25 failures identified after manual grading. Annotation IDs and
reference answers are for offline evaluation only and never enter the LLM prompt.
"""

import argparse
import csv
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
CONDITIONS = ("no_history", "recent", "bm25", "oracle")
PROMPT_VERSION = "starter-v3-sanitized-ids-oracle-diagnostic"
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
        lines = [f"{date} | session {i:04d}"]
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


def select_sessions(row, condition, enc):
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
    for condition in CONDITIONS:
        selected = select_sessions(row, condition, enc)
        prompt = make_prompt(row, selected)
        assert "answer_new_2" not in prompt
        assert "sharegpt_old" not in prompt
        assert "has_answer" not in prompt and "answer_session_ids" not in prompt
        assert ("Seattle" not in prompt) == (condition == "no_history")
        if condition == "oracle":
            assert [s["id"] for s in selected] == ["answer_new_2"]
            assert coverage_info(row, selected)["evidence_coverage"] == "all"
    many = {**row, "answer_session_ids": ["answer_new_2", "sharegpt_old"]}
    assert coverage_info(many, [sessions[0]])["evidence_coverage"] == "some"
    assert coverage_info(row, [sessions[0]])["evidence_coverage"] == "none"
    assert coverage_info(many, sessions)["evidence_coverage"] == "all"
    assert key("q", "bm25", MODEL, PROMPT_VERSION, "none") != key(
        "q", "bm25", MODEL, PROMPT_VERSION, "medium"
    )
    print("PASS: sanitized prompts, chronological selection, oracle, coverage, effort keys; NO API calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--ids", nargs="+", help="Explicit question IDs; overrides the default fixed 25")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["bm25"])
    parser.add_argument("--effort", choices=["none", "medium"], default="none")
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

    jobs = []
    skipped = []
    for row in rows:
        for condition in dict.fromkeys(args.conditions):
            selected = select_sessions(row, condition, enc)
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
            print(f"History tokens={hist_n} "
                  f"input estimate={estimate}\n--- EXACT MODEL INPUT ---\n{prompt}\n--- END ---")
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
        job_key = key(row["question_id"], condition, MODEL, PROMPT_VERSION, args.effort)
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
            "prompt_version": PROMPT_VERSION,
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
