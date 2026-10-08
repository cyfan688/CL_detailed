"""A cautious first runner for the cleaned LongMemEval-S dataset.

Windows PowerShell, from the folder containing this script and the dataset:
    py -m pip install openai tiktoken
    py longmemeval_starter.py --self-test
    py longmemeval_starter.py --data longmemeval_s_cleaned.json
    # Inspect the printed prompt and selected session IDs before spending money.
    # Set OPENAI_API_KEY privately in your terminal, never in this file.
    py longmemeval_starter.py --data longmemeval_s_cleaned.json --run --conditions bm25 --max-calls 1
    py longmemeval_starter.py --data longmemeval_s_cleaned.json --run --max-calls 15

The default is a FREE local preview. --run is the only path that calls the API.
The first five examples are for debugging, not a formal 20-30-question study.
Use --ids ID1 ID2 ... to pick your own question IDs after inspecting the data.
Results are append-only in results.jsonl; never put your API key in a file.
"""

import argparse
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
CONDITIONS = ("no_history", "recent", "bm25")
PROMPT_VERSION = "starter-v1"
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenizer():
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("Install dependencies first: py -m pip install openai tiktoken") from exc
    # Local estimate, not a billing count. Record actual usage after every call.
    return tiktoken.get_encoding("o200k_base")


def words(text):
    return WORD_RE.findall(text.lower())


def load_data(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    # query whether is a list of dictionaries.
    if not isinstance(data, list):
        raise ValueError("Expected the cleaned LongMemEval-S JSON to be a list")
    ids = [row["question_id"] for row in data]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question_id values in dataset")
    return data


def choose_questions(data, ids):
    if ids:
        index = {row["question_id"]: row for row in data}
        missing = set(ids) - set(index)
        if missing:
            raise ValueError(f"Unknown question IDs: {sorted(missing)}")
        if len(ids) != len(set(ids)) or len(ids) > 30:
            raise ValueError("Supply 1-30 distinct question IDs")
        return [index[question_id] for question_id in ids]
    types = ("single-session-user", "single-session-assistant",
             "knowledge-update", "temporal-reasoning", "multi-session")
    chosen = []
    for kind in types:
        match = next((row for row in data if row.get("question_type") == kind
                      and not str(row["question_id"]).endswith("_abs")), None)
        if match is not None:
            chosen.append(match)
    if len(chosen) != 5:
        raise ValueError("Could not pick five types automatically; supply --ids")
    return chosen


def sessions_for(row):
    dates = row["haystack_dates"]
    ids = row["haystack_session_ids"]
    sessions = row["haystack_sessions"]
    if not (len(dates) == len(ids) == len(sessions)):
        raise ValueError(f"Unaligned sessions for {row['question_id']}")
    result = []
    for date, sid, turns in zip(dates, ids, sessions):
        # Deliberately serialize ONLY role and content; never has_answer.
        lines = [f"{str(date)} | session {sid}"]
        for turn in turns:
            if turn.get("role") not in ("user", "assistant"):
                raise ValueError(f"Unexpected role in {row['question_id']}")
            lines.append(f"{turn['role']}: {turn['content']}")
        result.append({"id": sid, "date": str(date), "text": "\n".join(lines)})
    return sorted(result, key=lambda session: session["date"])


def bm25_order(question, sessions):
    #parameters
    k1 = 1.2
    b = 0.75

    #use set() to remove duplicates in question
    query = set(words(question))

    #don't remove duplicates in documents because we are going to count the number of vocabulary occurrences.
    docs = [words(session["text"]) for session in sessions]
    lengths = [len(doc) for doc in docs]

    #prevent zero division.
    avg = sum(lengths) / max(len(docs), 1)
    df = Counter(term for doc in docs for term in set(doc))

    def score(i):
        counts = Counter(docs[i])
        total = 0.0
        for term in query:
            count = counts[term]
            if count:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                total += idf * count * (k1 + 1) / (count + k1 * (b + (1 - b) * lengths[i] / max(avg, 1)))
        return total

    return sorted(range(len(sessions)), key=lambda i: (-score(i), i))


def select_sessions(row, condition, enc):
    if condition == "no_history":
        return []
    sessions = sessions_for(row)
    order = list(reversed(range(len(sessions)))) if condition == "recent" else bm25_order(row["question"], sessions)
    selected = []
    remaining = HISTORY_TOKEN_BUDGET
    for index in order:
        size = len(enc.encode(sessions[index]["text"]))
        if size <= remaining:
            selected.append(index)
            remaining -= size
    # Present selected sessions chronologically for update and temporal questions.
    return [sessions[i] for i in sorted(selected)]


def make_prompt(row, selected):
    if selected:
        history = "\n\n".join(session["text"] for session in selected)
    else:
        history = "(No chat history provided)"
    return (
        "Answer the question using only the chat history below. "
        "If the history does not contain enough information, say 'I don't know'. "
        "Give a concise answer.\n\n"
        f"Question date: {row.get('question_date', 'unknown')}\n"
        f"Chat history:\n{history}\n\nQuestion: {row['question']}"
    )


def append_jsonl(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def key(qid, condition, model , prompt_version):
    return (str(qid), condition, model, prompt_version)


def self_test():
    class TestTokenizer:
        def encode(self, text):
            return words(text)

    enc = TestTokenizer()  # No package install or API access required.
    row = {
        "question_id": "local-test", "question_type": "knowledge-update",
        "question": "Where did Mira move in 2024?", "question_date": "2025-01-01",
        "answer": "Seattle", "answer_session_ids": ["new"],
        "haystack_dates": ["2022-01-01", "2024-01-01"],
        "haystack_session_ids": ["old", "new"],
        "haystack_sessions": [
            [{"role": "user", "content": "Mira lived in Boston."}],
            [{"role": "user", "content": "Mira moved to Seattle in 2024.", "has_answer": True}],
        ],
    }
    for condition in CONDITIONS:
        selected = select_sessions(row, condition, enc)
        prompt = make_prompt(row, selected)
        assert "has_answer" not in prompt and "answer_session_ids" not in prompt
        assert ("Seattle" not in prompt) == (condition == "no_history")
        if condition in ("recent", "bm25"):
            assert "new" in [session["id"] for session in selected]
    print("Local self-test passed: selection, chronology, and label exclusion. No API used.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--ids", nargs="+", help="Explicit question IDs; default: five different types")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--results", type=Path, default=Path("results.jsonl"))
    parser.add_argument("--max-calls", type=int, default=1, help="Maximum NEW API calls this execution may start")
    parser.add_argument("--run", action="store_true", help="Spend API credits; absent means free preview")
    parser.add_argument("--retry-uncertain", action="store_true", help="Explicitly repeat a logged attempt with no result")
    parser.add_argument("--self-test", action="store_true", help="Local synthetic test; no data or API needed")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.data or not args.data.is_file():
        parser.error("Give an existing --data path (or use --self-test)")
    if not 1 <= args.max_calls <= 105:
        parser.error("--max-calls must be 1..105")
    enc = tokenizer()
    rows = choose_questions(load_data(args.data), args.ids)
    jobs = []
    for row in rows:
        for condition in dict.fromkeys(args.conditions):
            selected = select_sessions(row, condition, enc)
            prompt = make_prompt(row, selected)
            estimated_input = len(enc.encode(prompt))
            if estimated_input > INPUT_TOKEN_GUARD:
                raise ValueError(f"Input estimate {estimated_input} exceeds guard for {row['question_id']}")
            jobs.append((row, condition, selected, prompt, estimated_input))

    if not args.run:
        print(f"FREE PREVIEW: {len(rows)} questions, {len(jobs)} planned calls; model={MODEL}; NO API request.")
        for row, condition, selected, prompt, estimate in jobs:
            evidence = set(row.get("answer_session_ids", []))
            selected_ids = [item["id"] for item in selected]
            print(f"\nID={row['question_id']} TYPE={row['question_type']} CONDITION={condition}")
            print(f"Selected session IDs: {selected_ids} | Evidence among selected: {bool(evidence.intersection(selected_ids))}")
            print(f"Local input token estimate: {estimate}\n--- EXACT MODEL INPUT ---\n{prompt}\n--- END ---")
        return

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is missing; do not paste the key into this script")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install dependencies first: py -m pip install openai tiktoken") from exc
    client = OpenAI(max_retries=0, timeout=60.0)
    attempts_path = args.results.with_name(args.results.stem + ".attempts.jsonl")
    completed = {key(x["question_id"], x["condition"], x["model"], x["prompt_version"]) for x in read_jsonl(args.results)}
    started = {key(x["question_id"], x["condition"], x["model"], x["prompt_version"]) for x in read_jsonl(attempts_path)}
    count = 0
    for row, condition, selected, prompt, estimate in jobs:
        job_key = key(row["question_id"], condition, MODEL, PROMPT_VERSION)
        if job_key in completed:
            print(f"SKIP saved result: {row['question_id']} / {condition}")
            continue
        if job_key in started and not args.retry_uncertain:
            print(f"SKIP uncertain prior attempt: {row['question_id']} / {condition}. Inspect Usage before retrying.")
            continue
        if count >= args.max_calls:
            break
        record = {"question_id": row["question_id"], "question_type": row["question_type"],
                  "condition": condition, "model": MODEL, "prompt_version": PROMPT_VERSION,
                  "selected_session_ids": [s["id"] for s in selected],
                  "local_input_token_estimate": estimate}
        append_jsonl(attempts_path, record)  # Persist before any paid request.
        count += 1
        try:
            response = client.responses.create(
                model=MODEL, reasoning={"effort": "none"},
                max_output_tokens=MAX_OUTPUT_TOKENS, input=prompt,
            )
        except Exception as exc:
            print(f"STOP: {type(exc).__name__}: {exc}. Attempt logged. Do not retry automatically.", file=sys.stderr)
            break
        record.update({"status": response.status, "response_id": response.id,
                       "hypothesis": response.output_text or "",
                       "input_tokens": response.usage.input_tokens if response.usage else None,
                       "output_tokens": response.usage.output_tokens if response.usage else None})
        append_jsonl(args.results, record)
        completed.add(job_key)
        print(f"SAVED {row['question_id']} / {condition} | {response.status} | "
              f"tokens in={record['input_tokens']} out={record['output_tokens']} | {record['hypothesis']!r}")
        if response.status != "completed" or not record["hypothesis"].strip():
            print("STOP: incomplete or empty answer. Review results and usage.")
            break
        if record["input_tokens"] is not None and record["input_tokens"] > INPUT_TOKEN_GUARD:
            print("STOP: actual input token count exceeded local estimate guard.")
            break
    print(f"New paid requests started in this run: {count}. Results: {args.results}")


if __name__ == "__main__":
    main()
