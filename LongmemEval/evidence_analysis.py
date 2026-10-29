import json
from collections import Counter
from statistics import mean

files = ["bm25_none.jsonl", "bm25_medium.jsonl","oracle_medium.jsonl","oracle_none.jsonl"]

for filename in files:
    with open(filename, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Only completed BM25 runs; deduplicate question IDs
    rows = {
        r["question_id"]: r
        for r in records
        if r["condition"] == "bm25" or r ["condition"] == "oracle" and r["status"] == "completed"
    }

    n = len(rows)
    counts = Counter( r["evidence_coverage"] for r in rows.values())

    correctness = sum(r["correct"] for r in rows.values() )
    recalls = [
        r["evidence_recall"]
        for r in rows.values()
        if r["evidence_recall"] is not None
    ]

    print(f"\n{filename} (N={n})")

    for label in ("all", "some", "none"):
        count = counts[label]
        print(f"{label}: {count}/{n} ({count/n:.1%})")

    print(f"Mean Evidence Recall: {mean(recalls):.4f}")

    print(f"Correctness: {correctness}/{n} ({correctness/n:.1%})")