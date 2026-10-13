# LongMemEval Evaluation

A small evaluation script for comparing three chat-history settings:

- `no_history`: answer without chat history.
- `recent`: prioritize the most recent sessions.
- `bm25`: prioritize sessions using BM25 relevance scores.

Selected sessions are presented in chronological order. The script supports local previews, API calls, and incremental result saving.

## 1. Download the dataset

The dataset is not included in this repository due to its size.

Download [longmemeval_s_cleaned.json](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/main/longmemeval_s_cleaned.json) and place it in the same directory as `longmemeval_starter.py`.

The commands below assume that PowerShell is open in that directory.

## 2. Install dependencies

```powershell
py -m pip install openai tiktoken
```

If you use a virtual environment, use its Python executable consistently for both installation and execution.

## 3. Run the local self-test

```powershell
py longmemeval_starter.py --self-test
```

This checks basic history selection and prompt construction using synthetic data. It does not call the API or evaluate model accuracy.

## 4. Generate a preview

```powershell
py -X utf8 longmemeval_starter.py --data longmemeval_s_cleaned.json > preview.txt
```

The preview includes:

- Question IDs, question types, and conditions.
- Selected session IDs.
- Whether at least one annotated evidence session was selected.
- Estimated input token counts.
- The prompt text prepared for each task.

No API key is required for preview mode.

`Evidence among selected: True` does not guarantee that all required evidence was retrieved or that the model will answer correctly.

Running this command again overwrites `preview.txt`.

## 5. Set the OpenAI API key

Enter the key using a hidden PowerShell prompt:

```powershell
$secureKey = Read-Host "Paste OpenAI API Key" -AsSecureString
$keyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)

try {
    $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPtr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPtr)
}
```

The environment variable is set for this PowerShell session and its child processes. A new terminal session may require setting it again.

Do not put the API key in source code or commit it to Git.

Check whether the key is set without displaying it:

```powershell
if ($env:OPENAI_API_KEY) {
    "API key is set."
}
else {
    "API key is not set."
}
```

## 6. Run the evaluation

Check the `MODEL` setting in the Python script before running.

Run all three conditions for the selected questions:

```powershell
py -X utf8 longmemeval_starter.py --data longmemeval_s_cleaned.json --run --max-calls 15 --results debug_results.jsonl
```

Important:

By default, the script selects five questions from each of five question types, giving 25 questions.

Each question is evaluated under three conditions, giving 75 tasks in total. To run a separate five-question debugging set, supply five question IDs with `--ids`.

`--max-calls 15` allows at most 15 new API requests in this execution. Tasks with saved results are skipped.

To select specific questions, replace the example IDs below with real dataset IDs:

```powershell
py -X utf8 longmemeval_starter.py --data longmemeval_s_cleaned.json --ids REAL_ID_1 REAL_ID_2 --run --max-calls 6 --results custom_results.jsonl
```

## 7. Output files and resuming

For `--results debug_results.jsonl`, the script uses:

| File | Contents |
|---|---|
| `debug_results.jsonl` | Model response records and token usage |
| `debug_results.attempts.jsonl` | Records written before API requests |
| `preview.txt` | Preview output, when explicitly redirected |

Results are appended rather than overwritten.

To continue an interrupted experiment, use the same question selection, settings, and results path.

An attempt with no saved result is skipped by default because the request may already have reached the API. Inspect the records and API usage before using `--retry-uncertain`.

Use a separate results file when changing the model, history budget, prompt, or retrieval implementation to avoid mixing experiments.

## 8. Display model answers

```powershell
Get-Content .\debug_results.jsonl -Encoding utf8 |
    ForEach-Object { $_ | ConvertFrom-Json } |
    Sort-Object question_id, condition |
    Select-Object question_id, question_type, condition, hypothesis |
    Format-Table -Wrap
```

This sorts the displayed rows by question ID and condition. It does not modify the results file.

`hypothesis` is the model-generated answer.

## 9. Display reference answers

This reads the question IDs from the results file and looks up their questions and reference answers in the dataset.

```powershell
@'
import json

with open("longmemeval_s_cleaned.json", encoding="utf-8") as file:
    dataset = json.load(file)

index = {row["question_id"]: row for row in dataset}

with open("debug_results.jsonl", encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

ids = list(dict.fromkeys(record["question_id"] for record in records))

for qid in ids:
    row = index[qid]
    print("ID:", qid)
    print("Question:", row["question"])
    print("Reference:", row["answer"])
    print()
'@ | py -X utf8 -
```

This displays reference answers for manual review; it does not automatically grade model responses.

## 10. Remove the API key from the current session

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

This removes the environment variable from the current PowerShell session. It does not revoke the key on the API platform.

## 11. Compare output token usage

Run the following code using `results_25.jsonl`, `results_25_luna_medium.jsonl`, and `results_25_sol.jsonl` to compare output token usage across the three settings.
```
@'
import json
from pathlib import Path

files = {
    "Luna none": Path("results_25.jsonl"),
    "Luna medium": Path("results_25_luna_medium.jsonl"),
    "Sol medium": Path("results_25_sol.jsonl")
}

groups = {}

for name, path in files.items():
    with path.open(encoding="utf-8-sig") as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    rows = [r for r in rows if r["condition"] == "bm25"]
    ids = [r["question_id"] for r in rows]

    if len(ids) != len(set(ids)):
        raise SystemExit(f"{name}: duplicate question IDs found.")
    if any(r.get("status") != "completed" for r in rows):
        raise SystemExit(f"{name}: unfinished responses found.")

    groups[name] = {r["question_id"]: r for r in rows}

none = groups["Luna none"]
medium1 = groups["Luna medium"]
medium2 = groups["Sol medium"]

if set(none) != set(medium1) or set(none) != set(medium2) or not none:
    raise SystemExit("The three files do not contain the same nonempty question set.")

totals = {}
for name, rows in groups.items():
    values = [r["output_tokens"] for r in rows.values()]
    totals[name] = sum(values)
    print(f"{name}:")
    print(f"  Questions: {len(values)}")
    print(f"  Total output tokens: {sum(values):,}")
    print(f"  Mean output tokens: {sum(values)/len(values):.2f}")
    print(f"  Range: {min(values)}--{max(values)}")

a = totals["Luna none"]
b = totals["Luna medium"]
c = totals["Sol medium"]

print(f"\nAdditional output tokens(Luna): {b-a:+,}")
print(f"\nAdditional output tokens(Sol): {c-a:+,}")

if a:
    print(f"Luna-Medium / Luna-none: {b/a:.2f}x")
    print(f"Increase: {(b-a)/a*100:.2f}%")
    print(f"Sol-Medium / Luna-none: {c/a:.2f}x")
    print(f"Increase: {(c-a)/a*100:.2f}%")

'@ | python -
```


## Notes

- File names and paths can be changed, but all corresponding commands must be updated.
- History selection uses whole sessions. A session that exceeds the remaining budget is skipped.
- The history token budget does not include all instructions and question text.
- Local token counts are estimates; API-reported usage is saved separately.
