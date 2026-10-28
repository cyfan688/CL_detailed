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


---

## 12. Sanitized BM25 and Oracle diagnostics (follow-up study)

**Status:** These commands use the newer diagnostic runner, \`longmemeval_starter_v4_duplicate_safe.py\`, which is separate from the original \`longmemeval_starter.py\` described above. The new runner and \`evidence_analysis.py\` are prepared locally and will be uploaded separately. Until those files are uploaded, their commands will not work on a fresh clone.

### Why the runner changed

The original history formatter included raw session IDs such as \`answer_...\` in both the BM25 retrieval text and the model prompt. Because those IDs can indicate answer-bearing evidence, the newer runner replaces them with neutral sequential session numbers. Original IDs are retained only in evaluation metadata/logs, where they are needed for evidence coverage and Oracle selection.

The follow-up experiment uses the **same fixed 25 questions** as the earlier study, with a **BM25 history budget of 8,000 tokens**. For every question, it records whether the retrieved history contains **all**, **some**, or **none** of the annotated supporting sessions, as well as the per-question evidence recall.

### Run the sanitized BM25 comparison

From PowerShell in \`LongmemEval\` (with the new runner and dataset present):

\`\`\`powershell
py -m pip install openai tiktoken
py .\longmemeval_starter_v4_duplicate_safe.py --self-test

# Free prompt preview (does not call the API)
py -X utf8 .\longmemeval_starter_v4_duplicate_safe.py --data .\longmemeval_s_cleaned.json --conditions bm25 --effort none > preview_bm25.txt

# Paid experiments, 25 calls per effort (requires OPENAI_API_KEY)
py -X utf8 .\longmemeval_starter_v4_duplicate_safe.py --data .\longmemeval_s_cleaned.json --conditions bm25 --effort none --run --max-calls 25 --results bm25_none.jsonl
py -X utf8 .\longmemeval_starter_v4_duplicate_safe.py --data .\longmemeval_s_cleaned.json --conditions bm25 --effort medium --run --max-calls 25 --results bm25_medium.jsonl
\`\`\`

Use separate result files for none and medium. Results and attempt logs are append-only; completed jobs are skipped when rerunning with the same settings. Manual correctness grading is still required: \`status="completed"\` alone does not mean the answer is correct.

### Oracle: supply annotated supporting sessions

Use the \`oracle\` condition on **previously identified failed question IDs**. The model receives the annotated supporting session texts, not the gold reference answer or annotation markers. For example, substitute your own failed IDs:

\`\`\`powershell
# Free Oracle preview
py -X utf8 .\longmemeval_starter_v4_duplicate_safe.py --data .\longmemeval_s_cleaned.json --conditions oracle --ids 3ba21379 gpt4_8279ba03 --effort medium --oracle-max-input-tokens 18000 > preview_oracle.txt

# Paid Oracle requests
py -X utf8 .\longmemeval_starter_v4_duplicate_safe.py --data .\longmemeval_s_cleaned.json --conditions oracle --ids 3ba21379 gpt4_8279ba03 --effort medium --oracle-max-input-tokens 18000 --run --max-calls 2 --results oracle_medium.jsonl
\`\`\`

**Configurable Oracle input limit:** the default full-prompt guard is 9,000 estimated input tokens. Pass \`--oracle-max-input-tokens 18000\` (or another checked value) when all annotated evidence would exceed that guard. **This changes only the Oracle full-input safety threshold; it does not change the 8,000-token BM25 history budget.** Verify the model's supported context window and anticipated cost before increasing the limit. Inspect skipped cases in preview. Some Oracle histories exceed 8,000 tokens; consequently Oracle is a diagnostic **not a strict equal-budget accuracy comparison**. Oracle also removes distractor sessions.

### Summarize evidence coverage, recall, and correctness

The separately uploaded \`evidence_analysis.py\` is an **offline, standard-library-only** result summarizer. Put it beside the JSONL files (or pass explicit paths):

\`\`\`powershell
py .\evidence_analysis.py
py .\evidence_analysis.py bm25_none.jsonl bm25_medium.jsonl
py .\evidence_analysis.py oracle_none.jsonl oracle_medium.jsonl
\`\`\`

The updated script reads files relative to its own directory when needed, selects only completed BM25/Oracle records, checks duplicate question IDs, and prints the **all/some/none** distribution and **mean per-question evidence recall (macro recall)**. If every result includes a manually assigned \`correct\` value of 0/1, it also reports accuracy. If labels are missing or null, it reports that correctness is not fully graded rather than pretending the run's accuracy is zero.

In this 25-question follow-up, BM25 found all supporting sessions for 19 questions, some for 3, none for 3; mean evidence recall was **0.816**. Strict benchmark accuracy was **13/25 (none)** and **18/25 (medium)**. Six medium-effort BM25 failures with incomplete evidence were then tested with Oracle: **4/6** were correct with medium and **2/6** with none under strict reference scoring. Two of the remaining medium Oracle failures appear to involve ambiguous support/reference answers; report this separately from the benchmark score.
