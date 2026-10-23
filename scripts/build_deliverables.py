"""Generate deliverables/ from the saved results, so every brief deliverable is a standalone file
whose numbers always match the experiments. Run from the repo root:  python scripts/build_deliverables.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
from src.prompts import SYSTEM_PROMPTS, PROMPT_RATIONALES, build_prompt   # noqa: E402
from src.schema import JSON_SCHEMA                                                # noqa: E402
from src import tools                                                             # noqa: E402
from src.evaluate import score_rows                                               # noqa: E402

OUT = "deliverables"
os.makedirs(OUT, exist_ok=True)
rd = lambda p: pd.read_csv(p, keep_default_na=False)


def table(df, index=True, floatfmt="{:.1f}"):
    """Markdown table without extra dependencies."""
    d = df.reset_index() if index else df.copy()
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def write(name, text):
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    print("wrote", name)


METRIC_COLS = ["F1_overall", "EM_overall", "F1_answerable", "hallucination_rate", "over_abstention_rate",
               "evidence_faithfulness", "json_validity"]

# ---------------------------------------------------------------- 1. dataset
ev = pd.read_json("data/eval_set.jsonl", lines=True)
fs = pd.read_json("data/fewshot_pool.jsonl", lines=True)
stats = pd.read_csv("report/dataset_stats.csv", index_col=0)
samples = pd.concat([ev[ev.is_answerable].head(2), ev[~ev.is_answerable].head(2)])
write("01_dataset_and_preprocessing.md", f"""
# Dataset description and preprocessing record

**Dataset:** SQuAD v2.0 (`rajpurkar/squad_v2` on Hugging Face) — crowd-sourced questions on Wikipedia paragraphs.
Answerable questions have a gold answer that is an exact span of the paragraph; **unanswerable** questions were written
adversarially to look answerable but have no answer in the paragraph (the correct output is to abstain).

**Fields used:** `id`, `title` (Wikipedia article), `context` (paragraph), `question`, `answers.text` (gold answers,
possibly several), `answers.answer_start`. Derived: `is_answerable` = at least one gold answer.

## Subset used for all experiments (frozen)
| Item | Value |
|---|---|
| Source split | validation (11,873 questions, 35 articles) |
| Evaluation set | {len(ev)} questions: {int(ev.is_answerable.sum())} answerable + {int((~ev.is_answerable).sum())} unanswerable |
| Articles covered | {ev.title.nunique()} (all validation articles) |
| Few-shot examples | {len(fs)} from the **train** split ({int(fs.is_answerable.sum())} answerable + {int((~fs.is_answerable).sum())} unanswerable) |
| Random seed | 42 |
| Files | `data/eval_set.jsonl`, `data/fewshot_pool.jsonl` |

## Preprocessing steps (notebook 01)
1. Load the validation split; mark each question answerable / unanswerable.
2. To spread questions across articles, keep at most 3 questions per article per type, then sample 100 answerable and
   100 unanswerable questions (seed 42) and shuffle. The file is then **frozen**: every experiment uses these 200 questions.
3. Few-shot pool: from the train split, paragraphs shorter than 600 characters; 3 answerable + 1 unanswerable (seed 42).
   Evidence = the sentence containing the gold answer, found from `answer_start`.
4. Leakage check: train and validation use different Wikipedia articles, so no few-shot paragraph or question can
   appear in the evaluation set (also asserted in code).
5. No text cleaning or truncation is applied: paragraphs are passed to the model as written.

## Size of the inputs (Qwen tokenizer)
{table(stats)}

## Representative examples
{table(samples[["question", "gold_answers", "is_answerable", "title"]].assign(gold_answers=lambda d: d.gold_answers.map(lambda g: sorted(set(g)) or "— (unanswerable)")), index=False)}

## Dataset challenges
- Unanswerable questions reuse words from the paragraph, so keyword matching produces confident wrong answers.
- Multiple valid gold answers per question — scoring takes the best match (Exact Match and F1, official normalisation).
- Single domain (Wikipedia); results may not transfer to other document types.
- The model may have seen Wikipedia/SQuAD in pre-training (possible contamination), so systems are compared on the same
  questions rather than by absolute accuracy.
""")

# ---------------------------------------------------------------- 2. prompts
cmp_ = pd.read_csv("results/summary_prompt_comparison.csv", index_col=0)
base = rd("results/exp_prompt_baseline.csv")
picks = pd.concat([base[base.is_answerable.astype(str) == "True"].head(1),
                   base[base.is_answerable.astype(str) == "False"].head(2)])
def render(variant):
    """The exact messages build_prompt() sends to the model (placeholders for the paragraph and question)."""
    msgs = build_prompt("<paragraph>", "<question>", variant=variant,
                        few_shot_examples=fs.to_dict("records") if variant == "few_shot" else None)
    if variant == "few_shot":     # show the instructions, ONE worked example pair, and the final query
        msgs = msgs[:3] + [{"role": "...", "content": f"({len(msgs) - 4} more example messages)"}] + msgs[-1:]
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)


BASELINE_WHY = ("Deliberately minimal: no format, no evidence, no rules. It shows what the raw model does, so every "
                "later improvement can be measured against it (requirement 5).")
prompt_sections = "\n\n".join(
    f"### {v}\n**Why:** {BASELINE_WHY if v == 'baseline' else PROMPT_RATIONALES.get(v, '')}\n\n```text\n{render(v)}\n```"
    for v in ["baseline", "zero_shot", "few_shot", "role_based", "constraint_based"])
write("02_prompt_experiment_log.md", f"""
# Prompt experiment log (E1 baseline, E2 prompt comparison)

All prompts are produced by one reusable template, `build_prompt(context, question, variant, few_shot_examples)` in
`src/prompts.py` (requirement 11). Model: Qwen2.5-1.5B-Instruct, greedy decoding, `max_new_tokens=128`,
`repetition_penalty=1.0`, the frozen 200 questions. Every raw output is saved in `results/exp_prompt_<variant>.csv`.

## The prompts (exact messages sent to the model) and the reason for each design
{prompt_sections}

The few-shot variant uses the zero-shot instructions plus 4 worked examples from the train split (3 answerable, 1
unanswerable) given as earlier user/assistant turns, so the difference between zero-shot and few-shot measures the
examples alone.

## E1 — three baseline outputs
{table(picks[["question", "gold_answers", "raw_output"]], index=False)}

## E2 — comparison on 200 questions (%)
{table(cmp_[METRIC_COLS + ["flag_conflicts"]])}

**Selection rule (stated before choosing):** among prompts with JSON validity ≥ 90%, the highest overall F1, ties broken
by lower hallucination → **few-shot** (`results/best_prompt.txt`). Few-shot vs baseline: +14.5 F1, 95% CI [7.0, 21.8],
McNemar p < 0.001 (notebook 07).
""")

# ---------------------------------------------------------------- 3. decoding
dec = pd.read_csv("results/summary_decoding_comparison.csv")
noise = pd.read_csv("results/summary_context_noise.csv", index_col=0)
ex = pd.read_csv("results/summary_decoding_examples.csv")
q0 = ex.question.iloc[0]
write("03_decoding_comparison.md", f"""
# Decoding comparison (E3)

Same prompt and 30 questions (15 answerable, 15 unanswerable); greedy run twice, every sampling setting 5 times with
different seeds. Qwen's hidden defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1) are switched off,
so each row changes only what its name says. Chart: `results/decoding_chart.png`.

{table(dec, index=False)}

*agreement* = share of runs giving the most common answer (100 = identical every run); *distinct_answers* = different
answers per question; *F1_std_across_runs* = run-to-run spread.

**Findings.** Greedy has the best answerable F1 and is perfectly stable. Higher temperature lowers F1 and agreement. The
lower hallucination of sampling comes with more refusals of answerable questions (over-abstention 20% → 37%), i.e. random
refusals rather than better judgement. Top-p 0.9 restores some stability at temperature 0.7. Top-k 50 gives results
identical to temperature 0.7 alone: with shared seeds, this shows the probability mass sits in far fewer than 50 tokens.
→ **Greedy decoding is used in the final system.**

## Output example — same question, different runs
*{q0}*

{table(ex[ex.question == q0][["decoding", "run", "pred_answer", "error_type"]], index=False)}

## Context noise (same 100 questions, greedy)
{table(noise[METRIC_COLS])}
""")

# ---------------------------------------------------------------- 4. schema
e4 = rd("results/E4_structured_output.csv")
e4s = pd.concat([e4[e4.is_answerable.astype(str) == "True"].head(3), e4[e4.is_answerable.astype(str) == "False"].head(2),
                 e4[pd.to_numeric(e4.attempts) > 1].head(2)])
samples_json = [{"question": r.question, "gold_answers": json.loads(r.gold_answers),
                 "attempts": int(r.attempts), "first_attempt_errors": json.loads(r.errors) if r.errors else [],
                 "validated_output": json.loads(r.raw_output) if r.json_valid in (True, "True") else r.raw_output}
                for r in e4s.itertuples()]
json.dump({"model_output_schema": JSON_SCHEMA,
           "validation_rules": {
               "R1": "reply contains a parseable JSON object", "R2": "answerable/answer/evidence present with the right types",
               "R3": "confidence, if present, within [0, 1]", "R4": "answerable=true -> answer and evidence non-empty",
               "R5": "answerable=true -> evidence occurs in the paragraph (final system: >= 90% character alignment)",
               "R6": "answerable=false -> answer empty ('not in the document' normalised to '')",
               "R7": "answerable=true -> answer is not a refusal phrase",
               "R8": "answer must not only repeat words of the question (ablation row D)"},
           "retry_policy": "on failure the model sees its own error list and answers again; if still invalid the system abstains",
           "confidence": "added by the system: geometric-mean probability of the decision tokens (src/confidence.py)"},
          open(f"{OUT}/04_output_schema.json", "w"), indent=2, ensure_ascii=False)
json.dump(samples_json, open(f"{OUT}/04_sample_validated_outputs.json", "w"), indent=2, ensure_ascii=False)
print("wrote 04_output_schema.json, 04_sample_validated_outputs.json")

# ---------------------------------------------------------------- 5. tool
spec = {k: v for k, v in tools.TOOL_SPEC.items()}
json.dump(spec, open(f"{OUT}/05_tool_spec.json", "w"), indent=2, ensure_ascii=False)
tm = rd("results/ablation/tool_mode_fixed.csv")
sm = score_rows(tm)
good = sm[(sm.error_type == "correct") & sm.is_answerable & sm.gold_retrieved.astype(str).eq("True")]
demo = good.loc[good.evidence.str.len().idxmin()]                 # clearest example: right paragraph, short evidence
tool_cmp = pd.read_csv("results/ablation/summary_tool_mode.csv", index_col=0)
write("05_tool_call_demo.md", f"""
# Tool / function calling (E5)

**Tool:** `retrieve_context(question, top_k)` — BM25 search over a library of Wikipedia paragraphs (20 SQuAD articles,
676 paragraphs), implemented in `src/tools.py`. Input and output schemas: `05_tool_spec.json`. Arguments are validated
with pydantic before execution (`top_k` limited to 1–5, query at least 3 characters).

**Workflow (brief §5):** user request → LLM → tool decision → tool result → final response.
1. The user asks a question **without** a paragraph.
2. The model receives the tool description and replies with a tool call:
   `{{"tool_call": {{"name": "retrieve_context", "arguments": {{"question": "...", "top_k": 3}}}}}}`
3. Python validates the arguments and runs BM25; the top passages are returned.
4. The final answer is written with the same few-shot prompt and validator as paragraph mode, and its evidence is
   checked against the retrieved passages.

## Representative invocation
- **User question:** {demo.question}
- **Model called the tool itself:** {demo.get("tool_called_by_model", "")}
- **Gold paragraph among retrieved passages:** {demo.get("gold_retrieved", "")}
- **Final response:** `{demo.raw_output}`

## Results on 112 questions from the indexed articles (%)
{table(tool_cmp[[c for c in ["F1_overall", "F1_answerable", "hallucination_rate", "over_abstention_rate", "evidence_faithfulness", "gold_paragraph_retrieved_%", "model_called_tool_%"] if c in tool_cmp]].fillna(""))}

The first row is the original tool mode (final answer written with a zero-shot tool prompt); the second uses the
few-shot prompt chosen in E2 for the final answer. The gap to the third row is the measured cost of retrieval
compared with being given the right paragraph.
""")

# ---------------------------------------------------------------- 6. evaluation + failures
abl = pd.read_csv("results/ablation/summary_ablation.csv", index_col=0)
ci = pd.read_csv("results/summary_confidence_intervals.csv", index_col=0)
paired = pd.read_csv("results/summary_paired_tests.csv", index_col=0)
fail = pd.read_csv("results/summary_failure_analysis.csv")
inj = pd.read_csv("results/ablation/summary_injection.csv")
write("06_evaluation_and_failure_analysis.md", f"""
# Evaluation and failure analysis

## Evaluation procedure
Frozen set of 200 SQuAD v2 validation questions (100 answerable, 100 unanswerable, seed 42), Qwen2.5-1.5B-Instruct,
greedy decoding, scored by `src/evaluate.py` (unit-tested). Criteria:

| Criterion | Measured on | Meaning |
|---|---|---|
| Exact Match / F1 | all | official SQuAD normalisation; best match over gold answers; correct abstention = 1 |
| Hallucination rate | unanswerable | answered although the paragraph has no answer |
| Over-abstention rate | answerable | refused although the answer was there |
| Evidence faithfulness | answered | evidence is in the paragraph and contains the answer |
| JSON validity | all | output parses and has the required keys |
| Flag conflicts | all | answerable flag contradicts the answer text |

## Baseline vs final system, with every intermediate step (ablation, %)
{table(abl[METRIC_COLS + ["mean_attempts"]])}

## Uncertainty (notebook 07, 5,000 bootstrap resamples)
{table(ci)}

{table(paired)}

Key results: the prompt change is the largest and only significant F1 gain. Harness components change F1 by amounts
that are not statistically detectable on 200 questions, while guaranteeing valid JSON and grounded evidence. The
1.5B model with the harness hallucinates on 20 trap questions versus 38 for the 3B model without it (McNemar p = 0.0003).

## Failure analysis (E6)
{table(fail, index=False)}

## Prompt-injection tests (responsible use)
{table(inj, index=False)}
""")

# ---------------------------------------------------------------- index
write("README.md", """
# Deliverables (brief §9)

Generated from the saved experiment results by `scripts/build_deliverables.py`, so every number here matches the
notebooks.

| Brief deliverable | File |
|---|---|
| Technical report | submitted separately (PDF) |
| Working Colab notebook | `notebooks/01`–`07` (run order in the main README) |
| Dataset description and preprocessing record | `01_dataset_and_preprocessing.md` |
| Prompt experiment log (baseline + improved prompts) | `02_prompt_experiment_log.md` (+ `results/exp_prompt_*.csv`, every output) |
| Decoding comparison table | `03_decoding_comparison.md` (+ `results/decoding_chart.png`) |
| Structured-output schema and sample validated outputs | `04_output_schema.json`, `04_sample_validated_outputs.json` |
| Tool/function-calling specification and demonstration | `05_tool_spec.json`, `05_tool_call_demo.md` |
| Evaluation and failure-analysis table | `06_evaluation_and_failure_analysis.md` |
| Group demonstration | presented live |
| Individual contribution statement | in the technical report |
""")
