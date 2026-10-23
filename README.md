# Answer-or-Abstain: a grounded question-answering assistant on SQuAD v2

**PCAM ZG516 Applied Generative AI · Group Major Project 1 · Group 7** (member contributions are listed in the technical report)

A user gives a document and a question. The assistant answers **only from the document**, returns the sentence it used
as **evidence** and a **confidence** score, and **abstains** when the answer is not there. With no document, it calls a
**retrieval tool** first. The model is a pretrained causal LM, **Qwen2.5-1.5B-Instruct**, used without training.

## Headline results (200 SQuAD v2 validation questions, 100 answerable / 100 unanswerable)

| System | F1 | Hallucination | Over-abstention | Valid JSON | Grounded evidence |
|---|---|---|---|---|---|
| Baseline prompt (E1) | 62.5 | 51% | 0% | 0% | — |
| Few-shot prompt (E2 winner) | 77.0 | 21% | 17% | 98% | 82% |
| **Final system** (few-shot + validator + retry + fuzzy evidence + injection guard) | **75.8** | **20%** | 21% | **100%** | 85% |
| Qwen2.5-**3B**, few-shot, no harness | 72.6 | 38% | 5% | 97% | 76% |

- The prompt change is the largest gain: **+14.5 F1** over the baseline (95% CI [7.0, 21.8], McNemar p < 0.001).
- The harness guarantees valid, grounded output and blocks the tested prompt injections, with **no statistically
  detectable change in F1** relative to the prompt alone (all 95% CIs include 0).
- The 1.5B model with the harness hallucinates on **20** trap questions versus **38** for the 3B model without it
  (McNemar p = 0.0003).

Full tables: `deliverables/06_evaluation_and_failure_analysis.md`; statistics: `notebooks/07_statistical_analysis.ipynb`.

## How the system works

```
 user question (+ paragraph?) ──► no paragraph? ──► LLM decides tool call ──► retrieve_context (BM25) ──┐
                │                                                                                       │
                ▼                                                                                       ▼
        GUIDE: injection guard (document text is data, injected instructions removed)  ◄────────────────┘
                ▼
        few-shot prompt (src/prompts.py) ──► tokenizer ──► Qwen2.5-1.5B-Instruct ──► greedy decoding
                ▼
        SENSOR: validator (src/schema.py + src/harness.py) — JSON, types, flag/answer consistency,
                evidence really in the paragraph (≥ 90% character alignment)
                ▼  fails → error list fed back to the model, one retry → still fails → abstain
        CONTROL: confidence = geometric-mean probability of the decision tokens (src/confidence.py)
                ▼
        {"answerable": true, "answer": "Kublai Khan", "evidence": "...established by Kublai Khan...", "confidence": 0.97}
```

## Repository layout

```
data/          eval_set.jsonl (frozen 200 test questions), fewshot_pool.jsonl (4 train-split examples)
src/           prompts.py     prompt variants + reusable template build_prompt()            (req. 6, 11)
               model.py       generate() / generate_with_logprobs() with explicit decoding settings
               evaluate.py    all metrics, shared by every experiment                      (req. 12)
               schema.py      output schema, validator R1-R7, retry                        (req. 9, E4)
               confidence.py  confidence from token probabilities, calibration/ECE
               tools.py       retrieve_context tool: schemas, BM25, tool-calling loop     (req. 10, E5)
               harness.py     injection guard, fuzzy evidence, echo check, threshold, tool mode
               runlog.py      shared CSV format, eval-set loader, cached runs
notebooks/     01 -> 07, run in order (table below)
results/       raw outputs of every experiment (exp_*.csv, E*.csv) and summary tables (summary_*.csv)
results/ablation/  harness ablation, 3B comparison, tool mode, injection tests
deliverables/  every brief deliverable as a standalone file (generated from results/)
report/        dataset statistics and figures
scripts/       build_deliverables.py, build_07.py
tests/         offline unit tests (no GPU): python -m unittest discover tests -v   (32 tests)
```

## How to run

Notebooks 01–06 need **Google Colab with a T4 GPU** (Runtime → Change runtime type → T4 GPU). Notebook 07 runs on any
CPU from the saved results.

| Notebook | Covers | Runtime |
|---|---|---|
| `01_data_foundations` | Problem definition, dataset, frozen eval set, generative-model families, LLM architecture (req. 1–4) | ~5 min |
| `02_prompting` | Baseline E1, four prompt variants E2, reusable template, injection test (req. 5, 6, 11, 13) | ~60 min |
| `03_decoding_evaluation` | Metric self-test, decoding study E3, context-noise test, failure candidates (req. 7, 12) | ~50 min |
| `04_structured_tools` | Structured output E4, confidence and calibration, retrieval tool E5 (req. 9, 10) | ~30 min |
| `05_final_system` | Integrated system E7 and demos | ~20 min |
| `06_harness_ablation` | What each harness part adds, 3B comparison, tool-mode fix, injection tests (req. 8, 12, 13) | ~70 min |
| `07_statistical_analysis` | Confidence intervals, paired tests, errors by question type, calibration, PEFT feasibility | seconds (CPU) |

Notebooks 04–06 clone this repository in their first cell; 01–03 ask for the needed repo files in an upload box.
Regenerate the deliverables after any re-run: `python scripts/build_deliverables.py`.

**Reproducibility.** Same model, same 200 questions (seed 42) and the same scorer everywhere; greedy decoding unless an
experiment varies it; `max_new_tokens=128`; `repetition_penalty=1.0` (switching off Qwen's hidden default of 1.1, along
with its default sampling settings). Every raw model output is saved, so all tables can be recomputed without a GPU.

## Deliverables (brief §9)

See `deliverables/README.md`: dataset and preprocessing record, prompt experiment log, decoding comparison, output
schema and validated samples, tool specification and demo, evaluation and failure-analysis tables.

## Responsible use and limitations

Only public Wikipedia text is used; no personal data. Controls implemented and tested: document text treated as data
(injection guard), evidence must come from the document, strict output schema, tool arguments validated before
execution, abstention when the document does not contain the answer. Limits: the injection guard is pattern-based and
new wordings can evade it; confidence is only weakly informative (AUROC 0.68) and over-confident (ECE 0.21), so it is
shown to users but not used to decide; most remaining hallucinations quote real sentences from adversarial questions,
which evidence checks cannot catch; the model may have seen SQuAD in pre-training, so systems are compared on the same
questions rather than by absolute accuracy.
