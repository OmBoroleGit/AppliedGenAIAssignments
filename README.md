# Group 7 - SQuAD v2 "Answer-or-Abstain" assistant (PCAM ZG516)

Qwen2.5-1.5B-Instruct (no training) + prompts, decoding control, validated JSON output, token-probability
confidence and a BM25 retrieval tool. Submission: 24 Oct.

## Run in Colab (Runtime > Change runtime type > T4 GPU)
Open a notebook from `notebooks/` in Colab (File > Open notebook > GitHub). Each notebook's first cell clones this repo
and installs `requirements.txt`. Order for the full system: `04_structured_tools.ipynb`, then `05_final_system.ipynb`
(Runtime > Restart and run all). Seed 42, frozen `data/eval_set.jsonl`, greedy decoding, `max_new_tokens=128`.

## Layout
| Path | Owner | What |
|---|---|---|
| `data/eval_set.jsonl`, `data/fewshot_pool.jsonl` | Pranil | frozen 200-question eval set, 4 few-shot examples |
| `notebooks/01_data_foundations.ipynb` | Pranil | problem, dataset, model families, architecture |
| `notebooks/02_prompting.ipynb`, `src/prompts.py` | Harshal | E1 baseline, E2 four prompts, injection test, reusable template |
| `notebooks/03_decoding_evaluation.ipynb`, `src/evaluate.py` | Dolsy | E3 decoding, noise test, metrics, failure analysis |
| `src/model.py` | Om | `generate()`, `generate_with_logprobs()` |
| `src/schema.py` | Om | pydantic schema, validator (R1-R7), retry |
| `src/confidence.py` | Om | confidence from token probabilities, calibration/ECE |
| `src/tools.py` | Om | `retrieve_context` (BM25) + tool-calling loop |
| `src/runlog.py` | Om | CSV columns, eval-set loader, `make_build_messages()` (uses `prompts.build_prompt`) |
| `notebooks/04_structured_tools.ipynb` | Om | E4 structured output, confidence, E5 tool |
| `notebooks/05_final_system.ipynb` | Om | E7 final integrated system + demo |
| `tests/` | Om | offline unit tests (no GPU): `python -m unittest discover tests -v` |

## Output format
`{"answerable": true, "answer": "1876", "evidence": "Bell patented the telephone in 1876.", "confidence": 0.82}`
The model writes `answerable/answer/evidence`; `confidence` is added from its token probabilities (`src/confidence.py`).
An abstention is `"answer": "not in the document"` in the raw reply and `""` after validation.

## Known limits
Tool index covers 20 articles. Qwen may have seen SQuAD/Wikipedia in pre-training (data contamination).
