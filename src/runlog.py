"""Helpers so every experiment writes the same CSV columns."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

COLUMNS = ["id", "question", "context", "gold_answers", "is_answerable", "prompt_variant",
           "decoding", "raw_output", "pred_answerable", "pred_answer", "evidence",
           "confidence", "json_valid"]
# extra columns added by notebooks 04-06 (evaluate.py ignores them)
EXTRA = ["attempts", "json_valid_first_attempt", "confidence_source", "errors", "warnings"]


def load_eval_set(path: str = "data/eval_set.jsonl") -> List[Dict[str, Any]]:
    """Read the frozen eval set (built in notebook 01). Tolerates answers as list or SQuAD dict."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gold = r.get("gold_answers", r.get("answers", []))
        if isinstance(gold, dict):
            gold = gold.get("text", [])
        r["gold_answers"] = list(gold)
        r["is_answerable"] = bool(r.get("is_answerable", len(r["gold_answers"]) > 0))
        rows.append(r)
    return rows


def make_row(example: Dict[str, Any], out: Dict[str, Any], prompt_variant: str,
             decoding: str) -> Dict[str, Any]:
    res = out["result"]
    return {
        "id": example["id"], "question": example["question"], "context": example["context"],
        "gold_answers": json.dumps(example["gold_answers"]),
        "is_answerable": example["is_answerable"],
        "prompt_variant": prompt_variant, "decoding": decoding,
        "raw_output": out["raw_output"], "pred_answerable": res["answerable"],
        "pred_answer": res["answer"], "evidence": res["evidence"],
        "confidence": res["confidence"], "json_valid": out["json_valid"],
        "attempts": out["attempts"],
        "json_valid_first_attempt": bool(out["json_valid"] and out["attempts"] == 1),
        "confidence_source": out["confidence_source"],
        "errors": json.dumps(out["errors"]), "warnings": json.dumps(out["warnings"]),
    }


def load_fewshot(path: str = "data/fewshot_pool.jsonl") -> List[Dict[str, Any]]:
    """The 4 few-shot examples from the train split (3 answerable + 1 unanswerable)."""
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def make_build_messages(variant: str = "few_shot", fewshot_path: str = "data/fewshot_pool.jsonl"):
    """Return build_messages(context, question) that uses the team's reusable template
    src/prompts.build_prompt(). Best prompt from E2 = few_shot (results/best_prompt.txt)."""
    from .prompts import build_prompt
    shots = load_fewshot(fewshot_path) if variant == "few_shot" else None

    def build_messages(context: str, question: str):
        return build_prompt(context, question, variant=variant, tokenizer=None, few_shot_examples=shots)
    return build_messages


def run_cached(path, items, fn, key, save_every: int = 20):
    """Run fn(item) -> row dict for every item, saving to `path` every few rows and
    SKIPPING items already in the file, so a Colab disconnect only loses a few rows."""
    import pandas as pd
    from pathlib import Path as _P
    _P(path).parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(path, keep_default_na=False).to_dict("records") if _P(path).exists() else []
    done = {r["_key"] for r in rows}
    try:
        from tqdm.auto import tqdm
        it = tqdm(items)
    except Exception:
        it = items
    new = 0
    for item in it:
        k = key(item)
        if k in done:
            continue
        r = fn(item)
        r["_key"] = k
        rows.append(r)
        new += 1
        if new % save_every == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path, keep_default_na=False)    # empty answers stay "" (not NaN)
