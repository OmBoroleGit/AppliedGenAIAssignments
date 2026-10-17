"""
evaluate.py  -  Group 7 SQuAD v2 project, shared evaluation (owner: Dolsy)

Everyone saves experiment results as a CSV with the agreed columns:
    id, question, context, gold_answers, is_answerable, prompt_variant,
    decoding, raw_output  (+ run, if the experiment repeats questions)

This file turns those raw model outputs into scores.

    from src.evaluate import evaluate, score_rows, compare_systems, variability

    metrics = evaluate(results_df)           # one dictionary of headline numbers
    scored  = score_rows(results_df)         # every row with em, f1, error_type
    table   = compare_systems({"baseline": df1, "final": df2})
    vtable  = variability(decoding_df)       # how much answers change between runs
"""

import ast
import json
import re
import string
from collections import Counter

import pandas as pd

REQUIRED_KEYS = {"answerable", "answer", "evidence"}

REFUSAL_PATTERN = re.compile(
    r"unanswerable|not (?:in|found|mentioned|provided|stated)|cannot be answered|no answer",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. Text normalisation and the two official SQuAD scores (EM and F1)
# ---------------------------------------------------------------------------

def normalize_answer(text):
    """Official SQuAD normalisation: lowercase, drop punctuation, drop a/an/the,
    squash spaces. So "The Normans." and "normans" count as the same answer."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1_single(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    # If either side is empty (an abstention), F1 is 1 only when both are empty.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred, golds):
    """1.0 if the prediction matches ANY gold answer after normalisation."""
    return max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)


def f1_score(pred, golds):
    """Word-overlap score against the BEST matching gold answer (0 to 1)."""
    return max(_f1_single(pred, g) for g in golds)


# ---------------------------------------------------------------------------
# 2. Reading the model's raw text output
# ---------------------------------------------------------------------------

def _as_list(value):
    """gold_answers may arrive as a real list, a JSON string or a Python-list string
    (CSV files turn lists into text). Always return a plain list of strings."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            continue
    return [text] if text else []


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def parse_output(raw):
    """Turn one raw model output into answerable / answer / evidence / json_valid.

    json_valid is True only if the output contains a JSON object with all three
    required keys. If the JSON is broken we still try to use the text, so one
    formatting slip does not hide whether the content was right."""
    raw = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    data = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None

    if isinstance(data, dict):
        answer = str(data.get("answer", "") or "").strip()
        answerable = _as_bool(data.get("answerable", bool(answer)))
        evidence = str(data.get("evidence", "") or "").strip()
        if not answerable:
            answer = ""
        return {
            "json_valid": REQUIRED_KEYS.issubset(data.keys()),
            "pred_answerable": answerable,
            "pred_answer": answer,
            "evidence": evidence,
        }

    # Fallback: no usable JSON. Treat the free text as the answer unless it refuses.
    text = raw.strip()
    if not text or REFUSAL_PATTERN.search(text):
        return {"json_valid": False, "pred_answerable": False, "pred_answer": "", "evidence": ""}
    return {"json_valid": False, "pred_answerable": True, "pred_answer": text, "evidence": ""}


def parse_outputs(df):
    """Add pred_answerable / pred_answer / evidence / json_valid columns from raw_output.
    Re-parses every row so all experiments are scored by exactly the same rule."""
    df = df.copy()
    parsed = df["raw_output"].apply(parse_output).apply(pd.Series)
    for col in parsed.columns:
        df[col] = parsed[col]
    return df


# ---------------------------------------------------------------------------
# 3. Scoring every row
# ---------------------------------------------------------------------------

def _squash(text):
    return " ".join(str(text).lower().split())


def evidence_ok(row):
    """For answered questions: is the evidence copied from the context, and does it
    contain the answer? Abstentions return None (not counted)."""
    if not row["pred_answerable"]:
        return None
    evidence = _squash(row["evidence"])
    if not evidence:
        return False
    in_context = evidence in _squash(row["context"])
    answer_in_evidence = normalize_answer(row["pred_answer"]) in normalize_answer(row["evidence"])
    return bool(in_context and answer_in_evidence)


def _error_type(row):
    if not row["is_answerable"]:
        return "hallucination" if row["pred_answerable"] else "correct"
    if not row["pred_answerable"]:
        return "over-abstention"
    if row["em"] == 1:
        return "correct"
    if row["f1"] > 0:
        return "partial / over-long span"
    return "wrong answer"


def score_rows(df):
    """Return every row with em, f1, evidence_ok and an error_type label."""
    df = parse_outputs(df)
    df["is_answerable"] = df["is_answerable"].apply(_as_bool)
    df["gold_answers"] = df["gold_answers"].apply(_as_list)
    df["pred_answer"] = df["pred_answer"].fillna("").astype(str)
    df["evidence"] = df["evidence"].fillna("").astype(str)

    # SQuAD v2 rule: for an unanswerable question the only correct answer is "".
    golds = df.apply(lambda r: r["gold_answers"] if r["is_answerable"] else [""], axis=1)
    preds = df.apply(lambda r: r["pred_answer"] if r["pred_answerable"] else "", axis=1)
    df["em"] = [exact_match(p, g) for p, g in zip(preds, golds)]
    df["f1"] = [f1_score(p, g) for p, g in zip(preds, golds)]
    df["evidence_ok"] = df.apply(evidence_ok, axis=1)
    df["error_type"] = df.apply(_error_type, axis=1)
    return df


# ---------------------------------------------------------------------------
# 4. Headline metrics
# ---------------------------------------------------------------------------

def _pct(series):
    series = series.dropna()
    return float(round(100 * series.astype(float).mean(), 1)) if len(series) else None


def evaluate(df):
    """Headline numbers (all in %) for one experiment."""
    s = score_rows(df)
    ans = s[s["is_answerable"]]
    una = s[~s["is_answerable"]]
    return {
        "n_questions": len(s),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "EM_overall": _pct(s["em"]),
        "F1_overall": _pct(s["f1"]),
        "EM_answerable": _pct(ans["em"]),
        "F1_answerable": _pct(ans["f1"]),
        "abstention_accuracy": _pct(~una["pred_answerable"].astype(bool)),
        "hallucination_rate": _pct(una["pred_answerable"]),
        "over_abstention_rate": _pct(~ans["pred_answerable"].astype(bool)),
        "evidence_faithfulness": _pct(s["evidence_ok"]),
        "json_validity": _pct(s["json_valid"]),
    }


def compare_systems(results):
    """results = {"baseline": df1, "final": df2, ...} -> one table, a row per system."""
    return pd.DataFrame({name: evaluate(df) for name, df in results.items()}).T


# ---------------------------------------------------------------------------
# 5. Variability between repeated runs (decoding study)
# ---------------------------------------------------------------------------

def variability(df, setting_col="decoding", run_col="run"):
    """For each decoding setting: average quality AND how stable the answers are.

    agreement      = on average, what share of runs gave the most common answer
                     (100 = identical every time)
    distinct_answers = average number of different answers per question
    F1_std_across_runs = how much the overall F1 moves from one run to the next"""
    s = score_rows(df)
    s["norm_pred"] = s.apply(
        lambda r: normalize_answer(r["pred_answer"]) if r["pred_answerable"] else "<abstain>", axis=1
    )
    rows = []
    for setting, g in s.groupby(setting_col, sort=False):
        per_q = g.groupby("id")["norm_pred"]
        agreement = per_q.apply(lambda x: x.value_counts().iloc[0] / len(x)).mean()
        distinct = per_q.nunique().mean()
        run_f1 = g.groupby(run_col)["f1"].mean() * 100 if run_col in g else pd.Series([g["f1"].mean() * 100])
        m = evaluate(g)
        rows.append({
            setting_col: setting,
            "runs": g[run_col].nunique() if run_col in g else 1,
            "F1_answerable": m["F1_answerable"],
            "hallucination_rate": m["hallucination_rate"],
            "json_validity": m["json_validity"],
            "agreement": round(100 * agreement, 1),
            "distinct_answers": round(distinct, 2),
            "F1_std_across_runs": round(run_f1.std(ddof=0), 2),
        })
    return pd.DataFrame(rows)
