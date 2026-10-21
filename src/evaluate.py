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
    r"^\s*(?:none|n/?a|null)\s*\.?\s*$|unanswerable|cannot be answered|no answer"
    r"|not (?:explicitly )?(?:in|found|mentioned|provided|stated|specified|given|applicable)",
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


def _is_refusal(answer):
    """A short answer that only says 'not in the document' / 'unanswerable' etc."""
    answer = str(answer).strip()
    return bool(answer) and len(answer.split()) <= 8 and bool(REFUSAL_PATTERN.search(answer))


def _strict_json(raw):
    """Read the first complete JSON object in the text (ignores text before/after it)."""
    start = raw.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _lenient_fields(raw):
    """Recover the fields from output that is NOT valid JSON. Three real cases seen in our runs:
    (1) JSON cut off at the token limit, (2) evidence with unescaped quotes copied from the text,
    (3) 'answerable: True / answer: ... / evidence: ...' written as plain lines."""
    flag = re.search(r'"?answerable"?\s*:\s*"?(true|false)"?', raw, re.IGNORECASE)
    if not flag:
        return None
    ans = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if not ans:
        ans = re.search(r'^\s*answer\s*:\s*"?(.*?)"?\s*$', raw, re.IGNORECASE | re.MULTILINE)
    ev = re.search(r'"evidence"\s*:\s*"(.*)', raw, re.DOTALL)
    if ev:
        evidence = re.sub(r'"\s*\}?\s*$', "", ev.group(1)).strip()
    else:
        ev = re.search(r'^\s*evidence\s*:\s*"?(.*?)"?\s*$', raw, re.IGNORECASE | re.MULTILINE)
        evidence = ev.group(1).strip() if ev else ""
    return {"answerable": flag.group(1).lower() == "true",
            "answer": ans.group(1).strip() if ans else "", "evidence": evidence}


def parse_output(raw):
    """Turn one raw model output into answerable / answer / evidence / json_valid / flag_conflict.

    - json_valid: True only for a complete JSON object with all three keys (the FORMAT score).
    - If the JSON is broken, the fields are still recovered, so the CONTENT is scored on its merits.
    - The answer text wins over the true/false flag: "answerable": true with answer
      "not in the document" is a refusal. Such contradictions are counted in flag_conflict.
    - An empty answer counts as abstaining."""
    raw = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)

    data = _strict_json(raw)
    json_valid = isinstance(data, dict) and REQUIRED_KEYS.issubset(data.keys())
    if isinstance(data, dict):
        fields = {"answerable": _as_bool(data.get("answerable", bool(data.get("answer")))),
                  "answer": str(data.get("answer", "") or "").strip(),
                  "evidence": str(data.get("evidence", "") or "").strip()}
    else:
        fields = _lenient_fields(raw)

    if fields is None:
        # Plain free text (e.g. the baseline): the text is the answer unless it refuses.
        text = raw.strip()
        refused = not text or bool(REFUSAL_PATTERN.search(text))
        return {"json_valid": False, "pred_answerable": not refused,
                "pred_answer": "" if refused else text, "evidence": "", "flag_conflict": False}

    flag, answer = fields["answerable"], fields["answer"]
    says_refusal = _is_refusal(answer) or not answer
    if flag and says_refusal:          # "true" but the answer refuses -> abstain
        answerable, conflict = False, True
    elif not flag and answer and not REFUSAL_PATTERN.search(answer):   # "false" but a real answer is given -> trust the flag
        answerable, conflict = False, True
    else:
        answerable, conflict = flag, False
    return {"json_valid": json_valid, "pred_answerable": answerable,
            "pred_answer": answer if answerable else "", "evidence": fields["evidence"],
            "flag_conflict": conflict}


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
    """Headline numbers (all in %) for one experiment.
    flag_conflicts = outputs whose true/false flag contradicts their own answer text."""
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
        "flag_conflicts": _pct(s["flag_conflict"]),
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
            "over_abstention_rate": m["over_abstention_rate"],
            "json_validity": m["json_validity"],
            "agreement": round(100 * agreement, 1),
            "distinct_answers": round(distinct, 2),
            "F1_std_across_runs": round(run_f1.std(ddof=0), 2),
        })
    return pd.DataFrame(rows)
