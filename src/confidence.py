"""Confidence from the model's own token probabilities (owner: Om).

Method (explain this in the demo):
  The JSON the model writes contains two *decisions*:
    1. the boolean after  "answerable":   (answer or abstain?)
    2. the text inside    "answer": "..." (what is the answer?)
  confidence = exp( mean log-probability of the tokens of those decisions )
  i.e. the geometric-mean probability the model assigned to its own decision
  tokens. For an abstention only the boolean token is used.
This is a real probability signal, not the model saying "I'm sure". The
confidence field the model writes itself is ignored and overwritten.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional

_ANSWERABLE_RE = re.compile(r'"answerable"\s*:\s*(true|false)')
_ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _spans(tokens: List[str]):
    """Character span (start, end) of every token inside the concatenated text."""
    out, pos = [], 0
    for t in tokens:
        out.append((pos, pos + len(t)))
        pos += len(t)
    return out


def _tokens_in(span, spans):
    s, e = span
    return [i for i, (a, b) in enumerate(spans) if b > s and a < e]


def confidence_from_logprobs(tokens: List[str], logprobs: List[float]) -> Optional[float]:
    """Return confidence in [0, 1], or None if the decision tokens cannot be located."""
    if not tokens or len(tokens) != len(logprobs):
        return None
    text = "".join(tokens)
    spans = _spans(tokens)
    idx: List[int] = []

    m = _ANSWERABLE_RE.search(text)
    if not m:
        return None
    flag = _tokens_in((m.start(1), m.end(1)), spans)
    idx += flag
    is_true = m.group(1) == "true"

    if is_true:
        a = _ANSWER_RE.search(text)
        if a and a.group(1):
            idx += _tokens_in((a.start(1), a.end(1)), spans)

    idx = sorted(set(idx))
    if not idx:
        return None
    mean_lp = sum(logprobs[i] for i in idx) / len(idx)
    return max(0.0, min(1.0, math.exp(mean_lp)))


def calibration_table(confidences, correct, n_bins: int = 5):
    """Bin confidence and report accuracy per bin + Expected Calibration Error."""
    rows, n, ece = [], len(confidences), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not sel:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "mean_conf": None, "accuracy": None})
            continue
        mc = sum(confidences[i] for i in sel) / len(sel)
        acc = sum(1 for i in sel if correct[i]) / len(sel)
        ece += len(sel) / n * abs(mc - acc)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                     "mean_conf": round(mc, 3), "accuracy": round(acc, 3)})
    return rows, round(ece, 4)
