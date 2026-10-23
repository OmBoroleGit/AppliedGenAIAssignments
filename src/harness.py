"""Harness extensions + ablation support.

The final system (E7) = few-shot prompt -> model -> validator (R1-R7) -> one retry -> confidence.
That is already a small harness: a SENSOR (the validator) and a FEEDBACK LOOP (the retry).
The E7 results showed what it costs: compared with the few-shot prompt alone it fixed 3
hallucinations but turned 12 answerable questions into refusals (F1 77.0 -> 74.0). Two causes:
  1. R5 demands an EXACT quote. Copies that are 88-98% identical (a dropped full stop, a cut-off
     end) were rejected.
  2. The retry message suggests abstaining ("If the paragraph does not contain the answer use
     answerable=false ..."), so after any rejection the model tends to give up.

This module adds the harness parts below, each with an on/off switch so the ablation
(notebook 06) can measure what every part changes:
  GUIDE   input_guard      - treat the document as data: strip injected instructions
  SENSOR  fuzzy_evidence   - accept evidence whose characters align >= 90% with the paragraph (R5 relaxed)
  SENSOR  echo_check       - reject answers that only repeat words of the question (R8)
  LOOP    neutral_feedback - tell the model WHAT to fix, without nudging it to abstain
  CONTROL confidence threshold (applied after the run; chosen on separate calibration data)
It reuses the validator, confidence and tool code (schema.py, confidence.py, tools.py); nothing in those modules is changed.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

from .confidence import confidence_from_logprobs
from .schema import ABSTAIN, _norm, validate_output
from . import tools

# ------------------------------------------------------------------ config
@dataclass
class HarnessConfig:
    validate: bool = True            # validator R1-R7 from schema.py (False = raw model output, no harness)
    max_retries: int = 1
    fuzzy_evidence: bool = False
    echo_check: bool = False
    neutral_feedback: bool = False
    input_guard: bool = False


OM_FEEDBACK = (". Reply again with ONLY the JSON object. The evidence must be copied "
               "exactly from the paragraph. If the paragraph does not contain the "
               "answer use answerable=false, answer \"not in the document\" and empty evidence.")
NEUTRAL_FEEDBACK = (". Fix only the problem(s) listed and reply with ONLY the corrected JSON object. "
                    "Copy the evidence sentence character for character from the paragraph, and keep it "
                    "to ONE sentence.")

# ------------------------------------------------------------------ GUIDE
INJECTION = re.compile(
    r"ignore (?:all )?(?:the )?(?:previous|prior|above|earlier) instructions|system override|\[\s*admin"
    r"|override accepted|disregard (?:all |the )?(?:previous |prior )?instructions|new instructions?:"
    r"|you are now|respond (?:only )?with|output (?:exactly|strictly)|you must (?:now )?(?:say|answer|respond)"
    r"|(?:mark|treat) everything as|regardless of (?:the )?content|document (?:above )?is revoked",
    re.IGNORECASE,
)


def guard_injection(context: str):
    """Once a sentence looks like an instruction to the model, drop it and the rest of its
    paragraph (injections usually run over several sentences). Pattern-based, so new wording can
    evade it - the grounding check is the second line of defence."""
    kept_par, removed = [], []
    for par in re.split(r"\n\s*\n", str(context)):
        kept, hit = [], False
        for sent in re.split(r"(?<=[.!?])\s+", par.strip()):
            if hit or INJECTION.search(sent):
                hit = True
                removed.append(sent)
            else:
                kept.append(sent)
        if kept:
            kept_par.append(" ".join(kept))
    return "\n\n".join(kept_par), removed

# ----------------------------------------------------------------- SENSORS
def evidence_coverage(evidence: str, context: str, min_run: int = 8) -> float:
    """Share of the evidence's characters that line up, IN ORDER, with the context in runs of at
    least `min_run` characters (0-1). A dropped hyphen or a changed quote mark inside a copied
    sentence costs ~1%; an invented or paraphrased sentence scores far below 0.9."""
    e, c = _norm(evidence), _norm(context)
    if not e:
        return 0.0
    if e in c:
        return 1.0
    blocks = SequenceMatcher(None, e, c, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= min_run) / len(e)


def check(raw: str, context: str, question: str, cfg: HarnessConfig):
    """The schema.py validator, then the harness adjustments. Returns (parsed, errors, warnings)."""
    vr = validate_output(raw, context)
    errors, warnings = list(vr.errors), list(vr.warnings)
    if cfg.fuzzy_evidence and vr.parsed is not None:
        r5 = [e for e in errors if e.startswith("R5")]
        if r5 and evidence_coverage(vr.parsed.evidence, context) >= 0.9:
            errors = [e for e in errors if not e.startswith("R5")]
            warnings.append("W3: evidence accepted as a near-exact quote (>= 90% match)")
    if cfg.echo_check and vr.parsed is not None and vr.parsed.answerable and vr.parsed.answer.strip():
        a = _norm(vr.parsed.answer)
        if a and a in _norm(question):
            errors.append("R8: the answer only repeats words from the question")
    return vr.parsed, errors, warnings

# --------------------------------------------------------------- the loop
def _as_gen(res) -> Dict[str, Any]:
    return res if isinstance(res, dict) else {"text": res, "tokens": None, "logprobs": None}


def answer_harness(question: str, context: str, build_messages: Callable, gen_fn: Callable,
                   cfg: HarnessConfig, **gen_kwargs) -> Dict[str, Any]:
    """Same return format as schema.answer_json (so runlog.make_row works), plus a trace."""
    trace: Dict[str, Any] = {"config": asdict(cfg), "removed_by_guard": [], "attempts": []}
    if cfg.input_guard:
        context, removed = guard_injection(context)
        trace["removed_by_guard"] = removed
    msgs = build_messages(context, question)
    raws, all_errors, warnings, g, ok, parsed = [], [], [], None, False, None
    tries = 1 + (cfg.max_retries if cfg.validate else 0)
    for _ in range(tries):
        g = _as_gen(gen_fn(msgs, **gen_kwargs))
        raws.append(g["text"])
        if not cfg.validate:
            break
        parsed, errors, warnings = check(g["text"], context, question, cfg)
        trace["attempts"].append({"output": g["text"], "errors": errors})
        if not errors:
            ok = True
            break
        all_errors.append(errors)
        msgs = msgs + [{"role": "assistant", "content": g["text"]},
                       {"role": "user", "content": "Your reply was not valid: " + "; ".join(errors) +
                        (NEUTRAL_FEEDBACK if cfg.neutral_feedback else OM_FEEDBACK)}]

    conf = confidence_from_logprobs(g["tokens"], g["logprobs"]) if g and g.get("tokens") else None
    if not cfg.validate:                          # no harness: score the raw model text, like E2
        from .evaluate import parse_output
        p = parse_output(g["text"])
        out = {"answerable": p["pred_answerable"], "answer": p["pred_answer"], "evidence": p["evidence"],
               "confidence": round(conf, 4) if conf is not None else None}
        return {"result": out, "json_valid": p["json_valid"], "attempts": 1, "raw_output": g["text"],
                "first_raw_output": raws[0], "raw_outputs": raws, "errors": [], "warnings": [],
                "confidence_source": "token_probs" if conf is not None else "none",
                "context_used": context, "trace": trace}
    if ok:
        out = parsed.model_dump()
        out["confidence"], source = (round(conf, 4), "token_probs") if conf is not None else (0.0, "none")
    else:                                         # still failing: abstain rather than return it
        out, source = dict(ABSTAIN), "fallback_abstain"
    return {"result": out, "json_valid": ok, "attempts": len(raws),
            "raw_output": json.dumps(out, ensure_ascii=False),      # always the FINAL decision
            "first_raw_output": raws[0], "raw_outputs": raws, "errors": all_errors, "warnings": warnings,
            "confidence_source": source, "context_used": context, "trace": trace}


def answer_with_retrieval(question: str, build_messages: Callable, gen_fn: Callable, cfg: HarnessConfig,
                          top_k: int = 3, **gen_kwargs) -> Dict[str, Any]:
    """Tool mode: the model decides on the tool call (tools.TOOL_SYSTEM_PROMPT and parser), Python runs
    tools.retrieve_context, and the FINAL answer uses the same few-shot prompt and harness as paragraph
    mode - so tool mode and paragraph mode differ only in where the context comes from."""
    msgs = [{"role": "system", "content": tools.TOOL_SYSTEM_PROMPT}, {"role": "user", "content": f"Question: {question}"}]
    first = _as_gen(gen_fn(msgs, **gen_kwargs))["text"]
    call = tools.parse_tool_call(first)
    forced = call is None
    if forced:
        call = {"name": "retrieve_context", "arguments": {"question": question, "top_k": top_k}}
    try:
        args = tools.RetrieveArgs(**call["arguments"])
    except Exception:
        args = tools.RetrieveArgs(question=question, top_k=top_k)
    result = tools.retrieve_context(**args.model_dump())
    context = "\n\n".join(p["text"] for p in result["passages"])
    out = answer_harness(question, context, build_messages, gen_fn, cfg, **gen_kwargs)
    out["trace"].update(tool_decision_output=first, tool_called_by_model=not forced,
                        tool_args=args.model_dump(), n_passages=len(result["passages"]))
    out["retrieved"] = result["passages"]
    return out

# ----------------------------------------------------------------- CONTROL
def apply_threshold(df, threshold: float):
    """Abstain on answers whose confidence is below the threshold (after the run, no regeneration)."""
    d = df.copy()
    if threshold is None or threshold <= 0:
        return d
    import pandas as pd
    conf = pd.to_numeric(d["confidence"], errors="coerce").fillna(0)
    low = d["pred_answerable"].astype(str).str.lower().isin(["true", "1"]) & (conf < threshold)
    abstain = json.dumps({**ABSTAIN, "confidence": None})
    d.loc[low, "raw_output"] = abstain
    d.loc[low, "pred_answerable"] = False
    d.loc[low, ["pred_answer", "evidence"]] = ""
    return d
