"""JSON schema, validator and retry (owner: Om) - experiment E4.

Output contract (locked):
    {"answerable": bool, "answer": str, "evidence": str, "confidence": float 0-1}

Validation rules (each one is a failure mode we saw or expect in a 1.5B model):
  R1  reply contains parseable JSON                 (models add prose / ``` fences)
  R2  answerable/answer/evidence present with the right types (pydantic, strict bool).
      `confidence` is OPTIONAL: the team's prompts ask for 3 keys; we add the real confidence ourselves.
  R3  confidence, if the model wrote one, within [0, 1]
  R4  answerable == true  -> answer and evidence are non-empty
  R5  answerable == true  -> evidence text really occurs in the paragraph
      (this is what stops invented "proof sentences")
  R6  answerable == false -> answer must be "" (the team's phrase "not in the document" counts
      as empty and is normalised to "")
  R7  answerable == true  -> answer must not be a refusal phrase
Soft warnings (logged, not failures): answer not contained in evidence;
evidence non-empty on an abstention.
Retry policy: one retry that shows the model its own error list.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, StrictBool, StrictStr, ValidationError

from .confidence import confidence_from_logprobs


class AnswerOutput(BaseModel):
    answerable: StrictBool
    answer: StrictStr = ""
    evidence: StrictStr = ""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)   # filled from token probabilities


JSON_SCHEMA = AnswerOutput.model_json_schema()

REFUSAL_PHRASES = {"not in the document", "not in document", "unanswerable", "none", "n/a", "no answer"}

ABSTAIN = {"answerable": False, "answer": "", "evidence": "", "confidence": 0.0}

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful reference librarian. Answer the question using ONLY the "
    "paragraph provided. If the paragraph does not contain the answer, you must "
    "abstain. Reply with ONLY one JSON object and nothing else, in this format:\n"
    '{"answerable": true|false, "answer": "<short answer or not in the document>", '
    '"evidence": "<one sentence copied exactly from the paragraph, or empty string>"}\n'
    'If the answer is not in the paragraph reply: '
    '{"answerable": false, "answer": "not in the document", "evidence": ""}'
)


def build_default_messages(context: str, question: str) -> List[Dict[str, str]]:
    """Minimal constraint prompt used for E4 on its own (Pranil's variants replace it in E7)."""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Paragraph:\n{context}\n\nQuestion: {question}"},
    ]


# ----------------------------------------------------------------- parsing
def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a reply (handles ``` fences and chatter)."""
    if not raw:
        return None
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _is_refusal(text: str) -> bool:
    return re.sub(r"[^a-z/ ]", "", text.lower()).strip() in REFUSAL_PHRASES


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\n\"'.,;:!?")


def evidence_in_context(evidence: str, context: str) -> bool:
    e = _norm(evidence)
    return bool(e) and e in _norm(context)


@dataclass
class ValidationResult:
    ok: bool
    parsed: Optional[AnswerOutput] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_output(raw: str, context: str) -> ValidationResult:
    obj = extract_json(raw)
    if obj is None:
        return ValidationResult(False, errors=["R1: no valid JSON object found in the reply"])
    if obj.get("answerable") is False and isinstance(obj.get("answer"), str) and _is_refusal(obj["answer"]):
        obj = {**obj, "answer": ""}          # "not in the document" == empty answer
    try:
        parsed = AnswerOutput(**{k: obj[k] for k in obj if k in AnswerOutput.model_fields})
    except ValidationError as e:
        errs = [f"R2/R3: field '{'.'.join(map(str, x['loc']))}': {x['msg']}" for x in e.errors()]
        return ValidationResult(False, errors=errs)

    errors, warnings = [], []
    if parsed.answerable:
        if not parsed.answer.strip():
            errors.append("R4: answerable is true but answer is empty")
        elif _is_refusal(parsed.answer):
            errors.append("R7: answerable is true but the answer says it is not in the document")
        if not parsed.evidence.strip():
            errors.append("R4: answerable is true but evidence is empty")
        elif not evidence_in_context(parsed.evidence, context):
            errors.append("R5: evidence is not an exact quote from the paragraph")
        if parsed.answer.strip() and parsed.evidence.strip() and \
                _norm(parsed.answer) not in _norm(parsed.evidence):
            warnings.append("W1: answer text does not appear inside the evidence sentence")
    else:
        if parsed.answer.strip():
            errors.append("R6: answerable is false but answer is not empty")
        if parsed.evidence.strip():
            warnings.append("W2: evidence given although answerable is false")
    return ValidationResult(not errors, parsed, errors, warnings)


# ------------------------------------------------------------ retry wrapper
def _as_gen_result(res) -> Dict[str, Any]:
    """Accept str (generate) or dict (generate_with_logprobs)."""
    return res if isinstance(res, dict) else {"text": res, "tokens": None, "logprobs": None}


def answer_json(messages: List[Dict[str, str]], context: str,
                gen_fn: Callable[..., Any], max_retries: int = 1,
                first: Any = None, **gen_kwargs) -> Dict[str, Any]:
    """Generate -> validate -> (retry once) -> attach real confidence.

    gen_fn: src.model.generate_with_logprobs (preferred) or generate.
    Returns dict with: result (final 4-field dict), json_valid, attempts,
    raw_output (the FINAL accepted reply as JSON incl. real confidence - what evaluate.py reads),
    first_raw_output, raw_outputs (all), errors, warnings, confidence_source.
    first: an already-generated first reply (str or dict) to validate instead of generating again.
    On final failure result = abstain object and json_valid = False.
    """
    msgs = list(messages)
    raws, all_errors, g = [], [], None
    vr = ValidationResult(False)
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts += 1
        g = _as_gen_result(first if (attempt == 0 and first is not None) else gen_fn(msgs, **gen_kwargs))
        raws.append(g["text"])
        vr = validate_output(g["text"], context)
        if vr.ok:
            break
        all_errors.append(vr.errors)
        msgs = msgs + [
            {"role": "assistant", "content": g["text"]},
            {"role": "user", "content": (
                "Your reply was not valid: " + "; ".join(vr.errors) +
                ". Reply again with ONLY the JSON object. The evidence must be copied "
                "exactly from the paragraph. If the paragraph does not contain the "
                "answer use answerable=false, answer \"not in the document\" and empty evidence.")},
        ]

    if vr.ok:
        out = vr.parsed.model_dump()
        conf = confidence_from_logprobs(g["tokens"], g["logprobs"]) if g.get("tokens") else None
        if conf is not None:
            out["confidence"], source = round(conf, 4), "token_probs"
        elif out.get("confidence") is not None:
            source = "model_stated"
        else:
            out["confidence"], source = 0.0, "none"
        final_raw = json.dumps(out, ensure_ascii=False)
    else:
        out, source, final_raw = dict(ABSTAIN), "fallback_abstain", raws[-1]

    return {"result": out, "json_valid": vr.ok, "attempts": attempts,
            "raw_output": final_raw, "first_raw_output": raws[0], "raw_outputs": raws, "errors": all_errors,
            "warnings": vr.warnings, "confidence_source": source}
