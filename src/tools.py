"""retrieve_context tool + tool-calling loop - experiment E5.

Flow (brief section 5):  user request -> LLM -> tool decision -> tool result -> final answer
  1. Model gets the question (and maybe a paragraph) and replies with EITHER
     a tool call  {"tool_call": {"name": "retrieve_context", "arguments": {...}}}
     OR a final JSON answer (when a paragraph was supplied).
  2. Python validates the arguments and runs the tool (BM25 search).
  3. Passages go back to the model as a message.
  4. Model writes the final JSON answer, validated by schema.answer_json
     against the retrieved passages (evidence must be quoted from them).
BM25 is implemented here (about 40 lines, no extra dependency) so it is easy to explain.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from .schema import ABSTAIN, answer_json, extract_json, validate_output

# ------------------------------------------------------------ tool spec
TOOL_SPEC = {
    "name": "retrieve_context",
    "description": "Search a library of Wikipedia passages (SQuAD articles) with BM25 and "
                   "return the passages most relevant to the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "search query, normally the user question"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
        },
        "required": ["question"],
    },
    "output_schema": {
        "type": "object",
        "properties": {"passages": {"type": "array", "items": {
            "type": "object",
            "properties": {"passage_id": {"type": "integer"}, "title": {"type": "string"},
                           "text": {"type": "string"}, "score": {"type": "number"}},
            "required": ["passage_id", "title", "text", "score"]}}},
        "required": ["passages"],
    },
}


class RetrieveArgs(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=5)


# ------------------------------------------------------------------ BM25
_STOP = set("a an the of in on at to for and or is are was were be been by with as from that this "
            "it its what which who whom when where why how did do does has have had not no".split())


def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOP]


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            out.append(s)
        return out


# ------------------------------------------------------------ corpus/index
_PASSAGES: List[Dict[str, Any]] = []
_INDEX: Optional[BM25] = None


def build_corpus(records, titles: Optional[List[str]] = None, n_articles: int = 20) -> List[Dict[str, Any]]:
    """records: iterable of dicts with 'title' and 'context' (e.g. SQuAD v2 validation rows).
    Keeps every distinct paragraph of the first `n_articles` titles (or the given titles).
    Choose the titles from data/eval_set.jsonl so the demo questions are answerable by the index."""
    seen, chosen = set(), []
    pool = list(records)
    if titles is None:
        titles = []
        for r in pool:
            if r["title"] not in titles:
                titles.append(r["title"])
            if len(titles) == n_articles:
                break
    tset = set(titles)
    for r in pool:
        if r["title"] in tset and r["context"] not in seen:
            seen.add(r["context"])
            chosen.append({"passage_id": len(chosen), "title": r["title"], "text": r["context"]})
    return chosen


def init_index(passages: List[Dict[str, Any]]) -> int:
    global _PASSAGES, _INDEX
    _PASSAGES = passages
    _INDEX = BM25([p["title"] + " " + p["text"] for p in passages])
    return len(passages)


def retrieve_context(question: str, top_k: int = 3) -> Dict[str, Any]:
    """The tool. Returns {"passages": [{passage_id, title, text, score}, ...]}."""
    args = RetrieveArgs(question=question, top_k=top_k)
    if _INDEX is None:
        raise RuntimeError("Call init_index(build_corpus(...)) first")
    sc = _INDEX.scores(args.question)
    order = sorted(range(len(sc)), key=lambda i: -sc[i])[: args.top_k]
    return {"passages": [{**_PASSAGES[i], "score": round(sc[i], 4)} for i in order if sc[i] > 0]}


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {"retrieve_context": retrieve_context}


# --------------------------------------------------------------- tool loop
TOOL_SYSTEM_PROMPT = (
    "You are a careful reference librarian with ONE tool.\n"
    "Tool: retrieve_context(question: string, top_k: integer 1-5) - searches a library of "
    "Wikipedia passages and returns the most relevant ones.\n"
    "Rules:\n"
    "- If the user gave you a Paragraph, answer from it directly (do not call the tool).\n"
    "- If the user gave NO paragraph, you must call the tool. Reply with ONLY this JSON:\n"
    '{"tool_call": {"name": "retrieve_context", "arguments": {"question": "<the question>", "top_k": 3}}}\n'
    "- When you answer, reply with ONLY this JSON:\n"
    '{"answerable": true|false, "answer": "<short phrase copied from the text>", '
    '"evidence": "<one sentence copied exactly from the text, or empty>"}\n'
    "- If the text does not contain the answer: "
    '{"answerable": false, "answer": "not in the document", "evidence": ""}'
)


def parse_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    obj = extract_json(raw)
    if not obj:
        return None
    call = obj.get("tool_call", obj if "name" in obj and "arguments" in obj else None)
    if isinstance(call, dict) and call.get("name") in TOOLS and isinstance(call.get("arguments"), dict):
        return call
    return None


def format_tool_result(result: Dict[str, Any]) -> str:
    if not result["passages"]:
        return "Tool result: no passages found."
    return "Tool result:\n" + "\n".join(
        f"[{i + 1}] ({p['title']}) {p['text']}" for i, p in enumerate(result["passages"]))


def answer_with_tools(question: str, gen_fn: Callable[..., Any], paragraph: Optional[str] = None,
                      top_k: int = 3, force_tool_fallback: bool = True, **gen_kwargs) -> Dict[str, Any]:
    """Run the full tool loop. Returns the answer_json() dict plus a `trace`."""
    user = (f"Paragraph:\n{paragraph}\n\n" if paragraph else "") + f"Question: {question}"
    msgs = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    trace: Dict[str, Any] = {"tool_called": False, "tool_args": None, "n_passages": 0,
                             "forced_fallback": False, "tool_error": None}

    g = gen_fn(msgs, **gen_kwargs)
    first_raw = g["text"] if isinstance(g, dict) else g
    call = parse_tool_call(first_raw)

    if call is None:
        if paragraph:  # decision "no tool": direct answer, validated against the paragraph
            out = answer_json(msgs, paragraph, gen_fn, first=g, **gen_kwargs)
            out["trace"] = trace
            return out
        if not force_tool_fallback:
            return {"result": dict(ABSTAIN), "json_valid": False, "attempts": 1,
                    "raw_output": first_raw, "raw_outputs": [first_raw], "errors": [["no tool call"]],
                    "warnings": [], "confidence_source": "fallback_abstain", "trace": trace}
        trace["forced_fallback"] = True   # model skipped the tool: Python makes the call
        call = {"name": "retrieve_context", "arguments": {"question": question, "top_k": top_k}}

    trace["tool_called"] = not trace["forced_fallback"]
    try:
        args = RetrieveArgs(**call["arguments"])
        trace["tool_args"] = args.model_dump()
        result = TOOLS[call["name"]](**args.model_dump())
    except (ValidationError, TypeError, RuntimeError) as e:
        trace["tool_error"] = str(e)[:200]
        result = {"passages": []}
    trace["n_passages"] = len(result["passages"])
    context = "\n".join(p["text"] for p in result["passages"])

    msgs2 = msgs + [
        {"role": "assistant", "content": first_raw if not trace["forced_fallback"] else
         json.dumps({"tool_call": call})},
        {"role": "user", "content": format_tool_result(result) +
         "\n\nNow answer the question using ONLY these passages. Reply with ONLY the JSON answer; "
         "the evidence must be copied exactly from a passage."},
    ]
    out = answer_json(msgs2, context, gen_fn, **gen_kwargs)
    out["trace"] = trace
    out["retrieved"] = result["passages"]
    return out
