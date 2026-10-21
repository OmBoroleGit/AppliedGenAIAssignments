"""Shared model wrapper (owner: Om).

Contract agreed by the group:
    generate(prompt, temperature=0, top_p=None, top_k=None) -> str

Extras used by Om's pieces (do not break the contract above):
    generate_with_logprobs(...) -> dict(text, tokens, logprobs)   # for confidence
`prompt` may be a plain string (becomes one user message) or a list of chat
messages [{"role": "...", "content": "..."}] (needed for the tool loop).
"""
from __future__ import annotations

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # locked decision
SEED = 42                                    # locked decision
MAX_NEW_TOKENS = 128                         # locked decision

_tok = None
_model = None


def set_seed(seed: int = SEED) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(name: str = MODEL_NAME):
    """Load tokenizer + model once (fp16 on GPU, fp32 on CPU)."""
    global _tok, _model
    if _model is not None:
        return _tok, _model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(SEED)
    _tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:                      # newer transformers use `dtype`, older ones `torch_dtype`
        _model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        _model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    _model.to("cuda" if torch.cuda.is_available() else "cpu")
    _model.eval()
    return _tok, _model


def _to_messages(prompt, system=None):
    if isinstance(prompt, str):
        msgs = [{"role": "user", "content": prompt}]
    else:
        msgs = list(prompt)
    if system and not (msgs and msgs[0]["role"] == "system"):
        msgs = [{"role": "system", "content": system}] + msgs
    return msgs


def _run(prompt, temperature, top_p, top_k, system, max_new_tokens, seed, want_scores):
    import torch

    tok, model = load_model()
    msgs = _to_messages(prompt, system)
    text_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text_in, return_tensors="pt").to(model.device)

    # temperature == 0 -> greedy. top_p / top_k alone switch sampling on at T=1.0.
    sampling = bool(temperature and temperature > 0) or top_p is not None or top_k is not None
    # Qwen ships hidden defaults (repetition_penalty 1.1, top_k 20, top_p 0.8, temperature 0.7).
    # Set everything explicitly so our runs match the rest of the team's notebooks.
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, repetition_penalty=1.0)
    if sampling:
        kwargs.update(do_sample=True, temperature=float(temperature) if temperature else 1.0,
                      top_p=float(top_p) if top_p is not None else 1.0,
                      top_k=int(top_k) if top_k is not None else 0)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    if seed is not None:
        set_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **enc, output_scores=want_scores, return_dict_in_generate=want_scores, **kwargs
        )

    seqs = out.sequences if want_scores else out
    new_ids = seqs[0, enc["input_ids"].shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    if not want_scores:
        return text

    trans = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    lps = trans[0].float().cpu().tolist()
    ids = new_ids.tolist()
    # drop the trailing end-of-text token(s) so they do not dilute confidence
    while ids and ids[-1] in (tok.eos_token_id, tok.pad_token_id):
        ids.pop()
        lps.pop()
    tokens = [tok.decode([i]) for i in ids]
    return {"text": text, "tokens": tokens, "logprobs": lps[: len(tokens)]}


def generate(prompt, temperature=0, top_p=None, top_k=None, system=None,
             max_new_tokens=MAX_NEW_TOKENS, seed=None) -> str:
    """Return the model's text for a prompt (the group contract)."""
    return _run(prompt, temperature, top_p, top_k, system, max_new_tokens, seed, False)


def generate_with_logprobs(prompt, temperature=0, top_p=None, top_k=None, system=None,
                           max_new_tokens=MAX_NEW_TOKENS, seed=None) -> dict:
    """Same as generate() but also returns per-token log-probabilities."""
    return _run(prompt, temperature, top_p, top_k, system, max_new_tokens, seed, True)
