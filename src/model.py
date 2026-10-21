
import json
import re

from src.schema import validate_output, OutputValidationError


SYSTEM_PROMPT = """
You are a context-grounded question answering assistant.

Use only the supplied context to answer the question.

Return exactly one JSON object with these four fields:
{
  "answerable": true,
  "answer": "short answer",
  "evidence": "exact supporting passage copied from context",
  "confidence": 0.95
}

Rules:
- If the context supports the answer, answer using the context.
- Evidence must be copied verbatim from the context.
- The answer must appear in the evidence.
- If the context does not support the answer, abstain:
  {"answerable": false, "answer": "", "evidence": "", "confidence": 0.0}
- Confidence must be a number between 0 and 1.
- Do not include Markdown fences or text outside the JSON.
"""


def build_messages(context, question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "Return the required JSON object."
            ),
        },
    ]


def generate_answer(context, question, tokenizer, model):
    """Generate and validate a structured answer using a loaded HF model."""
    messages = build_messages(context, question)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    import torch

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    # Permit accidental Markdown fences, but still require valid JSON.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(
            f"Model did not return valid JSON. Output: {raw_output!r}"
        ) from exc

    # Enforce consistent abstention semantics if the model explicitly abstains.
    if isinstance(parsed, dict) and parsed.get("answerable") is False:
        parsed["answer"] = ""
        parsed["evidence"] = ""

    return validate_output(parsed, context)
