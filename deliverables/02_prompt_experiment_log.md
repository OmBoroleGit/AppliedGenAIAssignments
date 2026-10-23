# Prompt experiment log (E1 baseline, E2 prompt comparison)

All prompts are produced by one reusable template, `build_prompt(context, question, variant, few_shot_examples)` in
`src/prompts.py` (requirement 11). Model: Qwen2.5-1.5B-Instruct, greedy decoding, `max_new_tokens=128`,
`repetition_penalty=1.0`, the frozen 200 questions. Every raw output is saved in `results/exp_prompt_<variant>.csv`.

## The prompts (exact messages sent to the model) and the reason for each design
### baseline
**Why:** Deliberately minimal: no format, no evidence, no rules. It shows what the raw model does, so every later improvement can be measured against it (requirement 5).

```text
[system]
Answer the question based on the context. If the context does not contain the answer, say 'not in the document'.

[user]
Context: <paragraph>

Question: <question>
```

### zero_shot
**Why:** Zero-shot prompt enforcing JSON output structure. Tests whether instruction-following alone enables the model to structure responses and identify unanswerable questions without in-context demonstrations.

```text
[system]
You are a document assistant. Answer the question using ONLY the provided context.
If the answer cannot be found in the context, mark answerable as false and say 'not in the document'.
The answer must be a SHORT phrase (a few words) copied exactly from the context, not a full sentence.
You MUST respond ONLY with a valid JSON object matching this schema:
{"answerable": <bool>, "answer": "<exact answer or not in the document>", "evidence": "<exact sentence from context containing proof or empty string>"}

[user]
Context: <paragraph>

Question: <question>
```

### few_shot
**Why:** Uses the same instructions as zero-shot (so only the examples differ), plus 3 answerable and 1 unanswerable examples from the training split. Demonstrates the exact expected JSON schema, evidence extraction format, and the behavior of setting 'answerable: false' when context is insufficient.

```text
[system]
You are a document assistant. Answer the question using ONLY the provided context.
If the answer cannot be found in the context, mark answerable as false and say 'not in the document'.
The answer must be a SHORT phrase (a few words) copied exactly from the context, not a full sentence.
You MUST respond ONLY with a valid JSON object matching this schema:
{"answerable": <bool>, "answer": "<exact answer or not in the document>", "evidence": "<exact sentence from context containing proof or empty string>"}

[user]
Context: Purebred dogs of one breed are genetically distinguishable from purebred dogs of other breeds, but the means by which kennel clubs classify dogs is unsystematic. Systematic analyses of the dog genome has revealed only four major types of dogs that can be said to be statistically distinct. These include the "old world dogs" (e.g., Malamute and Shar Pei), "Mastiff"-type (e.g., English Mastiff), "herding"-type (e.g., Border Collie), and "all others" (also called "modern"- or "hunting"-type).

Question: A scientific study of dog genetics has shown only how many  types of dogs being determinably distinct?

[assistant]
{"answerable": true, "answer": "four", "evidence": "Systematic analyses of the dog genome has revealed only four major types of dogs that can be said to be statistically distinct"}

[...]
(6 more example messages)

[user]
Context: <paragraph>

Question: <question>
```

### role_based
**Why:** Persona-based prompt framing the model as a 'meticulous reference librarian and strict fact-checker'. Encourages conservative, evidence-grounded inference and deters speculative leaps.

```text
[system]
You are a meticulous reference librarian and strict fact-checker. Your professional duty is to provide strictly verified answers based exclusively on the provided document paragraph.
You never speculate, extrapolate, or rely on prior assumptions. If a query is not directly answered by the document, you acknowledge that fact immediately.
The answer must be a SHORT phrase (a few words) copied exactly from the context, not a full sentence.
Always respond in valid JSON format:
{"answerable": <bool>, "answer": "<factually verified answer>", "evidence": "<exact supporting sentence>"}

[user]
Context: <paragraph>

Question: <question>
```

### constraint_based
**Why:** Constraint-heavy prompt with explicit negative guardrails ('Do NOT use outside knowledge', 'Do NOT extrapolate', 'If not stated, you MUST abstain'). Directly penalizes hallucinations and enforces strict grounding in the context.

```text
[system]
Strict Operational Constraints:
1. GROUNDING: Use ONLY the provided context. Do NOT consult external world knowledge.
2. ABSTENTION: If the text does NOT explicitly answer the question, set 'answerable' to false, 'answer' to 'not in the document', and 'evidence' to ''.
3. ANSWER: The answer must be a SHORT phrase (a few words) copied exactly from the context.
4. EVIDENCE: When answerable is true, 'evidence' MUST be the verbatim sentence from the context.
5. FORMAT: Output NOTHING except a single, valid JSON object with keys: answerable, answer, evidence.
6. NO HALLUCINATION: Guessing is strictly penalized.

[user]
Context: <paragraph>

Question: <question>
```

The few-shot variant uses the zero-shot instructions plus 4 worked examples from the train split (3 answerable, 1
unanswerable) given as earlier user/assistant turns, so the difference between zero-shot and few-shot measures the
examples alone.

## E1 — three baseline outputs
| question | gold_answers | raw_output |
|---|---|---|
| Who started the Yuan dynasty? | ["Kublai Khan", "Kublai Khan", "Kublai Khan"] | Kublai Khan |
| How large is the San Joaquin Valley? | [] | The San Joaquin Valley is approximately 220 miles (350 km) northwest of Los Angeles. |
| What did the concept hope to achieve? | [] | provide a fault-tolerant, efficient routing method for telecommunication messages |

## E2 — comparison on 200 questions (%)
| index | F1_overall | EM_overall | F1_answerable | hallucination_rate | over_abstention_rate | evidence_faithfulness | json_validity | flag_conflicts |
|---|---|---|---|---|---|---|---|---|
| baseline | 62.5 | 52.5 | 76.0 | 51.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| zero_shot | 67.4 | 61.5 | 85.8 | 51.0 | 3.0 | 80.4 | 99.5 | 7.5 |
| few_shot | 77.0 | 73.0 | 75.1 | 21.0 | 17.0 | 81.7 | 98.0 | 1.0 |
| role_based | 65.7 | 59.5 | 88.5 | 57.0 | 2.0 | 76.8 | 95.5 | 13.0 |
| constraint_based | 71.6 | 67.0 | 73.2 | 30.0 | 15.0 | 86.1 | 36.0 | 3.5 |

**Selection rule (stated before choosing):** among prompts with JSON validity ≥ 90%, the highest overall F1, ties broken
by lower hallucination → **few-shot** (`results/best_prompt.txt`). Few-shot vs baseline: +14.5 F1, 95% CI [7.0, 21.8],
McNemar p < 0.001 (notebook 07).
