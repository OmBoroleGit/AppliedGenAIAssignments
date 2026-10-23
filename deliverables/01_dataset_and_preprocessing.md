# Dataset description and preprocessing record

**Dataset:** SQuAD v2.0 (`rajpurkar/squad_v2` on Hugging Face) — crowd-sourced questions on Wikipedia paragraphs.
Answerable questions have a gold answer that is an exact span of the paragraph; **unanswerable** questions were written
adversarially to look answerable but have no answer in the paragraph (the correct output is to abstain).

**Fields used:** `id`, `title` (Wikipedia article), `context` (paragraph), `question`, `answers.text` (gold answers,
possibly several), `answers.answer_start`. Derived: `is_answerable` = at least one gold answer.

## Subset used for all experiments (frozen)
| Item | Value |
|---|---|
| Source split | validation (11,873 questions, 35 articles) |
| Evaluation set | 200 questions: 100 answerable + 100 unanswerable |
| Articles covered | 35 (all validation articles) |
| Few-shot examples | 4 from the **train** split (3 answerable + 1 unanswerable) |
| Random seed | 42 |
| Files | `data/eval_set.jsonl`, `data/fewshot_pool.jsonl` |

## Preprocessing steps (notebook 01)
1. Load the validation split; mark each question answerable / unanswerable.
2. To spread questions across articles, keep at most 3 questions per article per type, then sample 100 answerable and
   100 unanswerable questions (seed 42) and shuffle. The file is then **frozen**: every experiment uses these 200 questions.
3. Few-shot pool: from the train split, paragraphs shorter than 600 characters; 3 answerable + 1 unanswerable (seed 42).
   Evidence = the sentence containing the gold answer, found from `answer_start`.
4. Leakage check: train and validation use different Wikipedia articles, so no few-shot paragraph or question can
   appear in the evaluation set (also asserted in code).
5. No text cleaning or truncation is applied: paragraphs are passed to the model as written.

## Size of the inputs (Qwen tokenizer)
| index | context_tokens | question_tokens |
|---|---|---|
| count | 200.0 | 200.0 |
| mean | 177.6 | 11.5 |
| std | 74.2 | 3.8 |
| min | 82.0 | 6.0 |
| 25% | 135.0 | 9.0 |
| 50% | 168.0 | 10.0 |
| 75% | 206.0 | 14.0 |
| max | 502.0 | 23.0 |

## Representative examples
| question | gold_answers | is_answerable | title |
|---|---|---|---|
| Who started the Yuan dynasty? | ['Kublai Khan'] | True | Yuan_dynasty |
| By what main attribute are computational problems classified utilizing computational complexity theory?  | ['inherent difficulty', 'their inherent difficulty'] | True | Computational_complexity_theory |
| How large is the San Joaquin Valley? | — (unanswerable) | False | Fresno,_California |
| What did the concept hope to achieve? | — (unanswerable) | False | Packet_switching |

## Dataset challenges
- Unanswerable questions reuse words from the paragraph, so keyword matching produces confident wrong answers.
- Multiple valid gold answers per question — scoring takes the best match (Exact Match and F1, official normalisation).
- Single domain (Wikipedia); results may not transfer to other document types.
- The model may have seen Wikipedia/SQuAD in pre-training (possible contamination), so systems are compared on the same
  questions rather than by absolute accuracy.
