# Decoding comparison (E3)

Same prompt and 30 questions (15 answerable, 15 unanswerable); greedy run twice, every sampling setting 5 times with
different seeds. Qwen's hidden defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1) are switched off,
so each row changes only what its name says. Chart: `results/decoding_chart.png`.

| decoding | runs | F1_answerable | hallucination_rate | over_abstention_rate | json_validity | agreement | distinct_answers | F1_std_across_runs |
|---|---|---|---|---|---|---|---|---|
| greedy | 2 | 65.3 | 20.0 | 20.0 | 93.3 | 100.0 | 1.0 | 0.0 |
| temp_0.3 | 5 | 60.5 | 12.0 | 24.0 | 94.7 | 92.7 | 1.3 | 3.1 |
| temp_0.7 | 5 | 48.0 | 14.7 | 34.7 | 94.7 | 80.7 | 1.7 | 2.7 |
| temp_1.0 | 5 | 46.7 | 18.7 | 37.3 | 94.0 | 75.3 | 2.0 | 4.2 |
| temp_0.7_top_p_0.9 | 5 | 50.4 | 14.7 | 34.7 | 94.0 | 83.3 | 1.5 | 1.6 |
| temp_0.7_top_k_50 | 5 | 48.0 | 14.7 | 34.7 | 94.7 | 80.7 | 1.7 | 2.7 |
| temp_1.0_top_p_0.9 | 5 | 46.9 | 18.7 | 37.3 | 94.7 | 77.3 | 1.8 | 3.2 |

*agreement* = share of runs giving the most common answer (100 = identical every run); *distinct_answers* = different
answers per question; *F1_std_across_runs* = run-to-run spread.

**Findings.** Greedy has the best answerable F1 and is perfectly stable. Higher temperature lowers F1 and agreement. The
lower hallucination of sampling comes with more refusals of answerable questions (over-abstention 20% → 37%), i.e. random
refusals rather than better judgement. Top-p 0.9 restores some stability at temperature 0.7. Top-k 50 gives results
identical to temperature 0.7 alone: with shared seeds, this shows the probability mass sits in far fewer than 50 tokens.
→ **Greedy decoding is used in the final system.**

## Output example — same question, different runs
*How did the black death make it to the Mediterranean and Europe?*

| decoding | run | pred_answer | error_type |
|---|---|---|---|
| greedy | 0 | along the Silk Road | partial / over-long span |
| greedy | 1 | along the Silk Road | partial / over-long span |
| temp_0.3 | 0 | along the Silk Road | partial / over-long span |
| temp_0.3 | 1 | along the Silk Road | partial / over-long span |
| temp_0.3 | 2 | along the Silk Road | partial / over-long span |
| temp_0.3 | 3 | from there, it was most likely carried by Oriental rat fleas living on the black rats that were regular passengers on merchant ships | partial / over-long span |
| temp_0.3 | 4 | along the Silk Road | partial / over-long span |
| temp_0.7 | 0 | by 1343 | wrong answer |
| temp_0.7 | 1 | reaching Crimea by 1343 | wrong answer |
| temp_0.7 | 2 | along the Silk Road | partial / over-long span |
| temp_0.7 | 3 | from there, it was most likely carried by Oriental rat fleas living on the black rats that were regular passengers on merchant ships. | partial / over-long span |
| temp_0.7 | 4 | along the Silk Road | partial / over-long span |
| temp_0.7_top_k_50 | 0 | by 1343 | wrong answer |
| temp_0.7_top_k_50 | 1 | reaching Crimea by 1343 | wrong answer |
| temp_0.7_top_k_50 | 2 | along the Silk Road | partial / over-long span |
| temp_0.7_top_k_50 | 3 | from there, it was most likely carried by Oriental rat fleas living on the black rats that were regular passengers on merchant ships. | partial / over-long span |
| temp_0.7_top_k_50 | 4 | along the Silk Road | partial / over-long span |
| temp_0.7_top_p_0.9 | 0 | by 1343 | wrong answer |
| temp_0.7_top_p_0.9 | 1 | from there | wrong answer |
| temp_0.7_top_p_0.9 | 2 | along the Silk Road | partial / over-long span |
| temp_0.7_top_p_0.9 | 3 | from there, it was most likely carried by Oriental rat fleas living on the black rats that were regular passengers on merchant ships. | partial / over-long span |
| temp_0.7_top_p_0.9 | 4 | along the Silk Road | partial / over-long span |
| temp_1.0 | 0 | by 1343 | wrong answer |
| temp_1.0 | 1 | reaching Crimea by 1343 | wrong answer |
| temp_1.0 | 2 | nan | over-abstention |
| temp_1.0 | 3 | from there, it was most likely carried by Oriental rat fleas living on the black rats that were regular passengers on merchant ships. | partial / over-long span |
| temp_1.0 | 4 | along the Silk Road, reaching Crimea | partial / over-long span |
| temp_1.0_top_p_0.9 | 0 | by 1343 | wrong answer |
| temp_1.0_top_p_0.9 | 1 | reaching Crimea by 1343 | wrong answer |
| temp_1.0_top_p_0.9 | 2 | nan | over-abstention |
| temp_1.0_top_p_0.9 | 3 | from there, it was most likely carried by Oriental rat fleas living on the black rats that were regular passengers on merchant ships. | partial / over-long span |
| temp_1.0_top_p_0.9 | 4 | along the Silk Road | partial / over-long span |

## Context noise (same 100 questions, greedy)
| index | F1_overall | EM_overall | F1_answerable | hallucination_rate | over_abstention_rate | evidence_faithfulness | json_validity |
|---|---|---|---|---|---|---|---|
| clean (1 paragraph) | 73.6 | 70.0 | 75.1 | 28.0 | 16.0 | 82.1 | 94.0 |
| noisy (+2 distractors) | 64.6 | 62.0 | 61.2 | 32.0 | 24.0 | 70.4 | 94.0 |
