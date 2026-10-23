# Tool / function calling (E5)

**Tool:** `retrieve_context(question, top_k)` — BM25 search over a library of Wikipedia paragraphs (20 SQuAD articles,
676 paragraphs), implemented in `src/tools.py`. Input and output schemas: `05_tool_spec.json`. Arguments are validated
with pydantic before execution (`top_k` limited to 1–5, query at least 3 characters).

**Workflow (brief §5):** user request → LLM → tool decision → tool result → final response.
1. The user asks a question **without** a paragraph.
2. The model receives the tool description and replies with a tool call:
   `{"tool_call": {"name": "retrieve_context", "arguments": {"question": "...", "top_k": 3}}}`
3. Python validates the arguments and runs BM25; the top passages are returned.
4. The final answer is written with the same few-shot prompt and validator as paragraph mode, and its evidence is
   checked against the retrieved passages.

## Representative invocation
- **User question:** Where according to gross state product does Victoria rank in Australia?
- **Model called the tool itself:** True
- **Gold paragraph among retrieved passages:** True
- **Final response:** `{"answerable": true, "answer": "second", "evidence": "Victoria's total gross state product (GSP) is ranked second in Australia", "confidence": 0.95}`

## Results on 112 questions from the indexed articles (%)
| index | F1_overall | F1_answerable | hallucination_rate | over_abstention_rate | evidence_faithfulness | gold_paragraph_retrieved_% | model_called_tool_% |
|---|---|---|---|---|---|---|---|
| tool mode as in notebook 05 | 51.6 | 12.1 | 8.9 | 83.9 | 50.0 |  |  |
| fixed tool mode (few-shot + harness) | 60.3 | 54.5 | 33.9 | 33.9 | 58.9 | 78.6 | 99.1 |
| gold paragraph given (row D, same questions) | 73.7 | 68.8 | 21.4 | 26.8 | 86.8 |  |  |

The first row is the original tool mode (final answer written with a zero-shot tool prompt); the second uses the
few-shot prompt chosen in E2 for the final answer. The gap to the third row is the measured cost of retrieval
compared with being given the right paragraph.
