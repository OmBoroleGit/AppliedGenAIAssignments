
from src.model import generate_answer

DEFAULT_SIMILARITY_THRESHOLD = 0.20


def answer_question(
    question,
    retriever,
    tokenizer,
    model,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
):
    """
    Retrieve context and generate a validated answer.
    Abstain when no passage meets the retrieval threshold.
    """
    results = retriever.retrieve(question)

    if not results:
        return {
            "answerable": False,
            "answer": "",
            "evidence": "",
            "confidence": 0.0,
        }

    best_result = results[0]

    if best_result["score"] < similarity_threshold:
        return {
            "answerable": False,
            "answer": "",
            "evidence": "",
            "confidence": 0.0,
        }

    return generate_answer(
        context=best_result["context"],
        question=question,
        tokenizer=tokenizer,
        model=model,
    )
