
import json
import re

REQUIRED_KEYS = {"answerable", "answer", "evidence", "confidence"}


class OutputValidationError(ValueError):
    """Raised when model output violates the required schema."""


def _normalise(text):
    return re.sub(r"\s+", " ", text).strip()


def validate_output(raw_output, context):
    """Parse and validate a model response against the required schema."""
    try:
        data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    except (json.JSONDecodeError, TypeError) as exc:
        raise OutputValidationError("Output is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise OutputValidationError("Output must be a JSON object.")

    if set(data) != REQUIRED_KEYS:
        raise OutputValidationError(
            f"Expected exactly these keys: {sorted(REQUIRED_KEYS)}"
        )

    if not isinstance(data["answerable"], bool):
        raise OutputValidationError("'answerable' must be a boolean.")

    if not isinstance(data["answer"], str):
        raise OutputValidationError("'answer' must be a string.")

    if not isinstance(data["evidence"], str):
        raise OutputValidationError("'evidence' must be a string.")

    confidence = data["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise OutputValidationError(
            "'confidence' must be a number between 0 and 1."
        )

    answer = data["answer"].strip()
    evidence = data["evidence"].strip()

    if not data["answerable"]:
        if answer or evidence:
            raise OutputValidationError(
                "Abstentions must have empty answer and evidence."
            )
    else:
        if not answer or not evidence:
            raise OutputValidationError(
                "Answerable responses need both answer and evidence."
            )

        if _normalise(evidence) not in _normalise(context):
            raise OutputValidationError(
                "Evidence is not an exact passage from the context."
            )

        if _normalise(answer).casefold() not in _normalise(evidence).casefold():
            raise OutputValidationError(
                "Answer does not appear in the evidence."
            )

    return {
        "answerable": data["answerable"],
        "answer": answer,
        "evidence": evidence,
        "confidence": float(confidence),
    }
