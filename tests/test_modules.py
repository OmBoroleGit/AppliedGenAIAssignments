"""Offline tests (no GPU / no model): python -m unittest discover tests -v"""
import json, math, unittest
from src.schema import validate_output, extract_json, answer_json, ABSTAIN
from src.confidence import confidence_from_logprobs, calibration_table
from src import tools

CTX = "Alexander Graham Bell patented the telephone in 1876. He was born in Edinburgh."
GOOD = '{"answerable": true, "answer": "1876", "evidence": "Alexander Graham Bell patented the telephone in 1876.", "confidence": 0.8}'


class Schema(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(validate_output(GOOD, CTX).ok)

    def test_fenced_and_chatter(self):
        self.assertTrue(validate_output("Sure!\n```json\n" + GOOD + "\n```", CTX).ok)

    def test_no_json(self):
        r = validate_output("I think it was 1876", CTX); self.assertFalse(r.ok); self.assertIn("R1", r.errors[0])

    def test_invented_evidence(self):
        bad = GOOD.replace("Alexander Graham Bell patented the telephone in 1876.", "Bell invented it in 1877.")
        r = validate_output(bad, CTX); self.assertFalse(r.ok); self.assertTrue(any("R5" in e for e in r.errors))

    def test_abstain_must_have_empty_answer(self):
        r = validate_output('{"answerable": false, "answer": "1876", "evidence": "", "confidence": 0.1}', CTX)
        self.assertFalse(r.ok); self.assertTrue(any("R6" in e for e in r.errors))

    def test_valid_abstain(self):
        self.assertTrue(validate_output(json.dumps(ABSTAIN), CTX).ok)

    def test_team_format_no_confidence_and_refusal_phrase(self):
        # the team's prompts: 3 keys only, abstain answer = "not in the document"
        r = validate_output('{"answerable": false, "answer": "not in the document", "evidence": ""}', CTX)
        self.assertTrue(r.ok); self.assertEqual(r.parsed.answer, "")
        r = validate_output('{"answerable": true, "answer": "1876", "evidence": "Alexander Graham Bell patented the telephone in 1876."}', CTX)
        self.assertTrue(r.ok)

    def test_true_but_refusal_answer(self):
        r = validate_output('{"answerable": true, "answer": "not in the document", "evidence": "x"}', CTX)
        self.assertFalse(r.ok); self.assertTrue(any("R7" in e for e in r.errors))

    def test_bad_types(self):
        self.assertFalse(validate_output('{"answerable": "yes", "answer": "", "evidence": "", "confidence": 0.5}', CTX).ok)
        self.assertFalse(validate_output('{"answerable": false, "answer": "", "evidence": "", "confidence": 1.7}', CTX).ok)

    def test_nested_braces_in_prose(self):
        self.assertEqual(extract_json('x {bad} y ' + GOOD)["answer"], "1876")


def fake_gen(replies):
    """Mock generate_with_logprobs: returns scripted replies with fake token probs."""
    it = iter(replies)
    def gen(messages, **kw):
        text = next(it)
        toks = list(text)  # one char per token
        return {"text": text, "tokens": toks, "logprobs": [math.log(0.9)] * len(toks)}
    return gen


class Retry(unittest.TestCase):
    def test_retry_fixes_invalid(self):
        out = answer_json([{"role": "user", "content": "q"}], CTX, fake_gen(["nope", GOOD]))
        self.assertTrue(out["json_valid"]); self.assertEqual(out["attempts"], 2)
        self.assertEqual(out["confidence_source"], "token_probs")
        self.assertAlmostEqual(out["result"]["confidence"], 0.9, places=3)
        # evaluate.py reads raw_output: it must be the FINAL accepted reply, valid JSON
        self.assertEqual(json.loads(out["raw_output"])["answer"], "1876")

    def test_gives_up_after_one_retry(self):
        out = answer_json([{"role": "user", "content": "q"}], CTX, fake_gen(["nope", "still nope"]))
        self.assertFalse(out["json_valid"]); self.assertEqual(out["result"], ABSTAIN)
        self.assertEqual(out["attempts"], 2)


class Confidence(unittest.TestCase):
    def test_answer_tokens_used(self):
        text = '{"answerable": true, "answer": "1876", "evidence": "e", "confidence": 0.9}'
        toks = list(text); lps = [math.log(0.99)] * len(toks)
        i = text.index("1876")
        for j in range(i, i + 4): lps[j] = math.log(0.5)       # weak answer tokens
        c = confidence_from_logprobs(toks, lps)
        self.assertTrue(0.5 < c < 0.99)

    def test_abstain_uses_flag_only(self):
        text = '{"answerable": false, "answer": "", "evidence": "", "confidence": 0.0}'
        toks = list(text); lps = [math.log(0.1)] * len(toks)
        i = text.index("false")
        for j in range(i, i + 5): lps[j] = math.log(0.8)
        self.assertAlmostEqual(confidence_from_logprobs(toks, lps), 0.8, places=3)

    def test_ece(self):
        rows, ece = calibration_table([0.9, 0.9, 0.1, 0.1], [True, True, False, False])
        self.assertLess(ece, 0.11)


PASSAGES = [
    {"title": "Telephone", "text": CTX},
    {"title": "Photosynthesis", "text": "Photosynthesis converts light energy into chemical energy in chloroplasts."},
    {"title": "Rome", "text": "Rome is the capital of Italy and was founded according to legend in 753 BC."},
]


class Tools(unittest.TestCase):
    def setUp(self):
        tools.init_index([{"passage_id": i, **p} for i, p in enumerate(PASSAGES)])

    def test_bm25_ranks_right_passage_first(self):
        r = tools.retrieve_context("When was the telephone patented?", 2)
        self.assertEqual(r["passages"][0]["title"], "Telephone")
        r = tools.retrieve_context("capital of Italy", 1)
        self.assertEqual(r["passages"][0]["title"], "Rome")

    def test_args_validated(self):
        with self.assertRaises(Exception): tools.retrieve_context("hi", 9)

    def test_loop_model_calls_tool(self):
        call = json.dumps({"tool_call": {"name": "retrieve_context", "arguments": {"question": "When was the telephone patented?", "top_k": 2}}})
        out = tools.answer_with_tools("When was the telephone patented?", fake_gen([call, GOOD]))
        self.assertTrue(out["json_valid"]); self.assertTrue(out["trace"]["tool_called"])
        self.assertEqual(out["result"]["answer"], "1876")

    def test_loop_forced_fallback(self):
        out = tools.answer_with_tools("When was the telephone patented?", fake_gen(["I don't know", GOOD]))
        self.assertTrue(out["trace"]["forced_fallback"]); self.assertTrue(out["json_valid"])

    def test_loop_direct_answer_with_paragraph(self):
        out = tools.answer_with_tools("When was the telephone patented?", fake_gen([GOOD]), paragraph=CTX)
        self.assertFalse(out["trace"]["tool_called"]); self.assertTrue(out["json_valid"])

    def test_bad_tool_args_do_not_crash(self):
        call = json.dumps({"tool_call": {"name": "retrieve_context", "arguments": {"question": "x", "top_k": 99}}})
        out = tools.answer_with_tools("q?", fake_gen([call, json.dumps(ABSTAIN)]))
        self.assertIsNotNone(out["trace"]["tool_error"])


if __name__ == "__main__":
    unittest.main()
