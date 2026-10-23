"""Offline tests for src/harness.py (no GPU / no model): python -m unittest discover tests -v"""
import json, unittest
import pandas as pd
from src.harness import (HarnessConfig, answer_harness, answer_with_retrieval, apply_threshold, check,
                         evidence_coverage, guard_injection)
from src.prompts import ADVERSARIAL_CASES
from src import tools

CTX = "Warsaw stands on the Vistula River in east-central Poland, roughly 260 kilometres from the Baltic Sea. It is the capital."
Q = "What river does Warsaw stand on?"
NEAR = '{"answerable": true, "answer": "Vistula River", "evidence": "Warsaw stands on the Vistula River in east central Poland, roughly 260 kilometres from the Baltic Sea."}'   # hyphen dropped
EXACT = '{"answerable": true, "answer": "Vistula River", "evidence": "Warsaw stands on the Vistula River in east-central Poland, roughly 260 kilometres from the Baltic Sea."}'
INVENTED = '{"answerable": true, "answer": "Vistula River", "evidence": "The Vistula flows through Warsaw."}'
ECHO = '{"answerable": true, "answer": "Warsaw", "evidence": "It is the capital."}'


def scripted(outputs):
    it = iter(outputs)
    return lambda msgs, **k: next(it)


def build(context, question):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]


class Sensors(unittest.TestCase):
    def test_strict_rejects_near_copy(self):
        _, errors, _ = check(NEAR, CTX, Q, HarnessConfig())
        self.assertTrue(any(e.startswith("R5") for e in errors))

    def test_fuzzy_accepts_near_copy(self):
        _, errors, warnings = check(NEAR, CTX, Q, HarnessConfig(fuzzy_evidence=True))
        self.assertEqual(errors, []); self.assertTrue(any("W3" in w for w in warnings))

    def test_fuzzy_still_rejects_invented(self):
        _, errors, _ = check(INVENTED, CTX, Q, HarnessConfig(fuzzy_evidence=True))
        self.assertTrue(any(e.startswith("R5") for e in errors))
        self.assertLess(evidence_coverage("The Vistula flows through Warsaw.", CTX), 0.9)

    def test_echo(self):
        _, errors, _ = check(ECHO, CTX, Q, HarnessConfig(echo_check=True))
        self.assertTrue(any(e.startswith("R8") for e in errors))


class Loop(unittest.TestCase):
    def test_no_harness_returns_raw(self):
        out = answer_harness(Q, CTX, build, scripted([INVENTED]), HarnessConfig(validate=False))
        self.assertEqual(out["raw_output"], INVENTED); self.assertEqual(out["attempts"], 1)

    def test_retry_fixes(self):
        out = answer_harness(Q, CTX, build, scripted([INVENTED, EXACT]), HarnessConfig(neutral_feedback=True))
        self.assertTrue(out["json_valid"]); self.assertEqual(out["attempts"], 2)
        self.assertEqual(out["result"]["answer"], "Vistula River")

    def test_gives_up_abstains_and_raw_output_agrees(self):
        out = answer_harness(Q, CTX, build, scripted([INVENTED, INVENTED]), HarnessConfig())
        self.assertFalse(out["result"]["answerable"])
        self.assertFalse(json.loads(out["raw_output"])["answerable"])   # scored decision = shown decision


class Guard(unittest.TestCase):
    def test_both_attacks_removed(self):
        for case in ADVERSARIAL_CASES:
            clean, removed = guard_injection(case["context"])
            self.assertTrue(removed)
            self.assertNotIn(case["injection_target"].lower(), clean.lower())
            self.assertIn(case["expected_grounded_answer"].lower(), clean.lower())

    def test_normal_text_untouched(self):
        clean, removed = guard_injection(CTX)
        self.assertEqual(removed, []); self.assertEqual(clean, CTX)


class Retrieval(unittest.TestCase):
    def test_tool_mode_uses_given_prompt(self):
        tools.init_index([{"passage_id": 0, "title": "Warsaw", "text": CTX},
                          {"passage_id": 1, "title": "Paris", "text": "Paris is the capital of France."}])
        seen = []
        def gen(msgs, **k):
            seen.append(msgs)
            return '{"tool_call": {"name": "retrieve_context", "arguments": {"question": "Warsaw river", "top_k": 1}}}' \
                if len(seen) == 1 else EXACT
        out = answer_with_retrieval(Q, build, gen, HarnessConfig())
        self.assertTrue(out["trace"]["tool_called_by_model"])
        self.assertEqual(seen[1][0]["content"], "sys")                 # final answer used build_messages
        self.assertIn("Vistula", out["context_used"])
        self.assertTrue(out["json_valid"])


class Threshold(unittest.TestCase):
    def test_threshold(self):
        d = pd.DataFrame({"pred_answerable": [True, True, False], "confidence": ["0.3", "0.9", ""],
                          "raw_output": ["a", "b", "c"], "pred_answer": ["x", "y", ""], "evidence": ["e", "f", ""]})
        t = apply_threshold(d, 0.5)
        self.assertEqual(list(t.pred_answerable), [False, True, False])
        self.assertIs(apply_threshold(d, 0)["raw_output"].iloc[0], "a")


if __name__ == "__main__":
    unittest.main()
