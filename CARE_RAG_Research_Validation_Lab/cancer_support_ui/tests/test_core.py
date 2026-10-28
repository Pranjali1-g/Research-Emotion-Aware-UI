import unittest
from core import analyze_message, retrieve_evidence, build_demo_response

class CoreTests(unittest.TestCase):
    def test_urgent_message_escalates(self):
        result = analyze_message("I cannot breathe and have severe chest pain right now.")
        self.assertEqual(result["risk"], "escalate")

    def test_treatment_change_is_scope_limited(self):
        result = analyze_message("Should I stop my chemotherapy medicine and change the dose?")
        self.assertEqual(result["risk"], "scope")

    def test_general_question_low_route(self):
        result = analyze_message("What questions should I ask my doctor before an appointment?")
        self.assertEqual(result["risk"], "low")

    def test_retrieval_returns_list(self):
        docs = retrieve_evidence("questions for doctor appointment", "appointments_and_questions", top_k=3)
        self.assertIsInstance(docs, list)
        self.assertLessEqual(len(docs), 3)

    def test_scope_response_does_not_recommend_medication_change(self):
        result = analyze_message("Should I stop my chemotherapy medicine?")
        response = build_demo_response("Should I stop?", result, [])
        self.assertIn("can’t recommend", response)

if __name__ == "__main__":
    unittest.main()
