"""Dataset-wide safety checks for exact-question responses."""

import unittest

from model import solver
from model.model import model
from utils import dataset_loader


class DatasetIntegrityTests(unittest.TestCase):
    def test_every_loaded_record_is_answerable_or_explicitly_flagged(self):
        """Never choose an arbitrary solution for duplicate conflicting rows."""
        checked = 0
        conflicts = 0

        for subject in dataset_loader.get_subjects():
            entries = dataset_loader.get_subject(subject)
            for entry in entries:
                question = entry["question"]
                formatted = solver.format_exact_dataset_solution(entry, subject)
                status = solver.get_exact_dataset_match_status(question, entries)

                self.assertTrue(status["exact_match"], question)
                self.assertTrue(formatted.get("answer"), question)
                if status["ambiguous"]:
                    conflicts += 1
                    self.assertIsNone(status["entry"])
                else:
                    self.assertIsNotNone(status["entry"])
                checked += 1

        self.assertGreater(checked, 0)
        # The dataset may be corrected later, so do not pin the number of
        # conflicts.  The assertion above is the important behavior: any
        # conflict is surfaced instead of selecting a random duplicate row.
        self.assertGreaterEqual(conflicts, 0)

    def test_conflicting_knn_dataset_row_is_not_answered_arbitrarily(self):
        question = (
            "Classify test instance {'CGPA': 9.7, 'Assessment': 88, "
            "'Project': 8} using k-NN (k=3, Euclidean distance)."
        )
        entries = dataset_loader.get_subject("knn")
        status = solver.get_exact_dataset_match_status(question, entries)
        result = model.solve(question)

        if status["ambiguous"]:
            self.assertEqual(result["type"], "needs_input")
            self.assertTrue(result["dataset_conflict"])
            self.assertIn("Fail", result["conflicting_answers"])
            self.assertIn("Pass", result["conflicting_answers"])
        else:
            self.assertEqual(result["type"], "exact_dataset_match")

    def test_raw_question_is_used_when_routing_normalization_changes_words(self):
        question = (
            "Multiply matrices A = [[4, -1], [-5, 0]] and "
            "B = [[2, -3], [-5, -3]]."
        )
        result = model.solve(question, subject_hint="matrix")

        self.assertEqual(result["type"], "exact_dataset_match")
        self.assertEqual(result["answer"], "[[13, -9], [-10, 15]]")

    def test_conflicting_trigonometry_rows_use_independent_calculation(self):
        result = model.solve(
            "Can you determine Convert π radians to degrees.",
            subject_hint="trigonometry",
        )

        self.assertEqual(result["type"], "solution")
        self.assertEqual(result["answer"], "180°")
        self.assertEqual(
            result["algorithm"],
            "Independent Radian-to-Degree Solver",
        )


if __name__ == "__main__":
    unittest.main()
