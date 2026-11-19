"""Regression tests for independent trigonometry solving."""

import unittest

from model.solver import solve_trigonometry


class GivenRatioExpressionTests(unittest.TestCase):
    def _solve(self, question):
        result = solve_trigonometry(question)
        self.assertIsNotNone(result)
        self.assertEqual(result["subject"], "trigonometry")
        self.assertEqual(
            result["algorithm"],
            "Independent Given-Ratio Trigonometry Solver",
        )
        return result

    def test_sine_ratio_with_unparenthesized_cosine_expression(self):
        result = self._solve(
            "If sin θ = 3/5 and θ is an acute angle, find the value of: "
            "(1 - cos θ) / (1 + cos θ)"
        )

        self.assertEqual(result["answer"], "1/9")
        self.assertIn("cos(θ) = √(1 - (3/5)²) = 4/5", str(result["solution_steps"]))

    def test_sine_ratio_with_parenthesized_cosine_expression(self):
        result = self._solve(
            "Given sin(θ) = 3/5 and θ is acute, evaluate "
            "(1 - cos(θ)) / (1 + cos(θ))"
        )

        self.assertEqual(result["answer"], "1/9")


if __name__ == "__main__":
    unittest.main()
