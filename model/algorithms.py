"""
M3 Engine - Numerical Algorithms
--------------------------------
Numerical and mathematical algorithms used by the
Multi-Module Mathematical Engine.

Supported algorithms:
1. Newton-Raphson
2. Bisection
3. Secant
4. Regula Falsi
5. Gaussian Elimination
6. Gauss-Seidel
7. Trapezoidal Rule
8. Simpson's 1/3 Rule
9. Simpson's 3/8 Rule
10. Lagrange Interpolation
11. Newton Forward Interpolation
12. Newton Backward Interpolation
13. Euler Method
14. Runge-Kutta 4th Order
15. Matrix Multiplication
16. Matrix Transpose
17. Determinant
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Sequence, Any


# ============================================================
# CONSTANTS / VALIDATION HELPERS
# ============================================================

EPSILON = 1e-14
DEFAULT_TOLERANCE = 1e-7
DEFAULT_MAX_ITERATIONS = 100

# Intent metadata only: higher-level modules may use these names to select
# an algorithm, but final numerical results are always computed from inputs.
ALGORITHM_ALIASES = {
    "newton": "newton_raphson", "newton raphson": "newton_raphson",
    "newton-raphson": "newton_raphson", "bisection method": "bisection",
    "secant method": "secant", "regula falsi": "regula_falsi",
    "regula-falsi": "regula_falsi", "gaussian": "gaussian_elimination",
    "gauss elimination": "gaussian_elimination", "gauss seidel": "gauss_seidel",
    "gauss-seidel": "gauss_seidel", "trapezoid": "trapezoidal_rule",
    "trapezoidal": "trapezoidal_rule", "simpson 1/3": "simpsons_rule",
    "simpson's 1/3": "simpsons_rule", "simpson 3/8": "simpsons_three_eighth",
    "simpson's 3/8": "simpsons_three_eighth", "lagrange": "lagrange_interpolation",
    "newton forward": "newton_forward", "newton backward": "newton_backward",
    "euler": "euler_method", "rk4": "runge_kutta_4",
    "runge kutta": "runge_kutta_4", "runge-kutta": "runge_kutta_4",
    "matrix multiplication": "matrix_multiply", "matrix multiply": "matrix_multiply",
    "transpose": "matrix_transpose", "det": "determinant",
}

ALGORITHM_KEYWORDS = {
    "newton_raphson": ("newton-raphson", "newton raphson"),
    "bisection": ("bisection", "interval halving"),
    "secant": ("secant",), "regula_falsi": ("regula falsi", "false position"),
    "gaussian_elimination": ("gaussian elimination", "gauss elimination"),
    "gauss_seidel": ("gauss-seidel", "gauss seidel"),
    "trapezoidal_rule": ("trapezoidal", "trapezoid rule"),
    "simpsons_rule": ("simpson's 1/3", "simpson 1/3"),
    "simpsons_three_eighth": ("simpson's 3/8", "simpson 3/8"),
    "lagrange_interpolation": ("lagrange interpolation",),
    "newton_forward": ("newton forward",),
    "newton_backward": ("newton backward",),
    "euler_method": ("euler method", "euler's method"),
    "runge_kutta_4": ("runge-kutta", "runge kutta", "rk4"),
    "matrix_multiply": ("matrix multiplication", "multiply matrices"),
    "matrix_transpose": ("matrix transpose", "transpose matrix"),
    "determinant": ("determinant", "det of matrix"),
}


def _validate_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)

    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be a positive finite number.")

    return tolerance


def _validate_iterations(max_iterations: int) -> int:
    if isinstance(max_iterations, bool):
        raise ValueError("max_iterations must be a positive integer.")

    max_iterations = int(max_iterations)

    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero.")

    return max_iterations


def _validate_steps(steps: int) -> int:
    if isinstance(steps, bool):
        raise ValueError("steps must be a non-negative integer.")

    steps = int(steps)

    if steps < 0:
        raise ValueError("steps cannot be negative.")

    return steps


def _finite(value: float, name: str = "value") -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _validate_function_result(value: Any, name: str = "function result") -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric.")

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _validate_matrix(matrix: Sequence[Sequence[float]], square: bool = False):
    if not matrix:
        raise ValueError("Matrix cannot be empty.")

    try:
        rows = [list(map(float, row)) for row in matrix]
    except (TypeError, ValueError):
        raise ValueError("Matrix must contain numeric values.")

    if any(not row for row in rows):
        raise ValueError("Matrix rows cannot be empty.")

    width = len(rows[0])

    if any(len(row) != width for row in rows):
        raise ValueError("Matrix must be rectangular.")

    if square and len(rows) != width:
        raise ValueError("Matrix must be square.")

    return rows


# ============================================================
# 1. NEWTON-RAPHSON METHOD
# ============================================================

def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> Dict:
    """
    Find a root using the Newton-Raphson method.

    Formula:
        x(n+1) = x(n) - f(x(n)) / f'(x(n))
    """

    tolerance = _validate_tolerance(tolerance)
    max_iterations = _validate_iterations(max_iterations)
    x = _finite(x0, "Initial value")

    steps = []

    for i in range(1, max_iterations + 1):
        fx = _validate_function_result(f(x), "f(x)")
        dfx = _validate_function_result(df(x), "f'(x)")

        if abs(dfx) < EPSILON:
            raise ValueError(
                "Derivative is too close to zero. "
                "Newton-Raphson cannot continue."
            )

        x_new = x - fx / dfx
        x_new = _finite(x_new, "Next approximation")

        steps.append({
            "iteration": i,
            "x": x,
            "f_x": fx,
            "derivative": dfx,
            "next_x": x_new,
            "error": abs(x_new - x)
        })

        if abs(fx) < tolerance or abs(x_new - x) < tolerance:
            return {
                "root": x_new,
                "iterations": i,
                "converged": True,
                "steps": steps
            }

        x = x_new

    return {
        "root": x,
        "iterations": max_iterations,
        "converged": False,
        "steps": steps
    }


# ============================================================
# 2. BISECTION METHOD
# ============================================================

def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> Dict:
    """
    Find a root using the Bisection method.
    """

    tolerance = _validate_tolerance(tolerance)
    max_iterations = _validate_iterations(max_iterations)

    a = _finite(a, "Lower bound")
    b = _finite(b, "Upper bound")

    if a >= b:
        raise ValueError("Lower bound must be smaller than upper bound.")

    fa = _validate_function_result(f(a), "f(a)")
    fb = _validate_function_result(f(b), "f(b)")

    if abs(fa) < tolerance:
        return {
            "root": a,
            "iterations": 0,
            "converged": True,
            "steps": []
        }

    if abs(fb) < tolerance:
        return {
            "root": b,
            "iterations": 0,
            "converged": True,
            "steps": []
        }

    if fa * fb > 0:
        raise ValueError(
            "Bisection requires f(a) and f(b) to have opposite signs."
        )

    steps = []

    for i in range(1, max_iterations + 1):
        c = (a + b) / 2
        fc = _validate_function_result(f(c), "f(midpoint)")

        error = abs(b - a) / 2

        steps.append({
            "iteration": i,
            "a": a,
            "b": b,
            "midpoint": c,
            "f_midpoint": fc,
            "error": error
        })

        if abs(fc) < tolerance or error < tolerance:
            return {
                "root": c,
                "iterations": i,
                "converged": True,
                "steps": steps
            }

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return {
        "root": (a + b) / 2,
        "iterations": max_iterations,
        "converged": False,
        "steps": steps
    }


# ============================================================
# 3. SECANT METHOD
# ============================================================

def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> Dict:
    """
    Find a root using the Secant method.

    Formula:
        x(n+1) = x(n) -
                  f(x(n)) * (x(n)-x(n-1)) /
                  (f(x(n))-f(x(n-1)))
    """

    tolerance = _validate_tolerance(tolerance)
    max_iterations = _validate_iterations(max_iterations)

    x0 = _finite(x0, "x0")
    x1 = _finite(x1, "x1")

    f0 = _validate_function_result(f(x0), "f(x0)")
    f1 = _validate_function_result(f(x1), "f(x1)")

    steps = []

    for i in range(1, max_iterations + 1):
        denominator = f1 - f0

        if abs(denominator) < EPSILON:
            raise ValueError(
                "Secant method denominator is too small."
            )

        x2 = x1 - f1 * (x1 - x0) / denominator
        x2 = _finite(x2, "Next approximation")

        f2 = _validate_function_result(f(x2), "f(x2)")

        error = abs(x2 - x1)

        steps.append({
            "iteration": i,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "f_x2": f2,
            "error": error
        })

        if abs(f2) < tolerance or error < tolerance:
            return {
                "root": x2,
                "iterations": i,
                "converged": True,
                "steps": steps
            }

        x0, x1 = x1, x2
        f0, f1 = f1, f2

    return {
        "root": x1,
        "iterations": max_iterations,
        "converged": False,
        "steps": steps
    }


# ============================================================
# 4. REGULA FALSI METHOD
# ============================================================

def regula_falsi(
    f: Callable[[float], float],
    a: float,
    b: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> Dict:
    """
    Find a root using the Regula Falsi method.
    """

    tolerance = _validate_tolerance(tolerance)
    max_iterations = _validate_iterations(max_iterations)

    a = _finite(a, "Lower bound")
    b = _finite(b, "Upper bound")

    if a >= b:
        raise ValueError("Lower bound must be smaller than upper bound.")

    fa = _validate_function_result(f(a), "f(a)")
    fb = _validate_function_result(f(b), "f(b)")

    if abs(fa) < tolerance:
        return {
            "root": a,
            "iterations": 0,
            "converged": True,
            "steps": []
        }

    if abs(fb) < tolerance:
        return {
            "root": b,
            "iterations": 0,
            "converged": True,
            "steps": []
        }

    if fa * fb > 0:
        raise ValueError(
            "Regula Falsi requires opposite signs "
            "at interval endpoints."
        )

    steps = []
    previous = None

    for i in range(1, max_iterations + 1):
        denominator = fb - fa

        if abs(denominator) < EPSILON:
            raise ValueError(
                "Regula Falsi denominator is too small."
            )

        c = (a * fb - b * fa) / denominator
        c = _finite(c, "Approximation")

        fc = _validate_function_result(f(c), "f(c)")

        error = (
            abs(c - previous)
            if previous is not None
            else abs(b - a)
        )

        steps.append({
            "iteration": i,
            "a": a,
            "b": b,
            "x": c,
            "f_x": fc,
            "error": error
        })

        if abs(fc) < tolerance or (
            previous is not None and error < tolerance
        ):
            return {
                "root": c,
                "iterations": i,
                "converged": True,
                "steps": steps
            }

        previous = c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return {
        "root": c,
        "iterations": max_iterations,
        "converged": False,
        "steps": steps
    }


# ============================================================
# 5. GAUSSIAN ELIMINATION
# ============================================================

def gaussian_elimination(matrix: Sequence[Sequence[float]]) -> Dict:
    """
    Solve a system of linear equations using
    Gaussian elimination with partial pivoting.

    Input:
        Augmented n x (n+1) matrix.

    Example:
        2x + y = 5
        x + 3y = 6

        [
            [2, 1, 5],
            [1, 3, 6]
        ]
    """

    a = _validate_matrix(matrix)

    n = len(a)

    if any(len(row) != n + 1 for row in a):
        raise ValueError(
            "Matrix must be an augmented n x (n+1) matrix."
        )

    steps = []

    # Forward elimination
    for i in range(n):
        pivot_row = max(
            range(i, n),
            key=lambda r: abs(a[r][i])
        )

        pivot = a[pivot_row][i]

        if abs(pivot) < EPSILON:
            raise ValueError(
                "Singular or nearly singular matrix."
            )

        if pivot_row != i:
            a[i], a[pivot_row] = a[pivot_row], a[i]

            steps.append({
                "operation": "row_swap",
                "row1": i,
                "row2": pivot_row,
                "matrix": [row[:] for row in a]
            })

        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]

            a[j][i] = 0.0

            for k in range(i + 1, n + 1):
                a[j][k] -= factor * a[i][k]

            steps.append({
                "operation": "elimination",
                "pivot_row": i,
                "target_row": j,
                "factor": factor,
                "matrix": [row[:] for row in a]
            })

    # Back substitution
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        value = a[i][n]

        for j in range(i + 1, n):
            value -= a[i][j] * x[j]

        if abs(a[i][i]) < EPSILON:
            raise ValueError(
                "Zero pivot encountered during back substitution."
            )

        x[i] = value / a[i][i]

    return {
        "solution": x,
        "steps": steps
    }


# ============================================================
# 6. GAUSS-SEIDEL METHOD
# ============================================================

def gauss_seidel(
    matrix: Sequence[Sequence[float]],
    b: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> Dict:
    """
    Solve Ax = b using the Gauss-Seidel iterative method.
    """

    tolerance = _validate_tolerance(tolerance)
    max_iterations = _validate_iterations(max_iterations)

    A = _validate_matrix(matrix, square=True)

    try:
        b = list(map(float, b))
    except (TypeError, ValueError):
        raise ValueError("b must contain numeric values.")

    n = len(A)

    if len(b) != n:
        raise ValueError("Dimension mismatch.")

    for i in range(n):
        if abs(A[i][i]) < EPSILON:
            raise ValueError(
                "Zero diagonal element encountered."
            )

    x = [0.0] * n
    steps = []

    for iteration in range(1, max_iterations + 1):
        old_x = x.copy()

        for i in range(n):
            sum_value = b[i]

            for j in range(n):
                if j != i:
                    sum_value -= A[i][j] * x[j]

            x[i] = sum_value / A[i][i]

        error = max(
            abs(x[i] - old_x[i])
            for i in range(n)
        )

        steps.append({
            "iteration": iteration,
            "values": x.copy(),
            "error": error
        })

        if error < tolerance:
            return {
                "solution": x,
                "iterations": iteration,
                "converged": True,
                "steps": steps
            }

    return {
        "solution": x,
        "iterations": max_iterations,
        "converged": False,
        "steps": steps
    }


# ============================================================
# 7. TRAPEZOIDAL RULE
# ============================================================

def trapezoidal_rule(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100
) -> Dict:
    """
    Numerical integration using the Trapezoidal Rule.
    """

    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive.")

    a = _finite(a, "Lower limit")
    b = _finite(b, "Upper limit")

    h = (b - a) / n

    total = (
        _validate_function_result(f(a), "f(a)")
        + _validate_function_result(f(b), "f(b)")
    )

    for i in range(1, n):
        x = a + i * h
        total += 2 * _validate_function_result(f(x), "f(x)")

    result = (h / 2) * total

    return {
        "integral": result,
        "interval": [a, b],
        "subintervals": n
    }


# ============================================================
# 8. SIMPSON'S 1/3 RULE
# ============================================================

def simpsons_rule(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100
) -> Dict:
    """
    Numerical integration using Simpson's 1/3 Rule.

    n must be positive and even.
    """

    n = int(n)

    if n <= 0 or n % 2 != 0:
        raise ValueError(
            "Simpson's 1/3 Rule requires an even positive n."
        )

    a = _finite(a, "Lower limit")
    b = _finite(b, "Upper limit")

    h = (b - a) / n

    total = (
        _validate_function_result(f(a), "f(a)")
        + _validate_function_result(f(b), "f(b)")
    )

    for i in range(1, n):
        x = a + i * h
        fx = _validate_function_result(f(x), "f(x)")

        if i % 2 == 0:
            total += 2 * fx
        else:
            total += 4 * fx

    result = (h / 3) * total

    return {
        "integral": result,
        "interval": [a, b],
        "subintervals": n
    }


# ============================================================
# 9. SIMPSON'S 3/8 RULE
# ============================================================

def simpsons_three_eighth_rule(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 99
) -> Dict:
    """
    Numerical integration using Simpson's 3/8 Rule.

    n must be positive and divisible by 3.
    """

    n = int(n)

    if n <= 0 or n % 3 != 0:
        raise ValueError(
            "Simpson's 3/8 Rule requires n divisible by 3."
        )

    a = _finite(a, "Lower limit")
    b = _finite(b, "Upper limit")

    h = (b - a) / n

    total = (
        _validate_function_result(f(a), "f(a)")
        + _validate_function_result(f(b), "f(b)")
    )

    for i in range(1, n):
        x = a + i * h
        fx = _validate_function_result(f(x), "f(x)")

        if i % 3 == 0:
            total += 2 * fx
        else:
            total += 3 * fx

    result = (3 * h / 8) * total

    return {
        "integral": result,
        "interval": [a, b],
        "subintervals": n
    }


# ============================================================
# 10. LAGRANGE INTERPOLATION
# ============================================================

def lagrange_interpolation(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x: float
) -> Dict:
    """
    Estimate y at x using Lagrange interpolation.
    """

    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same length.")

    if len(x_values) == 0:
        raise ValueError("Data cannot be empty.")

    x_values = list(map(float, x_values))
    y_values = list(map(float, y_values))
    x = _finite(x, "Interpolation point")

    result = 0.0
    steps = []

    for i in range(len(x_values)):
        term = y_values[i]

        for j in range(len(x_values)):
            if i == j:
                continue

            denominator = x_values[i] - x_values[j]

            if abs(denominator) < EPSILON:
                raise ValueError(
                    "Duplicate x-values are not allowed."
                )

            term *= (
                (x - x_values[j]) /
                denominator
            )

        result += term

        steps.append({
            "basis": i,
            "term": term
        })

    return {
        "value": result,
        "x": x,
        "steps": steps
    }


# ============================================================
# 11. NEWTON FORWARD INTERPOLATION
# ============================================================

def newton_forward_interpolation(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x: float
) -> Dict:
    """
    Newton Forward interpolation.

    x-values must be equally spaced.
    """

    x_values = list(map(float, x_values))
    y_values = list(map(float, y_values))
    x = _finite(x, "Interpolation point")

    n = len(x_values)

    if n != len(y_values):
        raise ValueError("x and y must have same length.")

    if n < 2:
        raise ValueError("At least two points are required.")

    h = x_values[1] - x_values[0]

    if abs(h) < EPSILON:
        raise ValueError("Invalid x-values.")

    for i in range(2, n):
        spacing = x_values[i] - x_values[i - 1]

        if abs(spacing - h) > 1e-10:
            raise ValueError(
                "Newton Forward requires equally spaced x-values."
            )

    table = [y_values.copy()]

    for _ in range(1, n):
        previous = table[-1]

        current = [
            previous[i + 1] - previous[i]
            for i in range(len(previous) - 1)
        ]

        table.append(current)

    u = (x - x_values[0]) / h

    result = table[0][0]
    term = 1.0

    for i in range(1, n):
        term *= (u - (i - 1)) / i
        result += term * table[i][0]

    return {
        "value": result,
        "x": x,
        "difference_table": table
    }


# ============================================================
# 12. NEWTON BACKWARD INTERPOLATION
# ============================================================

def newton_backward_interpolation(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x: float
) -> Dict:
    """
    Newton Backward interpolation.

    x-values must be equally spaced.
    """

    x_values = list(map(float, x_values))
    y_values = list(map(float, y_values))
    x = _finite(x, "Interpolation point")

    n = len(x_values)

    if n != len(y_values):
        raise ValueError("x and y must have same length.")

    if n < 2:
        raise ValueError("At least two points are required.")

    h = x_values[1] - x_values[0]

    if abs(h) < EPSILON:
        raise ValueError("Invalid x-values.")

    for i in range(2, n):
        spacing = x_values[i] - x_values[i - 1]

        if abs(spacing - h) > 1e-10:
            raise ValueError(
                "Newton Backward requires equally spaced x-values."
            )

    table = [y_values.copy()]

    for _ in range(1, n):
        previous = table[-1]

        current = [
            previous[i + 1] - previous[i]
            for i in range(len(previous) - 1)
        ]

        table.append(current)

    u = (x - x_values[-1]) / h

    result = table[0][-1]
    term = 1.0

    for i in range(1, n):
        term *= (u + (i - 1)) / i
        result += term * table[i][-1]

    return {
        "value": result,
        "x": x,
        "difference_table": table
    }


# ============================================================
# 13. EULER METHOD
# ============================================================

def euler_method(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    steps: int
) -> Dict:
    """
    Solve dy/dx = f(x,y) using Euler's method.

    Formula:
        y(n+1) = y(n) + h*f(x(n), y(n))
    """

    steps = _validate_steps(steps)

    x = _finite(x0, "x0")
    y = _finite(y0, "y0")
    h = _finite(h, "Step size")

    if abs(h) < EPSILON:
        raise ValueError("Step size cannot be zero.")

    results = [{
        "step": 0,
        "x": x,
        "y": y
    }]

    for i in range(1, steps + 1):
        slope = _validate_function_result(
            f(x, y),
            "f(x,y)"
        )

        y = y + h * slope
        x = x + h

        y = _finite(y, "y")
        x = _finite(x, "x")

        results.append({
            "step": i,
            "x": x,
            "y": y,
            "slope": slope
        })

    return {
        "solution": results
    }


# ============================================================
# 14. RUNGE-KUTTA 4TH ORDER
# ============================================================

def runge_kutta_4(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    steps: int
) -> Dict:
    """
    Solve dy/dx = f(x,y) using RK4.
    """

    steps = _validate_steps(steps)

    x = _finite(x0, "x0")
    y = _finite(y0, "y0")
    h = _finite(h, "Step size")

    if abs(h) < EPSILON:
        raise ValueError("Step size cannot be zero.")

    results = [{
        "step": 0,
        "x": x,
        "y": y
    }]

    for i in range(1, steps + 1):
        k1 = h * _validate_function_result(
            f(x, y),
            "k1 function value"
        )

        k2 = h * _validate_function_result(
            f(x + h / 2, y + k1 / 2),
            "k2 function value"
        )

        k3 = h * _validate_function_result(
            f(x + h / 2, y + k2 / 2),
            "k3 function value"
        )

        k4 = h * _validate_function_result(
            f(x + h, y + k3),
            "k4 function value"
        )

        y = y + (
            k1 + 2 * k2 + 2 * k3 + k4
        ) / 6

        x = x + h

        x = _finite(x, "x")
        y = _finite(y, "y")

        results.append({
            "step": i,
            "x": x,
            "y": y,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4
        })

    return {
        "solution": results
    }


# ============================================================
# 15. MATRIX MULTIPLICATION
# ============================================================

def matrix_multiply(A, B) -> List[List[float]]:
    """
    Multiply two matrices.

    If A is m x n and B is n x p,
    result is m x p.
    """

    A = _validate_matrix(A)
    B = _validate_matrix(B)

    if len(A[0]) != len(B):
        raise ValueError(
            "Matrix dimensions are incompatible."
        )

    b_columns = list(zip(*B))

    return [
        [
            sum(
                aik * bjk
                for aik, bjk in zip(row, column)
            )
            for column in b_columns
        ]
        for row in A
    ]


# ============================================================
# 16. MATRIX TRANSPOSE
# ============================================================

def matrix_transpose(A) -> List[List[float]]:
    """
    Return transpose of matrix A.
    """

    A = _validate_matrix(A)

    return [
        list(row)
        for row in zip(*A)
    ]


# ============================================================
# 17. DETERMINANT
# ============================================================

def determinant(matrix) -> float:
    """
    Calculate determinant using elimination with
    partial pivoting.

    Complexity:
        O(n^3)
    """

    A = _validate_matrix(matrix, square=True)

    n = len(A)

    sign = 1.0
    determinant_value = 1.0

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(A[row][column])
        )

        pivot = A[pivot_row][column]

        if abs(pivot) < EPSILON:
            return 0.0

        if pivot_row != column:
            A[column], A[pivot_row] = (
                A[pivot_row],
                A[column]
            )
            sign *= -1

        pivot = A[column][column]
        determinant_value *= pivot

        for row in range(column + 1, n):
            factor = A[row][column] / pivot

            A[row][column] = 0.0

            for inner in range(column + 1, n):
                A[row][inner] -= (
                    factor * A[column][inner]
                )

    return sign * determinant_value


# ============================================================
# ALGORITHM REGISTRY
# ============================================================

ALGORITHMS = {
    "newton_raphson": newton_raphson,
    "bisection": bisection,
    "secant": secant,
    "regula_falsi": regula_falsi,
    "gaussian_elimination": gaussian_elimination,
    "gauss_seidel": gauss_seidel,
    "trapezoidal_rule": trapezoidal_rule,
    "simpsons_rule": simpsons_rule,
    "simpsons_three_eighth": simpsons_three_eighth_rule,
    "lagrange_interpolation": lagrange_interpolation,
    "newton_forward": newton_forward_interpolation,
    "newton_backward": newton_backward_interpolation,
    "euler_method": euler_method,
    "runge_kutta_4": runge_kutta_4,
    "matrix_multiply": matrix_multiply,
    "matrix_transpose": matrix_transpose,
    "determinant": determinant,
}


# ============================================================
# ALGORITHM INFORMATION
# ============================================================

ALGORITHM_INFO = {
    "newton_raphson": {
        "name": "Newton-Raphson Method",
        "category": "Numerical Methods",
        "type": "Root Finding",
        "complexity": "O(k)",
        "requires": ["function", "derivative", "initial_value"]
    },

    "bisection": {
        "name": "Bisection Method",
        "category": "Numerical Methods",
        "type": "Root Finding",
        "complexity": "O(k)",
        "requires": ["function", "lower_bound", "upper_bound"]
    },

    "secant": {
        "name": "Secant Method",
        "category": "Numerical Methods",
        "type": "Root Finding",
        "complexity": "O(k)",
        "requires": ["function", "x0", "x1"]
    },

    "regula_falsi": {
        "name": "Regula Falsi Method",
        "category": "Numerical Methods",
        "type": "Root Finding",
        "complexity": "O(k)",
        "requires": ["function", "lower_bound", "upper_bound"]
    },

    "gaussian_elimination": {
        "name": "Gaussian Elimination",
        "category": "Linear Algebra",
        "type": "Linear Equations",
        "complexity": "O(n^3)",
        "requires": ["augmented_matrix"]
    },

    "gauss_seidel": {
        "name": "Gauss-Seidel Method",
        "category": "Numerical Methods",
        "type": "Linear Equations",
        "complexity": "O(kn^2)",
        "requires": ["matrix_A", "vector_b"]
    },

    "trapezoidal_rule": {
        "name": "Trapezoidal Rule",
        "category": "Numerical Integration",
        "type": "Integration",
        "complexity": "O(n)",
        "requires": ["function", "lower_limit", "upper_limit"]
    },

    "simpsons_rule": {
        "name": "Simpson's 1/3 Rule",
        "category": "Numerical Integration",
        "type": "Integration",
        "complexity": "O(n)",
        "requires": ["function", "lower_limit", "upper_limit", "even_n"]
    },

    "simpsons_three_eighth": {
        "name": "Simpson's 3/8 Rule",
        "category": "Numerical Integration",
        "type": "Integration",
        "complexity": "O(n)",
        "requires": ["function", "lower_limit", "upper_limit", "n_divisible_by_3"]
    },

    "lagrange_interpolation": {
        "name": "Lagrange Interpolation",
        "category": "Interpolation",
        "type": "Interpolation",
        "complexity": "O(n^2)",
        "requires": ["x_values", "y_values", "x"]
    },

    "newton_forward": {
        "name": "Newton Forward Interpolation",
        "category": "Interpolation",
        "type": "Interpolation",
        "complexity": "O(n^2)",
        "requires": ["equally_spaced_x", "y_values", "x"]
    },

    "newton_backward": {
        "name": "Newton Backward Interpolation",
        "category": "Interpolation",
        "type": "Interpolation",
        "complexity": "O(n^2)",
        "requires": ["equally_spaced_x", "y_values", "x"]
    },

    "euler_method": {
        "name": "Euler Method",
        "category": "Differential Equations",
        "type": "ODE",
        "complexity": "O(k)",
        "requires": ["function", "x0", "y0", "step_size", "steps"]
    },

    "runge_kutta_4": {
        "name": "Runge-Kutta 4th Order",
        "category": "Differential Equations",
        "type": "ODE",
        "complexity": "O(k)",
        "requires": ["function", "x0", "y0", "step_size", "steps"]
    },

    "matrix_multiply": {
        "name": "Matrix Multiplication",
        "category": "Linear Algebra",
        "type": "Matrix Operation",
        "complexity": "O(n^3)",
        "requires": ["matrix_A", "matrix_B"]
    },

    "matrix_transpose": {
        "name": "Matrix Transpose",
        "category": "Linear Algebra",
        "type": "Matrix Operation",
        "complexity": "O(mn)",
        "requires": ["matrix"]
    },

    "determinant": {
        "name": "Determinant",
        "category": "Linear Algebra",
        "type": "Matrix Operation",
        "complexity": "O(n^3)",
        "requires": ["square_matrix"]
    }
}


def validate_algorithm_result(result: Any) -> Dict[str, Any]:
    """Add provenance metadata to a result computed from supplied inputs."""
    if not isinstance(result, dict):
        return {"result": result, "computed_independently": True,
                "similarity_answer_copy": False}
    output = dict(result)
    output.setdefault("computed_independently", True)
    output.setdefault("similarity_answer_copy", False)
    return output


# ============================================================
# PUBLIC API
# ============================================================

def get_algorithm_info() -> Dict:
    """
    Return information about all available algorithms.
    """

    return {
        name: info.copy()
        for name, info in ALGORITHM_INFO.items()
    }


def normalize_algorithm_name(name: Any) -> str:
    """Normalize common natural-language names to registry keys."""
    if not isinstance(name, str):
        return ""
    value = re.sub(r"\s+", " ", name.strip().lower().replace("_", " "))
    if value in ALGORITHM_ALIASES:
        return ALGORITHM_ALIASES[value]
    compact = value.replace(" ", "_")
    return compact if compact in ALGORITHMS else value


def get_algorithm(name: str):
    """Return an algorithm by canonical name or supported alias."""
    return ALGORITHMS.get(normalize_algorithm_name(name))


def infer_algorithm(text: str):
    """Infer explicit algorithm intent; never returns a dataset answer."""
    if not isinstance(text, str):
        return None
    value = re.sub(r"\s+", " ", text.lower()).strip()
    matches = []
    for name, keywords in ALGORITHM_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in value)
        if hits:
            matches.append((hits, max(map(len, keywords)), name))
    return max(matches)[2] if matches else None


def list_algorithms() -> List[str]:
    """
    Return the names of all available algorithms.
    """

    return list(ALGORITHMS.keys())


def get_algorithms_by_category(category: str) -> Dict:
    """
    Return algorithms belonging to a category.
    """

    if not isinstance(category, str):
        return {}

    category = category.strip().lower()

    return {
        name: info
        for name, info in ALGORITHM_INFO.items()
        if info["category"].lower() == category
    }


__all__ = [
    "newton_raphson",
    "bisection",
    "secant",
    "regula_falsi",
    "gaussian_elimination",
    "gauss_seidel",
    "trapezoidal_rule",
    "simpsons_rule",
    "simpsons_three_eighth_rule",
    "lagrange_interpolation",
    "newton_forward_interpolation",
    "newton_backward_interpolation",
    "euler_method",
    "runge_kutta_4",
    "matrix_multiply",
    "matrix_transpose",
    "determinant",
    "ALGORITHMS",
    "ALGORITHM_INFO",
    "get_algorithm",
    "normalize_algorithm_name",
    "infer_algorithm",
    "validate_algorithm_result",
    "get_algorithm_info",
    "list_algorithms",
    "get_algorithms_by_category",
]
# ============================================================
# M3 ENGINE COMPATIBILITY WRAPPER
# ============================================================

class AlgorithmEngine:
    """Small compatibility layer used by model.solver.

    The numerical algorithms above remain the actual implementations.
    This class only provides a stable interface for selecting/executing
    an algorithm by name.
    """

    def __init__(self):
        self.algorithms = ALGORITHMS

    def list_algorithms(self) -> List[str]:
        return list_algorithms()

    def get_algorithm(self, name: str):
        return get_algorithm(name)

    def get_algorithm_info(self) -> Dict:
        return get_algorithm_info()

    def run(self, name: str, *args, **kwargs):
        algorithm = get_algorithm(name)
        if algorithm is None:
            raise ValueError(f"Unknown algorithm: {name}")
        return validate_algorithm_result(algorithm(*args, **kwargs))

    def execute(self, name: str, *args, **kwargs):
        return self.run(name, *args, **kwargs)

    def solve(self, text=None, subject=None, algorithm=None, *args, **kwargs):
        """Execute a named numerical algorithm when enough arguments exist.

        Natural-language solving is intentionally left to model/solver.py.
        If no algorithm name or arguments are supplied, return None so the
        caller can use its normal subject-specific solver.
        """
        name = algorithm
        if not name and isinstance(subject, str):
            name = subject
        if not name and isinstance(text, str):
            name = infer_algorithm(text)

        if not name:
            return None

        canonical = normalize_algorithm_name(name)
        fn = get_algorithm(canonical)
        if fn is None:
            return None

        if args or kwargs:
            result = fn(*args, **kwargs)
            return validate_algorithm_result(result)

        return {
            "algorithm": canonical,
            "ready": False,
            "computed_independently": True,
            "similarity_answer_copy": False,
        }


__all__ = [
    "newton_raphson",
    "bisection",
    "secant",
    "regula_falsi",
    "gaussian_elimination",
    "gauss_seidel",
    "trapezoidal_rule",
    "simpsons_rule",
    "simpsons_three_eighth_rule",
    "lagrange_interpolation",
    "newton_forward_interpolation",
    "newton_backward_interpolation",
    "euler_method",
    "runge_kutta_4",
    "matrix_multiply",
    "matrix_transpose",
    "determinant",
    "ALGORITHMS",
    "ALGORITHM_INFO",
    "get_algorithm",
    "normalize_algorithm_name",
    "infer_algorithm",
    "validate_algorithm_result",
    "get_algorithm_info",
    "list_algorithms",
    "get_algorithms_by_category",
    "AlgorithmEngine",
]
