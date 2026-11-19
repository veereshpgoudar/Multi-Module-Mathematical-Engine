import ast
import math
import re
import logging
import difflib
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import config


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM IMPORTS
# ════════════════════════════════════════════════════════════════════════════

try:
    # solver.py is imported as model.solver.  The earlier fallback chain
    # referenced a misspelled "alorithm" module, so numerical methods were
    # silently unavailable even though model/algorithms.py exists.
    from .algorithms import (
        newton_raphson,
        bisection,
        secant,
        regula_falsi,
        gaussian_elimination,
        gauss_seidel,
        trapezoidal_rule,
        simpsons_rule,
        simpsons_three_eighth_rule,
        lagrange_interpolation,
        newton_forward_interpolation,
        newton_backward_interpolation,
        euler_method,
        runge_kutta_4,
        matrix_multiply,
        matrix_transpose,
        determinant,
        get_algorithm_info,
        get_algorithm,
    )
    ALGORITHMS_AVAILABLE = True
except ImportError:
    ALGORITHMS_AVAILABLE = False


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

MAX_EXPRESSION_LENGTH = 500
MAX_POWER = 1000
MAX_FACTORIAL = 1000


# ════════════════════════════════════════════════════════════════════════════
# SAFE ARITHMETIC ENGINE
# ════════════════════════════════════════════════════════════════════════════

_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,

    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,

    ast.USub,
    ast.UAdd,
)


_SAFE_FUNCS: Dict[str, Any] = {
    "sqrt": math.sqrt,

    "abs": abs,
    "round": round,

    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,

    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,

    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,

    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,

    "exp": math.exp,

    "ceil": math.ceil,
    "floor": math.floor,

    "factorial": math.factorial,
    "gcd": math.gcd,

    "pi": math.pi,
    "e": math.e,

    "degrees": math.degrees,
    "radians": math.radians,

    "pow": pow,

    "max": max,
    "min": min,
    "sum": sum,
}


def _check_safe(node: ast.AST) -> None:
    """
    Recursively verify that an AST contains only approved operations.
    """

    if not isinstance(node, _SAFE_NODES):
        raise ValueError(
            f"Unsafe node type: {type(node).__name__}"
        )

    for child in ast.iter_child_nodes(node):
        _check_safe(child)


def _check_math_limits(tree: ast.AST) -> None:
    """
    Prevent unnecessarily expensive mathematical expressions.
    """

    for node in ast.walk(tree):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, int):
                if abs(node.value) > 10 ** 12:
                    raise ValueError("Number is too large")

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):

            if isinstance(node.right, ast.Constant):
                exponent = node.right.value

                if isinstance(exponent, (int, float)):
                    if abs(exponent) > MAX_POWER:
                        raise ValueError(
                            "Exponent is too large"
                        )

        if isinstance(node, ast.Call):

            if isinstance(node.func, ast.Name):

                if node.func.id == "factorial":

                    if node.args:
                        arg = node.args[0]

                        if isinstance(arg, ast.Constant):
                            if isinstance(arg.value, int):
                                if arg.value < 0:
                                    raise ValueError(
                                        "Factorial requires a non-negative integer"
                                    )

                                if arg.value > MAX_FACTORIAL:
                                    raise ValueError(
                                        "Factorial value is too large"
                                    )


def _normalise_expr(expr: str) -> str:
    """
    Normalize common mathematical symbols.
    """

    expr = str(expr).strip()

    if not expr:
        return ""

    if len(expr) > MAX_EXPRESSION_LENGTH:
        return ""

    # Unicode multiplication
    expr = expr.replace("×", "*")

    # Unicode division
    expr = expr.replace("÷", "/")

    # Unicode minus variants
    expr = expr.replace("−", "-")
    expr = expr.replace("–", "-")
    expr = expr.replace("—", "-")

    # Power
    expr = re.sub(r"\^", "**", expr)

    # Unicode superscript digits, e.g. 15² -> 15**2.
    superscripts = str.maketrans({
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    })
    expr = re.sub(
        r"(?<=\d)[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda m: "**" + m.group(0).translate(superscripts),
        expr,
    )

    # Square root: √(144)
    expr = re.sub(
        r"√\s*\(([^()]+)\)",
        r"sqrt(\1)",
        expr
    )

    # Square root: √144
    expr = re.sub(
        r"√\s*(-?(?:\d+(?:\.\d*)?|\.\d+))",
        r"sqrt(\1)",
        expr
    )

    # Remove unnecessary whitespace
    expr = re.sub(r"\s+", " ", expr).strip()

    return expr


def evaluate_expression(expr: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression.

    Only explicitly allowed AST operations and functions are accepted.
    """

    expr = _normalise_expr(expr)

    if not expr:
        return None

    try:

        tree = ast.parse(
            expr,
            mode="eval"
        )

        _check_safe(tree.body)
        _check_math_limits(tree.body)

        # Reject unknown variables/functions
        for node in ast.walk(tree):

            if isinstance(node, ast.Name):

                if node.id not in _SAFE_FUNCS:
                    return None

        result = eval(
            compile(
                tree,
                "<safe-math-expression>",
                "eval"
            ),
            {
                "__builtins__": {}
            },
            _SAFE_FUNCS,
        )

        if isinstance(result, bool):
            return float(result)

        if not isinstance(result, (int, float)):
            return None

        if not math.isfinite(float(result)):
            return None

        return float(result)

    except Exception as exc:

        logger.debug(
            "Arithmetic evaluation failed: %s",
            exc
        )

        return None


# ════════════════════════════════════════════════════════════════════════════
# NUMBER FORMATTING
# ════════════════════════════════════════════════════════════════════════════

def _format_number(n: float) -> str:
    """
    Convert a numeric result into a clean display string.
    """

    try:

        n = float(n)

        if not math.isfinite(n):
            return str(n)

        if n.is_integer() and abs(n) < 10 ** 12:
            return str(int(n))

        return f"{n:.8g}"

    except Exception:
        return str(n)


# ════════════════════════════════════════════════════════════════════════════
# SOLVER-LEVEL NLP NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════

_SOLVER_TYPO_MAP = {
    "deminant": "determinant", "determinan": "determinant", "determininan": "determinant",
    "determininant": "determinant", "determinent": "determinant", "determinat": "determinant",
    "determinatn": "determinant", "determinnt": "determinant", "detminant": "determinant",
    "determinantt": "determinant", "determinent": "determinant",
    "transpos": "transpose", "tranpose": "transpose", "transpoes": "transpose",
    "invers": "inverse", "inverce": "inverse", "inversse": "inverse",
    "adjointt": "adjoint", "adjugent": "adjugate", "adjugte": "adjugate",
    "multiplcation": "multiplication", "multiplicaton": "multiplication", "multiplay": "multiply",
    "substract": "subtract", "substraction": "subtraction", "subtracion": "subtraction",
    "addtion": "addition", "adition": "addition", "aditionn": "addition",
    "symetric": "symmetric", "symmtric": "symmetric", "skewsymetric": "skew symmetric",
    "eigenvaluee": "eigenvalue", "eigenvlue": "eigenvalue", "eigenvectorr": "eigenvector",
    "gausian": "gaussian", "gaussianelimination": "gaussian elimination",
    "bisectionn": "bisection", "seccant": "secant", "eullar": "euler",
    "lagrangee": "lagrange", "interpolaton": "interpolation", "interpolationn": "interpolation",
}

_SOLVER_PHRASES = {
    "det of": "determinant", "det(a)": "determinant", "det a": "determinant",
    "find det": "find determinant", "calculate det": "calculate determinant",
    "find the det": "find the determinant", "value of det": "value of determinant",
    "transpose of": "transpose", "transposition of": "transpose",
    "inverse of": "inverse", "inv of": "inverse", "adjoint of": "adjoint",
    "adjugate of": "adjugate", "rank of": "rank", "trace of": "trace",
    "eigen values": "eigenvalues", "eigen vectors": "eigenvectors",
    "matrix product": "matrix multiplication", "product of matrices": "matrix multiplication",
    "multiply matrices": "matrix multiplication", "sum of matrices": "matrix addition",
    "difference of matrices": "matrix subtraction", "add matrices": "matrix addition",
    "subtract matrices": "matrix subtraction", "find x": "find x",
    "what is": "calculate", "what's": "calculate", "pls": "please", "plz": "please",
    "can u": "can you", "could u": "could you",
}

_SOLVER_VOCAB = [
    "determinant", "transpose", "inverse", "adjoint", "adjugate", "rank", "trace",
    "minor", "cofactor", "symmetric", "skew", "eigenvalue", "eigenvector",
    "addition", "subtraction", "multiplication", "division", "matrix", "matrices",
    "newton", "bisection", "secant", "regula", "gaussian", "gauss", "seidel",
    "trapezoidal", "simpson", "lagrange", "interpolation", "euler", "runge", "kutta",
]


def normalize_solver_text(text: str) -> str:
    """Normalize symbols, phrases and common OCR/user typos without touching data literals."""
    if not text:
        return ""
    value = str(text)
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("×", " * ").replace("÷", " / ").replace("·", " * ")
    value = value.replace("√", "sqrt")
    value = re.sub(r"\s+", " ", value).strip().lower()

    # Correct whole words before phrase expansion. This prevents a phrase such
    # as "find det" from corrupting a misspelling such as "determininan".
    def correct_word(match):
        token = match.group(0)
        fixed = _SOLVER_TYPO_MAP.get(token, token)
        if fixed not in _SOLVER_VOCAB and len(fixed) >= 5:
            close = difflib.get_close_matches(fixed, _SOLVER_VOCAB, n=1, cutoff=0.88)
            if close and abs(len(close[0]) - len(fixed)) <= 2:
                fixed = close[0]
        return fixed

    value = re.sub(r"[a-z]+", correct_word, value)

    # Expand complete phrases only after typo correction.
    for src, dst in sorted(_SOLVER_PHRASES.items(), key=lambda x: -len(x[0])):
        if re.fullmatch(r"[a-z ]+", src):
            value = re.sub(r"(?<![a-z])" + re.escape(src) + r"(?![a-z])", dst, value)
        else:
            value = value.replace(src, dst)

    return re.sub(r"\s+", " ", value).strip()


def detect_operation_nlp(text: str) -> Optional[str]:
    """Return a high-confidence matrix operation from natural language."""
    low = normalize_solver_text(text).lower()
    checks = [
        ("eigenvectors", ("eigenvector", "eigen vector")),
        ("eigenvalues", ("eigenvalue", "eigen value")),
        ("determinant", ("determinant", "det")),
        ("inverse", ("inverse", "inv(")),
        ("adjoint", ("adjoint", "adjugate")),
        ("transpose", ("transpose", "transposition")),
        ("cofactor", ("cofactor",)),
        ("minor", ("minor",)),
        ("rank", ("rank",)),
        ("trace", ("trace",)),
        ("skew_symmetric", ("skew symmetric", "skew-symmetric")),
        ("symmetric", ("symmetric",)),
        ("scalar_multiplication", ("scalar multiplication", "multiply matrix by scalar")),
        ("multiplication", ("matrix multiplication", "multiply matrices", "matrix product", "product of matrices")),
        ("subtraction", ("matrix subtraction", "subtract matrices", "difference of matrices")),
        ("addition", ("matrix addition", "add matrices", "sum of matrices")),
    ]
    for operation, patterns in checks:
        if any(p in low for p in patterns):
            return operation
    if re.search(r"\bA\s*\+\s*B\b", low, re.I): return "addition"
    if re.search(r"\bA\s*-\s*B\b", low, re.I): return "subtraction"
    if re.search(r"\bA\s*(?:\*|x)\s*B\b", low, re.I): return "multiplication"
    return None


# ════════════════════════════════════════════════════════════════════════════
# ARITHMETIC QUESTION DETECTION
# ════════════════════════════════════════════════════════════════════════════

# IMPORTANT:
# Do not treat every sentence containing a number as arithmetic.
# This prevents questions such as:
#
# "Find the covariance between X and Y for 144 samples"
#
# from being incorrectly sent to the arithmetic engine.

_ARITHMETIC_COMMANDS = re.compile(
    r"\b(?:evaluate|calculate|compute|simplify|"
    r"what\s+is)\b",
    re.I,
)


_PURE_EXPRESSION_PATTERN = re.compile(
    r"^[\d\s\+\-\*\/\^\(\)\.%]+$"
)


_SQUARE_ROOT_PATTERN = re.compile(
    r"\b(?:the\s+)?"
    r"(?:square\s+root|sqrt|root)"
    r"\s+of\s+"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\b",
    re.I,
)


_POWER_PATTERN = re.compile(
    r"\b(?:what\s+is\s+)?"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*(?:\^|to\s+the\s+power\s+of|raised\s+to)"
    r"\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\b",
    re.I,
)


_FACTORIAL_PATTERN = re.compile(
    r"\b(?:factorial\s+of|factorial)\s*"
    r"(\d+)\b",
    re.I,
)


def _build_arithmetic_result(
    expression: str,
    value: float,
    description: str = "Identified arithmetic expression",
    formula: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a standard arithmetic response.
    """

    if formula is None:
        formula = expression

    return {
        "type": "arithmetic",
        "algorithm": "Safe AST Arithmetic Evaluation",
        "expression": expression,
        "result": value,
        "answer": _format_number(value),
        "confidence": 100.0,
        "solution_steps": [
            {
                "step": 1,
                "description": description,
                "formula": formula,
            },
            {
                "step": 2,
                "description": "Computed result",
                "result": f"= {_format_number(value)}",
            },
        ],
    }


def try_arithmetic(text: str) -> Optional[Dict]:
    """
    Detect and safely evaluate simple arithmetic questions.

    The function intentionally avoids extracting arbitrary numbers from
    natural-language questions because that can cause incorrect routing.
    """

    if not text:
        return None

    text = re.sub(r"[?.!]+\s*$", "", str(text).strip())

    if not text:
        return None

    # ------------------------------------------------------------
    # 1. Square root
    # ------------------------------------------------------------

    square_root_match = _SQUARE_ROOT_PATTERN.search(text)

    if square_root_match:

        operand = square_root_match.group(1)

        value = evaluate_expression(
            f"sqrt({operand})"
        )

        if value is not None:

            return _build_arithmetic_result(
                expression=f"sqrt({operand})",
                value=value,
                description="Identified a square-root expression",
                formula=f"√{operand}",
            )

    # ------------------------------------------------------------
    # 2. Full arithmetic command expression
    # ------------------------------------------------------------
    command_match = re.search(
        r"\b(?:evaluate|calculate|compute|simplify|what\s+is|find\s+(?:the\s+)?value\s+of)\b\s*(?P<expr>[-+*/%^().\d\s×÷⁰¹²³⁴⁵⁶⁷⁸⁹]+)$",
        text,
        re.I,
    )
    if command_match:
        expression = command_match.group("expr").strip()
        value = evaluate_expression(expression)
        if value is not None:
            return _build_arithmetic_result(
                expression=_normalise_expr(expression),
                value=value,
            )

    # ------------------------------------------------------------
    # 3. Power
    # ------------------------------------------------------------

    power_match = _POWER_PATTERN.search(text)

    if power_match:

        base = power_match.group(1)
        exponent = power_match.group(2)

        expression = f"{base}**{exponent}"

        value = evaluate_expression(expression)

        if value is not None:

            return _build_arithmetic_result(
                expression=expression,
                value=value,
                description="Identified a power expression",
                formula=f"{base}^{exponent}",
            )

    # ------------------------------------------------------------
    # 3. Factorial
    # ------------------------------------------------------------

    factorial_match = _FACTORIAL_PATTERN.search(text)

    if factorial_match:

        number = factorial_match.group(1)

        expression = f"factorial({number})"

        value = evaluate_expression(expression)

        if value is not None:

            return _build_arithmetic_result(
                expression=expression,
                value=value,
                description="Identified a factorial expression",
                formula=f"{number}!",
            )

    # ------------------------------------------------------------
    # 4. Pure arithmetic expression
    # ------------------------------------------------------------

    normalised = _normalise_expr(text)

    if (
        normalised
        and _PURE_EXPRESSION_PATTERN.fullmatch(normalised)
    ):

        value = evaluate_expression(normalised)

        if value is not None:

            return _build_arithmetic_result(
                expression=normalised,
                value=value,
            )

    # ------------------------------------------------------------
    # 5. Explicit arithmetic command
    #
    # Example:
    # "calculate 25 + 35"
    # "evaluate 20 * 5"
    #
    # Do NOT accept a random number extracted from a sentence.
    # ------------------------------------------------------------

    if _ARITHMETIC_COMMANDS.search(text):

        expression_match = re.search(
            r"(?P<expr>"
            r"[-+*/%^().\d\s×÷]+"
            r")$",
            text,
            re.I,
        )

        if expression_match:

            expression = expression_match.group("expr").strip()

            if expression:

                value = evaluate_expression(expression)

                if value is not None:

                    return _build_arithmetic_result(
                        expression=_normalise_expr(expression),
                        value=value,
                    )

    return None


# ════════════════════════════════════════════════════════════════════════════
# SAFE SYMBOLIC MATH FALLBACK
# ════════════════════════════════════════════════════════════════════════════

_SYMBOLIC_VARIABLES = {"x", "y", "z", "t", "a", "b", "c", "n"}
_SYMBOLIC_FUNCTIONS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "log",
    "exp", "abs", "pi", "e",
}


def _parse_safe_symbolic_expression(value: str):
    """Parse a small, explicit mathematical grammar with SymPy.

    This accepts common school-math notation only.  It deliberately rejects
    arbitrary identifiers, brackets and Python syntax before calling SymPy,
    so a question cannot be interpreted as executable Python.
    """
    if not value or len(str(value)) > MAX_EXPRESSION_LENGTH:
        return None

    source = str(value).strip().rstrip("?.!")
    source = (
        source.replace("−", "-").replace("×", "*")
        .replace("÷", "/").replace("^", "**")
        .replace("π", "pi")
    )
    source = re.sub(r"\bln\b", "log", source, flags=re.I)

    # Numbers/operators/parentheses/commas, followed by a small whitelisted
    # vocabulary.  In particular: no quotes, square brackets or underscores.
    if not re.fullmatch(r"[0-9A-Za-z+*/^().,\s-]+", source):
        return None

    identifiers = {
        token.lower()
        for token in re.findall(r"[A-Za-z]+", source)
    }
    if not identifiers.issubset(_SYMBOLIC_VARIABLES | _SYMBOLIC_FUNCTIONS):
        return None

    # Support ordinary implicit multiplication: 2x, 2sin(x), (x+1)(x-1).
    source = re.sub(r"(?<=\d)\s*(?=[A-Za-z])", "*", source)
    source = re.sub(r"\)\s*\(", ")*(", source)
    for name in _SYMBOLIC_VARIABLES:
        source = re.sub(rf"\b{name}\s*\(", f"{name}*(", source, flags=re.I)

    try:
        import sympy as sp

        locals_map = {
            name: sp.Symbol(name)
            for name in _SYMBOLIC_VARIABLES
        }
        locals_map.update({
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
            "abs": sp.Abs, "pi": sp.pi, "e": sp.E,
        })
        return sp.sympify(source, locals=locals_map, evaluate=True)
    except Exception:
        return None


def _symbolic_variable(expression):
    """Pick one deterministic variable from a supported symbolic expression."""
    try:
        import sympy as sp

        names = sorted(str(symbol) for symbol in expression.free_symbols)
        if not names:
            return None
        preferred = "x" if "x" in names else names[0]
        return sp.Symbol(preferred)
    except Exception:
        return None


def _symbolic_response(operation: str, source: str, result, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format a result produced solely from the user's symbolic expression."""
    try:
        import sympy as sp

        answer = sp.sstr(result)
    except Exception:
        answer = str(result)

    return {
        "type": "solution",
        "subject": "symbolic_math",
        "algorithm": "Safe Symbolic Mathematics Solver",
        "operation": operation,
        "expression": source,
        "answer": answer,
        "result": answer,
        "confidence": 100.0,
        "solution_steps": steps,
        "validated": True,
        "similarity_answer_copy": False,
    }


def try_symbolic_math(text: str) -> Optional[Dict[str, Any]]:
    """Solve safe, common symbolic calculus/algebra requests from user input.

    Dataset patterns can route a question here, but every value in the result
    is calculated from the current user expression, never copied from a row.
    """
    source = str(text or "").strip()
    if not source:
        return None

    patterns = (
        ("derivative", r"^(?:please\s+)?(?:find\s+)?(?:the\s+)?(?:derivative\s+of|differentiate|derive)\s+(.+?)\s*[?.!]*$"),
        ("integral", r"^(?:please\s+)?(?:find\s+)?(?:the\s+)?(?:integral\s+of|integrate)\s+(.+?)\s*[?.!]*$"),
        ("simplify", r"^(?:please\s+)?(?:simplify)\s+(.+?)\s*[?.!]*$"),
        ("factor", r"^(?:please\s+)?(?:factor|factorise|factorize)\s+(.+?)\s*[?.!]*$"),
    )

    for operation, pattern in patterns:
        match = re.match(pattern, source, flags=re.I)
        if not match:
            continue
        expression_text = match.group(1).strip()
        expression = _parse_safe_symbolic_expression(expression_text)
        if expression is None:
            return None

        try:
            import sympy as sp

            variable = _symbolic_variable(expression)
            if operation == "derivative":
                if variable is None:
                    return None
                result = sp.simplify(sp.diff(expression, variable))
                steps = [
                    {"step": 1, "description": f"Expression: {sp.sstr(expression)}"},
                    {"step": 2, "description": f"Differentiate with respect to {variable}."},
                    {"step": 3, "description": f"Derivative: {sp.sstr(result)}", "result": sp.sstr(result)},
                ]
            elif operation == "integral":
                if variable is None:
                    return None
                result = sp.simplify(sp.integrate(expression, variable))
                steps = [
                    {"step": 1, "description": f"Expression: {sp.sstr(expression)}"},
                    {"step": 2, "description": f"Integrate with respect to {variable}."},
                    {"step": 3, "description": f"Antiderivative: {sp.sstr(result)} + C", "result": f"{sp.sstr(result)} + C"},
                ]
                return _symbolic_response(operation, expression_text, f"{sp.sstr(result)} + C", steps)
            elif operation == "simplify":
                result = sp.simplify(expression)
                steps = [
                    {"step": 1, "description": f"Expression: {sp.sstr(expression)}"},
                    {"step": 2, "description": f"Simplified form: {sp.sstr(result)}", "result": sp.sstr(result)},
                ]
            else:
                result = sp.factor(expression)
                steps = [
                    {"step": 1, "description": f"Expression: {sp.sstr(expression)}"},
                    {"step": 2, "description": f"Factored form: {sp.sstr(result)}", "result": sp.sstr(result)},
                ]
            return _symbolic_response(operation, expression_text, result, steps)
        except Exception:
            return None

    return None


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM DETECTION
# ════════════════════════════════════════════════════════════════════════════

_ALGORITHM_PATTERNS = [
    (
        "newton",
        [
            "newton raphson",
            "newton-raphson",
            "newton raphson method",
        ],
    ),

    (
        "bisection",
        [
            "bisection",
            "bisection method",
        ],
    ),

    (
        "secant",
        [
            "secant method",
            "secant method",
            "secant",
        ],
    ),

    (
        "regula_falsi",
        [
            "regula falsi",
            "false position",
            "regula-falsi",
        ],
    ),

    (
        "gaussian",
        [
            "gaussian elimination",
            "gauss elimination",
            "gaussian method",
        ],
    ),

    (
        "gauss_seidel",
        [
            "gauss seidel",
            "gauss-seidel",
            "gauss seidel method",
        ],
    ),

    (
        "simpson_three_eighth",
        [
            "simpson 3/8",
            "simpson 3 8",
            "simpson three eighth",
            "simpson's 3/8",
        ],
    ),

    (
        "simpson",
        [
            "simpson's rule",
            "simpsons rule",
            "simpson rule",
            "simpson 1/3",
            "simpson one third",
        ],
    ),

    (
        "trapezoidal",
        [
            "trapezoidal rule",
            "trapezoidal method",
            "trapezoid rule",
        ],
    ),

    (
        "lagrange",
        [
            "lagrange interpolation",
            "lagrange method",
            "lagrange",
        ],
    ),

    (
        "newton_forward",
        [
            "newton forward interpolation",
            "newton forward",
            "forward interpolation",
        ],
    ),

    (
        "newton_backward",
        [
            "newton backward interpolation",
            "newton backward",
            "backward interpolation",
        ],
    ),

    (
        "euler",
        [
            "euler method",
            "euler's method",
            "euler numerical method",
        ],
    ),

    (
        "runge_kutta",
        [
            "runge kutta",
            "runge-kutta",
            "rk4",
            "rk 4",
            "fourth order runge",
        ],
    ),

    (
        "matrix_multiply",
        [
            "matrix multiplication",
            "multiply matrices",
            "matrix multiply",
            "product of matrices",
        ],
    ),

    (
        "matrix_transpose",
        [
            "matrix transpose",
            "transpose matrix",
            "transpose of matrix",
        ],
    ),

    (
        "determinant",
        [
            "determinant",
            "find determinant",
            "det of matrix",
        ],
    ),
]


def detect_algorithm(
    text: str,
    subject: Optional[str] = None
) -> Optional[str]:
    """Detect the most specific explicitly requested numerical algorithm."""
    if not text:
        return None

    text_lower = normalize_solver_text(str(text)).lower().strip()

    # Check longer/more specific phrases first so "Newton forward" is not
    # swallowed by the generic "Newton" detector.
    ordered = sorted(
        _ALGORITHM_PATTERNS,
        key=lambda item: max((len(p) for p in item[1]), default=0),
        reverse=True,
    )
    for algorithm_name, patterns in ordered:
        for pattern in sorted(patterns, key=len, reverse=True):
            if pattern in text_lower:
                return algorithm_name

    if "numerical integration" in text_lower or "numerical integral" in text_lower:
        return "simpson"

    return None


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM DISPLAY NAMES
# ════════════════════════════════════════════════════════════════════════════

_ALGORITHM_NAMES = {
    "newton": "Newton-Raphson Method",
    "bisection": "Bisection Method",
    "secant": "Secant Method",
    "regula_falsi": "Regula Falsi Method",
    "gaussian": "Gaussian Elimination",
    "gauss_seidel": "Gauss-Seidel Method",
    "trapezoidal": "Trapezoidal Rule",
    "simpson": "Simpson's 1/3 Rule",
    "simpson_three_eighth": "Simpson's 3/8 Rule",
    "lagrange": "Lagrange Interpolation",
    "newton_forward": "Newton Forward Interpolation",
    "newton_backward": "Newton Backward Interpolation",
    "euler": "Euler Method",
    "runge_kutta": "Runge-Kutta 4th Order Method",
    "matrix_multiply": "Matrix Multiplication",
    "matrix_transpose": "Matrix Transpose",
    "determinant": "Matrix Determinant",
}


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM SOLVER
# ════════════════════════════════════════════════════════════════════════════

def solve_with_algorithms(
    text: str,
    subject: Optional[str] = None
) -> Optional[Dict]:
    """
    Detect an explicitly requested numerical algorithm.

    This function does NOT pretend that an algorithm has been executed
    when required mathematical parameters are missing.

    The actual numerical functions are retained in alorithm.py and can
    be connected to parameter parsers in a later upgrade.
    """

    if not text:
        return None

    algorithm = detect_algorithm(
        text,
        subject
    )

    if algorithm is None:
        return None

    algorithm_name = _ALGORITHM_NAMES.get(
        algorithm,
        algorithm.replace("_", " ").title()
    )

    if not ALGORITHMS_AVAILABLE:

        return {
            "type": "algorithm",
            "algorithm": algorithm_name,
            "status": "unavailable",
            "message": (
                f"{algorithm_name} was detected, "
                "but the numerical algorithm module is unavailable."
            ),
            "solution_steps": [
                {
                    "step": 1,
                    "description": "Detected requested algorithm.",
                    "algorithm": algorithm_name,
                },
                {
                    "step": 2,
                    "description": (
                        "Algorithm implementation could not be loaded."
                    ),
                },
            ],
        }

    return {
        "type": "algorithm",
        "algorithm": algorithm_name,
        "algorithm_id": algorithm,
        "status": "detected",
        "message": (
            f"{algorithm_name} was detected. "
            "The problem requires the corresponding numerical "
            "parameters before execution."
        ),
        "solution_steps": [
            {
                "step": 1,
                "description": "Detected requested numerical method.",
                "algorithm": algorithm_name,
            },
            {
                "step": 2,
                "description": (
                    "Problem parameters must be extracted "
                    "before executing the algorithm."
                ),
            },
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# TEXT TOKENIZATION
# ════════════════════════════════════════════════════════════════════════════

def _tokenise(text: str) -> List[str]:
    """
    Tokenize text for similarity matching.

    Mathematical symbols are preserved as useful tokens where possible.
    """

    if not text:
        return []

    text = str(text).lower()

    return re.findall(
        r"[a-z]+(?:[a-z0-9]*[a-z])?|[0-9]+(?:\.[0-9]+)?",
        text
    )


def _cosine(
    a: Dict[str, float],
    b: Dict[str, float]
) -> float:
    """
    Calculate cosine similarity between two sparse vectors.
    """

    if not a or not b:
        return 0.0

    dot = sum(
        a.get(word, 0.0) * value
        for word, value in b.items()
    )

    norm_a = math.sqrt(
        sum(
            value * value
            for value in a.values()
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in b.values()
        )
    )

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def _term_freq(tokens: List[str]) -> Dict[str, float]:
    """
    Calculate normalized term frequency.
    """

    if not tokens:
        return {}

    counts = Counter(tokens)

    total = len(tokens)

    return {
        word: count / total
        for word, count in counts.items()
    }


# ════════════════════════════════════════════════════════════════════════════
# DATASET MATCH INDEX
# ════════════════════════════════════════════════════════════════════════════

class _MatchIndex:
    """
    Pre-tokenized question vectors for repeated similarity searches.
    """

    def __init__(
        self,
        entries: List[dict]
    ) -> None:

        self.entries = entries

        self.vectors = []

        self.norms = []

        for entry in entries:

            question = str(
                entry.get("question", "")
            )

            vector = _term_freq(
                _tokenise(question)
            )

            norm = math.sqrt(
                sum(
                    value * value
                    for value in vector.values()
                )
            )

            self.vectors.append(vector)

            self.norms.append(norm)

    def best_match(
        self,
        query: str
    ) -> Tuple[Optional[dict], float]:

        query_tokens = _tokenise(query)

        query_vector = _term_freq(
            query_tokens
        )

        query_norm = math.sqrt(
            sum(
                value * value
                for value in query_vector.values()
            )
        )

        if not query_norm:
            return None, 0.0

        best_entry = None
        best_score = 0.0

        for entry, vector, norm in zip(
            self.entries,
            self.vectors,
            self.norms
        ):

            if not norm:
                continue

            dot = sum(
                value * vector.get(
                    word,
                    0.0
                )
                for word, value
                in query_vector.items()
            )

            score = dot / (
                query_norm * norm
            )

            if score > best_score:

                best_entry = entry
                best_score = score

        return best_entry, best_score


# ════════════════════════════════════════════════════════════════════════════
# MATCH CACHE
# ════════════════════════════════════════════════════════════════════════════

_MATCH_INDEX_CACHE: "OrderedDict[Tuple[str, ...], _MatchIndex]" = (
    OrderedDict()
)

_MATCH_INDEX_CACHE_LIMIT = 32

# Exact lookup is intentionally separate from similarity matching.  Cache by
# list identity so a dataset reload naturally receives a new index.
_EXACT_MATCH_INDEX_CACHE: "OrderedDict[Tuple[int, int], Dict[str, Dict[str, Any]]]" = (
    OrderedDict()
)
_EXACT_MATCH_INDEX_CACHE_LIMIT = 16


def _get_match_index(
    entries: List[dict]
) -> _MatchIndex:
    """
    Return a cached match index for a dataset.
    """

    key = tuple(
        str(
            entry.get(
                "question",
                ""
            )
        )
        for entry in entries
    )

    index = _MATCH_INDEX_CACHE.pop(
        key,
        None
    )

    if index is None:

        index = _MatchIndex(
            entries
        )

    _MATCH_INDEX_CACHE[key] = index

    while len(_MATCH_INDEX_CACHE) > _MATCH_INDEX_CACHE_LIMIT:

        _MATCH_INDEX_CACHE.popitem(
            last=False
        )

    return index


def clear_match_cache() -> None:
    """
    Clear cached dataset indexes.

    Useful after a manual dataset reload.
    """

    _MATCH_INDEX_CACHE.clear()
    _EXACT_MATCH_INDEX_CACHE.clear()


# ════════════════════════════════════════════════════════════════════════════
# DATASET MATCHING
# ════════════════════════════════════════════════════════════════════════════

def _get_match_threshold() -> float:
    """
    Read MATCH_THRESHOLD safely from config.py.
    """

    try:

        threshold = float(
            getattr(
                config,
                "MATCH_THRESHOLD",
                0.55
            )
        )

        return max(
            0.0,
            min(
                threshold,
                1.0
            )
        )

    except Exception:

        return 0.55


def find_best_match(
    query: str,
    entries: List[dict],
    threshold: float = None,
) -> Tuple[Optional[dict], float]:
    """
    Find the dataset question most similar to the user's question.

    Returns:
        (matching_entry, similarity_score)

    If the score is below threshold, the entry is rejected.
    """

    if not query:
        return None, 0.0

    if not entries:
        return None, 0.0

    if threshold is None:
        threshold = _get_match_threshold()

    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.55

    threshold = max(
        0.0,
        min(
            threshold,
            1.0
        )
    )

    best_entry, best_score = _get_match_index(
        entries
    ).best_match(
        query
    )

    if best_entry is None:
        return None, best_score

    if best_score < threshold:
        return None, best_score

    return best_entry, best_score


def _canonical_dataset_question(value: Any) -> str:
    """Create a strict comparison key for exact dataset-question lookup.

    This intentionally does *not* perform fuzzy matching.  It allows harmless
    OCR differences such as whitespace, typographic multiplication signs and
    final sentence punctuation, while keeping different numbers/operators as
    different mathematical questions.
    """
    # Keep this deliberately lightweight because a large dataset may contain
    # tens of thousands of questions.  Rich/fuzzy NLP belongs to routing;
    # exact lookup only normalizes equivalent typography and whitespace.
    text = str(value or "").lower()
    text = (
        text.replace("−", "-").replace("–", "-").replace("—", "-")
        .replace("×", "*").replace("·", "*").replace("÷", "/")
        .replace("°", " degrees ").replace("²", "^2").replace("³", "^3")
    )
    text = text.replace("**", "^")
    text = re.sub(r"[?.!]+$", "", text)
    return re.sub(r"\s+", "", text).strip()


def _canonical_dataset_answer(value: Any) -> str:
    """Create a comparison key for answers from duplicate dataset rows."""
    text = str(value or "").lower()
    text = (
        text.replace("−", "-").replace("–", "-").replace("—", "-")
        .replace("×", "*").replace("·", "*").replace("÷", "/")
        .replace("°", " degrees ").replace("²", "^2").replace("³", "^3")
    )
    return re.sub(r"\s+", "", text).strip().rstrip(".")


def _exact_entry_answer(entry: dict) -> Optional[str]:
    """Extract one explicit displayed answer from a dataset entry."""
    steps = entry.get("solution", [])
    if not steps:
        steps = entry.get("solution_steps", [])
    return _dataset_entry_answer(entry, _normalise_solution_steps(steps))


def get_exact_dataset_match_status(
    query: str,
    entries: Optional[List[dict]],
) -> Dict[str, Any]:
    """Return exact-match metadata without silently resolving bad data.

    An identical question may occur more than once in a supplied dataset.  It
    is safe to return the stored solution only when all duplicate rows have
    the same final answer.  A conflict means the data itself cannot prove a
    single answer, so callers must calculate the result independently or tell
    the user that the dataset needs correction.
    """
    query_key = _canonical_dataset_question(query)
    empty = {
        "entry": None,
        "exact_match": False,
        "ambiguous": False,
        "answers": [],
        "record_count": 0,
    }
    if not query_key or not entries:
        return empty

    cache_key = (id(entries), len(entries))
    index = _EXACT_MATCH_INDEX_CACHE.get(cache_key)
    if index is None:
        index = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = _canonical_dataset_question(entry.get("question", ""))
            if not key:
                continue

            answer = _exact_entry_answer(entry)
            answer_key = _canonical_dataset_answer(answer)
            record = index.setdefault(
                key,
                {
                    "entry": entry,
                    "answers": [],
                    "answer_keys": set(),
                    "record_count": 0,
                },
            )
            record["record_count"] += 1
            if answer_key not in record["answer_keys"]:
                record["answer_keys"].add(answer_key)
                record["answers"].append(answer or "(missing answer)")

        _EXACT_MATCH_INDEX_CACHE[cache_key] = index
        _EXACT_MATCH_INDEX_CACHE.move_to_end(cache_key)
        while len(_EXACT_MATCH_INDEX_CACHE) > _EXACT_MATCH_INDEX_CACHE_LIMIT:
            _EXACT_MATCH_INDEX_CACHE.popitem(last=False)
    else:
        _EXACT_MATCH_INDEX_CACHE.move_to_end(cache_key)

    record = index.get(query_key)
    if record is None:
        return empty

    answers = list(record["answers"])
    ambiguous = len(record["answer_keys"]) > 1
    return {
        "entry": None if ambiguous else record["entry"],
        "exact_match": True,
        "ambiguous": ambiguous,
        "answers": answers,
        "record_count": int(record["record_count"]),
    }


def find_exact_dataset_match(
    query: str,
    entries: Optional[List[dict]],
) -> Optional[dict]:
    """Return an exact record only when duplicate rows agree on its answer."""
    return get_exact_dataset_match_status(query, entries).get("entry")


def get_exact_dataset_integrity(entries: Optional[List[dict]]) -> Dict[str, int]:
    """Summarise answer coverage and contradictory duplicate questions."""
    records = entries or []
    unique_questions = set()
    missing_questions = 0
    missing_answers = 0

    for entry in records:
        if not isinstance(entry, dict):
            missing_questions += 1
            continue
        key = _canonical_dataset_question(entry.get("question", ""))
        if not key:
            missing_questions += 1
            continue
        unique_questions.add(key)
        if not _exact_entry_answer(entry):
            missing_answers += 1

    ambiguous_questions = 0
    ambiguous_records = 0
    # The first status call builds the cached index; every following call is
    # constant-time and never performs a fuzzy match.
    for key in unique_questions:
        status = get_exact_dataset_match_status(key, records)
        if status.get("ambiguous"):
            ambiguous_questions += 1
            ambiguous_records += int(status.get("record_count", 0))

    return {
        "records": len(records),
        "unique_questions": len(unique_questions),
        "missing_questions": missing_questions,
        "missing_final_answers": missing_answers,
        "conflicting_duplicate_questions": ambiguous_questions,
        "conflicting_records": ambiguous_records,
    }


def _normalise_solution_steps(steps: Any) -> List[Dict[str, Any]]:
    """Convert supported dataset solution formats into frontend-safe steps."""
    if isinstance(steps, str):
        steps = [steps]
    if not isinstance(steps, (list, tuple)):
        return []

    normalised = []
    for index, item in enumerate(steps, start=1):
        if isinstance(item, dict):
            step = dict(item)
            step.setdefault("step", index)
            normalised.append(step)
        elif item is not None:
            normalised.append({
                "step": index,
                "description": str(item),
            })
    return normalised


def _dataset_entry_answer(entry: dict, steps: List[Dict[str, Any]]) -> Optional[str]:
    """Read only an explicit final answer from an exact dataset record."""
    for key in ("answer", "final_answer", "result"):
        value = entry.get(key)
        if value not in (None, "", [], {}):
            return str(value)

    for step in reversed(steps):
        for key in ("answer", "result"):
            value = step.get(key)
            if value not in (None, "", [], {}):
                return str(value)

        description = str(step.get("description", "")).strip()
        final = re.search(
            r"(?:final\s+(?:answer|prediction)|answer)\s*:\s*(.+?)\s*$",
            description,
            re.I,
        )
        if final:
            final_text = final.group(1).rstrip(". ")
            classified = re.search(
                r"classified\s+as\s+['\"]?([A-Za-z][A-Za-z _-]*)['\"]?",
                final_text,
                re.I,
            )
            if classified:
                return classified.group(1).rstrip(". ")
            return final_text

        classified = re.search(
            r"classified\s+as\s+['\"]?([A-Za-z][A-Za-z _-]*)['\"]?",
            description,
            re.I,
        )
        if classified:
            return classified.group(1).rstrip(". ")

        # Several datasets end their worked solution with a named result,
        # e.g. "Covariance = 13.92".  Display it as the final answer while
        # retaining every original step above it.
        if re.search(
            r"\b(?:covariance|correlation|mean|median|mode|variance|"
            r"standard\s+deviation|determinant)\s*=",
            description,
            re.I,
        ):
            return description.rstrip(". ")
    return None


# ════════════════════════════════════════════════════════════════════════════
# DATASET SOLUTION FORMATTER
# ════════════════════════════════════════════════════════════════════════════

def format_solution(
    entry: dict,
    subject: str,
    score: float
) -> dict:
    """
    Convert a dataset entry into the standard API response format.
    """

    steps = entry.get(
        "solution",
        []
    )

    # Support alternative dataset schemas.
    if not steps:

        steps = entry.get(
            "solution_steps",
            []
        )

    steps = _normalise_solution_steps(steps)

    return {
        "type": "dataset_match",

        "subject": subject,

        "matched_question": entry.get(
            "question",
            ""
        ),

        "confidence": round(
            max(
                0.0,
                min(
                    float(score) * 100.0,
                    100.0
                )
            ),
            1
        ),

        "solution_steps": steps,

        # IMPORTANT: Never expose/copy a stored dataset answer.
        # Dataset entries are used only for method/pattern reference.
        "answer_copied_from_dataset": False,
    }


def format_exact_dataset_solution(entry: dict, subject: str) -> dict:
    """Return a verified-by-identity dataset response for the same question.

    A dataset solution may be displayed only after exact normalized question
    equality.  This is deliberately separate from similarity matching: no
    nearest-neighbour answer is ever used for a different problem.
    """
    response = format_solution(entry, subject, 1.0)
    steps = response["solution_steps"]
    answer = _dataset_entry_answer(entry, steps)
    response.update({
        "type": "exact_dataset_match",
        "confidence": 100.0,
        "exact_dataset_match": True,
        "dataset_solution_source": "same_question",
        "similarity_answer_copy": False,
        "answer_copied_from_dataset": bool(answer is not None),
        "validation": "The entered question exactly matches a dataset record.",
    })
    if answer is not None:
        response["answer"] = answer
    return response


# ════════════════════════════════════════════════════════════════════════════
# NO-MATCH RESPONSE
# ════════════════════════════════════════════════════════════════════════════

def format_no_match(
    query: str,
    subject: str
) -> dict:
    """
    Standard response when no sufficiently similar dataset question exists.
    """

    return {
        "type": "no_match",

        "subject": subject,

        "query": query,

        "confidence": 0.0,

        "solution_steps": [],

        "message": (
            "No sufficiently similar problem was found "
            "in the selected dataset. Please rephrase "
            "your question or provide a more specific "
            "mathematical expression."
        ),
    }

# ════════════════════════════════════════════════════════════════════════════
# DATASET-ASSISTED STATISTICAL SOLVERS
# ════════════════════════════════════════════════════════════════════════════

def _extract_xy_lists(text: str) -> Optional[Tuple[List[float], List[float]]]:
    """Extract X=[...] and Y=[...] numeric lists from a question."""
    if not text:
        return None

    value = str(text).replace("−", "-").replace("–", "-")

    x_match = re.search(r"\bX\s*=\s*\[([^\]]+)\]", value, re.I)
    y_match = re.search(r"\bY\s*=\s*\[([^\]]+)\]", value, re.I)

    if not x_match or not y_match:
        return None

    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

    def parse_list(raw: str) -> List[float]:
        parts = re.findall(number_pattern, raw)
        return [float(item) for item in parts]

    x_values = parse_list(x_match.group(1))
    y_values = parse_list(y_match.group(1))

    if not x_values or not y_values or len(x_values) != len(y_values):
        return None

    return x_values, y_values


def _fmt_stat_number(value: float, decimals: int = 3) -> str:
    """Format statistical numbers without unnecessary trailing zeroes."""
    try:
        value = float(value)
        if abs(value) < 1e-12:
            value = 0.0
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)



def _get_dataset_method_learning(
    question: str,
    subject: Optional[str] = None,
    dataset_entries: Optional[List[dict]] = None,
    supplied_context: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Retrieve the learned method/formula representation for a new question.

    The learning layer may identify:
        - problem type / method
        - formula references
        - variable hints
        - learned patterns
        - confidence

    IMPORTANT:
        No stored final answer is read from this context.
        Mathematical results are always computed from the user's question.
    """
    if supplied_context and isinstance(supplied_context, dict):
        return supplied_context

    try:
        from utils import dataset_loader

        detector = getattr(dataset_loader, "get_learned_method_candidates", None)
        if not callable(detector):
            detector = getattr(dataset_loader, "get_pattern_candidates", None)

        if not callable(detector):
            return {
                "enabled": False,
                "learning_type": "unavailable",
                "answer_copy_allowed": False,
                "candidates": [],
            }

        try:
            candidates = detector(
                question,
                subject=subject,
                top_k=5,
            )
        except TypeError:
            try:
                candidates = detector(question, subject=subject)
            except TypeError:
                candidates = detector(question)

        if not isinstance(candidates, list):
            candidates = []

        safe_candidates: List[Dict[str, Any]] = []
        for item in candidates[:5]:
            if not isinstance(item, dict):
                continue
            safe_candidates.append({
                "subject": item.get("subject"),
                "method": item.get("method"),
                "score": item.get("score", 0.0),
                "confidence": item.get("confidence", 0.0),
                "patterns": list(item.get("patterns", []) or []),
                "formula_references": list(
                    item.get("formula_references", []) or []
                ),
                "variable_hints": list(item.get("variable_hints", []) or []),
                "training_examples": item.get("training_examples", 0),
                "learning_type": item.get(
                    "learning_type",
                    "dataset_learned_method_selection",
                ),
                "answer_copy_allowed": False,
            })

        return {
            "enabled": bool(safe_candidates),
            "learning_type": "dataset_learned_method_selection",
            "answer_copy_allowed": False,
            "candidate_count": len(safe_candidates),
            "best_method": (
                safe_candidates[0].get("method")
                if safe_candidates
                else None
            ),
            "best_subject": (
                safe_candidates[0].get("subject")
                if safe_candidates
                else None
            ),
            "best_confidence": (
                safe_candidates[0].get("confidence", 0.0)
                if safe_candidates
                else 0.0
            ),
            "candidates": safe_candidates,
            "reference_examples": len(dataset_entries or []),
        }
    except Exception as exc:
        logger.debug("Dataset method learning unavailable: %s", exc)
        return {
            "enabled": False,
            "learning_type": "unavailable",
            "answer_copy_allowed": False,
            "candidates": [],
            "reference_examples": len(dataset_entries or []),
        }


def _learning_best_method(learning_context: Optional[dict]) -> str:
    """Return the strongest learned method label without using any answer."""
    if not isinstance(learning_context, dict):
        return ""
    method = learning_context.get("best_method")
    if method:
        return str(method).strip().lower().replace("-", "_").replace(" ", "_")
    candidates = learning_context.get("candidates", [])
    if candidates and isinstance(candidates[0], dict):
        method = candidates[0].get("method")
        if method:
            return str(method).strip().lower().replace("-", "_").replace(" ", "_")
    return ""


def _learning_metadata(
    dataset_entries: Optional[List[dict]],
    learning_context: Optional[dict],
) -> Dict[str, Any]:
    """Build safe dataset-learning provenance for solver responses."""
    context = learning_context or _get_dataset_method_learning(
        "",
        dataset_entries=dataset_entries,
    )
    return {
        "enabled": bool(context.get("enabled")),
        "learning_type": context.get(
            "learning_type",
            "method_formula_pattern_learning",
        ),
        "best_method": context.get("best_method"),
        "best_subject": context.get("best_subject"),
        "best_confidence": context.get("best_confidence", 0.0),
        "candidate_count": context.get("candidate_count", 0),
        "answer_copy_allowed": False,
    }


def _dataset_learning_summary(
    dataset_entries: Optional[List[dict]],
    learning_context: Optional[dict] = None,
    question: str = "",
    subject: Optional[str] = None,
) -> dict:
    """Return dataset-learning metadata without exposing stored answers."""
    entries = dataset_entries or []
    context = learning_context
    if context is None and question:
        context = _get_dataset_method_learning(
            question,
            subject=subject,
            dataset_entries=entries,
        )
    summary = {
        "reference_examples": len(entries),
        "used_as_reference": bool(entries),
        "answer_copied_from_dataset": False,
        "learning_type": (
            context.get("learning_type", "method_formula_pattern_learning")
            if isinstance(context, dict)
            else "method_formula_pattern_learning"
        ),
        "best_method": (
            context.get("best_method")
            if isinstance(context, dict)
            else None
        ),
        "best_confidence": (
            context.get("best_confidence", 0.0)
            if isinstance(context, dict)
            else 0.0
        ),
    }
    return summary


def solve_covariance(
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Solve covariance directly from X and Y values.

    The dataset is used only as reference/context. The answer is calculated
    from the new X/Y values supplied by the user.

    This uses population covariance, matching the MCH dataset format:
        Cov(X,Y) = sum((xi-x_mean)(yi-y_mean)) / n
    """
    learning_context = _get_dataset_method_learning(
        question,
        subject="mch",
        dataset_entries=dataset_entries,
        supplied_context=dataset_learning,
    )

    if not question:
        return None

    if not re.search(r"\bcovariance\b", str(question), re.I):
        return None

    extracted = _extract_xy_lists(question)
    if extracted is None:
        return None

    x_values, y_values = extracted
    n = len(x_values)

    if n < 2:
        return None

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    products = []
    for x, y in zip(x_values, y_values):
        dx = x - x_mean
        dy = y - y_mean
        product = dx * dy
        products.append(product)

    sum_products = sum(products)
    covariance = sum_products / n

    steps: List[Dict[str, Any]] = [
        {
            "step": 1,
            "description": f"Number of observations n = {n}",
        },
        {
            "step": 2,
            "description": f"Mean of X = {_fmt_stat_number(x_mean)}",
            "formula": f"X̄ = ΣX / n = {_fmt_stat_number(x_mean)}",
        },
        {
            "step": 3,
            "description": f"Mean of Y = {_fmt_stat_number(y_mean)}",
            "formula": f"Ȳ = ΣY / n = {_fmt_stat_number(y_mean)}",
        },
        {
            "step": 4,
            "description": "Compute deviations and products:",
            "formula": "dx = xi − X̄,  dy = yi − Ȳ,  product = dx × dy",
        },
    ]

    for index, (x, y, product) in enumerate(
        zip(x_values, y_values, products),
        start=1,
    ):
        dx = x - x_mean
        dy = y - y_mean
        steps.append(
            {
                "step": len(steps) + 1,
                "description": (
                    f"For ({_fmt_stat_number(x)}, {_fmt_stat_number(y)}): "
                    f"dx = {_fmt_stat_number(dx)}, "
                    f"dy = {_fmt_stat_number(dy)}, "
                    f"product = {_fmt_stat_number(product)}"
                ),
            }
        )

    steps.append(
        {
            "step": len(steps) + 1,
            "description": f"Sum of products = {_fmt_stat_number(sum_products)}",
            "formula": "Σ(dx × dy) = " + _fmt_stat_number(sum_products),
        }
    )
    steps.append(
        {
            "step": len(steps) + 1,
            "description": (
                f"Covariance = Σ(dx × dy) / n = "
                f"{_fmt_stat_number(sum_products)} / {n}"
            ),
        }
    )
    steps.append(
        {
            "step": len(steps) + 1,
            "description": f"Covariance = {_fmt_stat_number(covariance)}",
            "result": _fmt_stat_number(covariance),
        }
    )

    return {
        "type": "solution",
        "subject": "mch",
        "algorithm": "Dataset-Assisted Covariance Solver",
        "answer": _fmt_stat_number(covariance),
        "result": covariance,
        "covariance": covariance,
        "n": n,
        "mean_x": x_mean,
        "mean_y": y_mean,
        "confidence": 100.0,
        "solution_steps": steps,
        "dataset_assisted": bool(dataset_entries),
        "dataset_entries": len(dataset_entries or []),
        "dataset_learning": _dataset_learning_summary(dataset_entries, learning_context=learning_context, question=question if 'question' in locals() else text, subject='mch'),
        "similarity_answer_copy": False,
        "validated": True,
    }


# ════════════════════════════════════════════════════════════════════════════
# DATASET-ASSISTED DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def _extract_numeric_list(text: str) -> Optional[List[float]]:
    """Extract one bracketed numeric list from the user's actual question."""
    if not text:
        return None
    match = re.search(r"\[\s*([^\]]+)\s*\]", str(text))
    if not match:
        return None
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", match.group(1))
    if not numbers:
        return None
    try:
        return [float(value) for value in numbers]
    except ValueError:
        return None


def _statistics_base_response(
    subject: str,
    question: str,
    dataset_entries: Optional[List[dict]],
    learning_context: dict,
) -> dict:
    return {
        "subject": subject,
        "dataset_assisted": bool(dataset_entries),
        "dataset_entries": len(dataset_entries or []),
        "dataset_learning": _dataset_learning_summary(
            dataset_entries,
            learning_context=learning_context,
            question=question,
            subject=subject,
        ),
        "similarity_answer_copy": False,
        "validated": True,
    }


def _statistics_operation(text: str) -> Optional[str]:
    """Detect a supported descriptive-statistics operation from natural language."""
    low = normalize_solver_text(text).lower()
    checks = [
        ("standard_deviation", (
            "standard deviation", "std deviation", "std. deviation",
            "standard dev", "std dev"
        )),
        ("variance", ("variance",)),
        ("median", ("median",)),
        ("mode", ("mode",)),
        ("mean", ("mean", "average", "arithmetic mean")),
        ("correlation", (
            "correlation", "correlation coefficient", "pearson correlation",
            "pearson's correlation", "pearson coefficient"
        )),
        ("covariance", ("covariance",)),
    ]
    for operation, phrases in checks:
        for phrase in phrases:
            if " " in phrase:
                if phrase in low:
                    return operation
            elif re.search(r"\b" + re.escape(phrase) + r"\b", low):
                return operation
    return None


def _statistics_result(
    question: str,
    operation: str,
    answer: str,
    result: Any,
    steps: List[Dict[str, Any]],
    dataset_entries: Optional[List[dict]],
    learning_context: Optional[dict],
) -> Dict[str, Any]:
    response = {
        "type": "solution",
        "subject": "statistics",
        "algorithm": f"Dataset-Assisted {operation.replace('_', ' ').title()} Solver",
        "operation": operation,
        "answer": answer,
        "result": result,
        "confidence": 100.0,
        "solution_steps": steps,
    }
    response.update(
        _statistics_base_response(
            "statistics", question, dataset_entries, learning_context or {}
        )
    )
    response["similarity_answer_copy"] = False
    return response


def solve_descriptive_statistics(
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
    operation_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Solve mean, median, mode, variance and standard deviation from user data."""
    if not question:
        return None

    text = str(question).strip()
    operation = operation_override or _statistics_operation(text)
    if operation not in {
        "mean", "median", "mode", "variance", "standard_deviation"
    }:
        return None

    values = _extract_numeric_list(text)
    if values is None or not values:
        return None

    learning_context = _get_dataset_method_learning(
        text,
        subject="statistics",
        dataset_entries=dataset_entries,
        supplied_context=dataset_learning,
    )

    n = len(values)
    mean = sum(values) / n
    steps: List[Dict[str, Any]] = [
        {"step": 1, "description": f"Data = {values}"},
        {"step": 2, "description": f"Number of observations n = {n}"},
    ]

    if operation == "mean":
        total = sum(values)
        result = total / n
        answer = _fmt_stat_number(result)
        steps.extend([
            {"step": 3, "formula": "x̄ = Σx / n",
             "calculation": f"x̄ = { _fmt_stat_number(total) } / {n}"},
            {"step": 4, "description": f"Final Answer: {answer}", "result": answer},
        ])
        return _statistics_result(text, operation, answer, result, steps,
                                  dataset_entries, learning_context)

    if operation == "median":
        ordered = sorted(values)
        if n % 2:
            result = ordered[n // 2]
            middle = f"{_fmt_stat_number(result)}"
        else:
            result = (ordered[n // 2 - 1] + ordered[n // 2]) / 2
            middle = (
                f"({_fmt_stat_number(ordered[n // 2 - 1])} + "
                f"{_fmt_stat_number(ordered[n // 2])}) / 2"
            )
        answer = _fmt_stat_number(result)
        steps.extend([
            {"step": 3, "description": f"Arrange data in ascending order: {ordered}"},
            {"step": 4, "formula": "Median = middle value" if n % 2 else "Median = average of the two middle values",
             "calculation": middle},
            {"step": 5, "description": f"Final Answer: {answer}", "result": answer},
        ])
        return _statistics_result(text, operation, answer, result, steps,
                                  dataset_entries, learning_context)

    if operation == "mode":
        counts = Counter(values)
        highest = max(counts.values())
        if highest == 1:
            answer = "No mode"
            result = None
            description = "Every value occurs once, so there is no mode."
        else:
            modes = sorted(v for v, count in counts.items() if count == highest)
            answer = ", ".join(_fmt_stat_number(v) for v in modes)
            result = modes[0] if len(modes) == 1 else modes
            description = f"Highest frequency = {highest}; mode(s) = {answer}."
        steps.extend([
            {"step": 3, "description": f"Frequencies = {dict(counts)}"},
            {"step": 4, "description": description, "result": answer},
        ])
        return _statistics_result(text, operation, answer, result, steps,
                                  dataset_entries, learning_context)

    # Variance / standard deviation.
    squared = [(x - mean) ** 2 for x in values]
    total_squared = sum(squared)
    is_sample = bool(
        re.search(r"\bsample\s+(?:variance|standard\s+deviation)\b", text, re.I)
        or re.search(r"\b(?:variance|standard\s+deviation)\s+of\s+the\s+sample\b", text, re.I)
    )
    denominator = n - 1 if is_sample else n
    if denominator <= 0:
        return None
    variance = total_squared / denominator
    result = math.sqrt(variance) if operation == "standard_deviation" else variance
    answer = _fmt_stat_number(result)
    label = "Sample" if is_sample else "Population"
    steps.extend([
        {"step": 3, "description": f"Mean = Σx / n = {_fmt_stat_number(mean)}",
         "formula": "x̄ = Σx / n"},
        {"step": 4, "description": f"Sum of squared deviations = {_fmt_stat_number(total_squared)}",
         "formula": "Σ(xi − x̄)²"},
        {"step": 5, "description": (
            f"{label} variance = {_fmt_stat_number(total_squared)} / {denominator} "
            f"= {_fmt_stat_number(variance)}"
        )},
    ])
    if operation == "standard_deviation":
        steps.append({
            "step": 6,
            "description": f"{label} standard deviation = √{_fmt_stat_number(variance)} = {answer}",
            "result": answer,
        })
    else:
        steps.append({
            "step": 6,
            "description": f"Final Answer: {answer}",
            "result": answer,
        })
    return _statistics_result(text, operation, answer, result, steps,
                              dataset_entries, learning_context)


def solve_variance(
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible variance entry point."""
    return solve_descriptive_statistics(
        question,
        dataset_entries=dataset_entries,
        dataset_learning=dataset_learning,
    )


def _statistics_operations(text: str) -> List[str]:
    """Return all explicitly requested supported statistics operations."""
    low = normalize_solver_text(str(text)).lower()
    found = []
    checks = [
        ("standard_deviation", ("standard deviation", "std deviation", "std. deviation", "standard dev", "std dev")),
        ("variance", ("variance",)),
        ("median", ("median",)),
        ("mode", ("mode",)),
        ("mean", ("mean", "average", "arithmetic mean")),
        ("correlation", ("correlation", "correlation coefficient", "pearson correlation", "pearson's correlation", "pearson coefficient")),
        ("covariance", ("covariance",)),
    ]
    for operation, phrases in checks:
        if any((phrase in low) if " " in phrase or "'" in phrase else re.search(r"\b" + re.escape(phrase) + r"\b", low) for phrase in phrases):
            found.append(operation)
    return found


def solve_statistics(
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Central statistics dispatcher using learned method signals."""
    if not question:
        return None

    text = str(question)
    operations = _statistics_operations(text)
    operation = operations[0] if operations else None

    # Multi-operation statistics questions are solved independently and then
    # combined. Dataset records may guide method selection, but never supply
    # a stored final answer.
    if len(operations) > 1:
        combined_results = {}
        combined_steps = []
        for op in operations:
            if op in {"mean", "median", "mode", "variance", "standard_deviation"}:
                result = solve_descriptive_statistics(
                    text, dataset_entries=dataset_entries,
                    dataset_learning=dataset_learning, operation_override=op,
                )
            elif op == "covariance":
                result = solve_covariance(
                    text, dataset_entries=dataset_entries,
                    dataset_learning=dataset_learning,
                )
            elif op == "correlation":
                result = None
                extracted = _extract_xy_lists(text)
                if extracted is not None:
                    x_values, y_values = extracted
                    n = len(x_values)
                    if n >= 2:
                        x_mean = sum(x_values) / n
                        y_mean = sum(y_values) / n
                        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
                        sx = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
                        sy = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
                        if sx and sy:
                            r = numerator / (sx * sy)
                            answer = _fmt_stat_number(r)
                            result = {"answer": answer, "result": r, "solution_steps": []}
            if result is not None:
                combined_results[op] = result.get("result", result.get("answer"))
                combined_steps.extend(result.get("solution_steps", []))

        if len(combined_results) == len(operations):
            labels = {
                "standard_deviation": "Standard Deviation",
                "variance": "Variance", "median": "Median",
                "mode": "Mode", "mean": "Mean",
                "correlation": "Correlation", "covariance": "Covariance",
            }
            answer_parts = [f"{labels[op]} = {_fmt_stat_number(combined_results[op]) if isinstance(combined_results[op], (int, float)) else combined_results[op]}" for op in operations]
            answer = ", ".join(answer_parts)
            return {
                "type": "solution", "subject": "statistics",
                "algorithm": "Dataset-Assisted Multi-Statistics Solver",
                "operation": "multiple_statistics", "operations": operations,
                "answer": answer, "result": combined_results,
                "confidence": 100.0, "solution_steps": combined_steps,
                **_statistics_base_response("statistics", text, dataset_entries,
                                            _get_dataset_method_learning(text, subject="statistics", dataset_entries=dataset_entries, supplied_context=dataset_learning)),
                "similarity_answer_copy": False,
            }

    if operation == "covariance":
        return solve_covariance(
            text,
            dataset_entries=dataset_entries,
            dataset_learning=dataset_learning,
        )

    if operation == "correlation":
        extracted = _extract_xy_lists(text)
        if extracted is None:
            return None
        x_values, y_values = extracted
        n = len(x_values)
        if n < 2:
            return None
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        numerator = sum((x - x_mean) * (y - y_mean)
                        for x, y in zip(x_values, y_values))
        sx = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
        sy = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
        if sx == 0 or sy == 0:
            return None
        r = numerator / (sx * sy)
        answer = _fmt_stat_number(r)
        steps = [
            {"step": 1, "description": f"X = {x_values}; Y = {y_values}"},
            {"step": 2, "formula": "r = Σ[(xi−x̄)(yi−ȳ)] / √(Σ(xi−x̄)² Σ(yi−ȳ)²)"},
            {"step": 3, "description": f"Numerator = {_fmt_stat_number(numerator)}"},
            {"step": 4, "description": f"Correlation coefficient r = {answer}",
             "result": answer},
        ]
        learning_context = _get_dataset_method_learning(
            text, subject="statistics", dataset_entries=dataset_entries,
            supplied_context=dataset_learning,
        )
        return _statistics_result(text, operation, answer, r, steps,
                                  dataset_entries, learning_context)

    return solve_descriptive_statistics(
        text,
        dataset_entries=dataset_entries,
        dataset_learning=dataset_learning,
    )


# ════════════════════════════════════════════════════════════════════════════
# FIND-X SOLVER
# ════════════════════════════════════════════════════════════════════════════

def _extract_find_x_equation(text: str) -> Optional[Tuple[str, str]]:
    """Extract the mathematical left/right sides from a Find-X question."""
    if not text or "=" not in str(text):
        return None

    value = str(text).strip()
    parts = value.split("=", 1)
    if len(parts) != 2:
        return None

    left, right = parts[0].strip(), parts[1].strip()

    # Remove only the instruction prefix. The previous implementation used
    # a greedy/over-flexible pattern containing `.*?`, which could consume the
    # actual left-hand expression in inputs such as:
    #   "Find x: 2x + 9 = 1(x + 1)"
    # Keep everything after the colon when a colon is present.
    if ":" in left:
        prefix, candidate = left.split(":", 1)
        if re.search(r"\b(?:find|solve|determine|calculate)\b", prefix, re.I):
            left = candidate.strip()
    else:
        # Strip only the natural-language command prefix. Keep the actual
        # equation intact, including forms such as `Solve 5(x + 4)`.
        left = re.sub(
            r"^(?:please\s+)?(?:find|solve|determine|calculate)\s+",
            "",
            left,
            flags=re.I,
        ).strip()
        left = re.sub(
            r"^(?:the\s+)?(?:value\s+of\s+)?x\s*(?=[:=])",
            "",
            left,
            flags=re.I,
        ).strip()


    # Natural-language questions often end the equation with punctuation.
    # Remove only trailing sentence punctuation; keep mathematical operators
    # and decimal points inside the expression intact.
    right = re.sub(r"[?.!]+\s*$", "", right).strip()
    left = re.sub(r"[?.!]+\s*$", "", left).strip()

    if not left or not right:
        return None

    return left, right


def _prepare_linear_expression(value: str) -> str:
    """
    Normalize a linear expression so SymPy can parse common
    mathematical notation such as:

        2x + 5
        9(x - 6)
        3(x + 4)
        (x + 2)(x - 3)
        2 (x + 5)
        4x(x + 1)

    The Find-X solver is intended for linear equations, so
    nonlinear expressions are rejected later by _linear_coefficients().
    """
    value = str(value).strip()

    # Normalize multiplication symbols.
    value = value.replace("×", "*")
    value = value.replace("·", "*")
    value = value.replace("−", "-")

    # Remove unnecessary whitespace around the expression.
    value = re.sub(r"\s+", " ", value)

    # ---------------------------------------------------------
    # Explicit multiplication for numbers followed by x
    # Example: 2x -> 2*x
    # ---------------------------------------------------------
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*[xX]\b",
        r"\1*x",
        value
    )

    # ---------------------------------------------------------
    # Explicit multiplication for numbers followed by (
    # Example: 9(x - 6) -> 9*(x - 6)
    # ---------------------------------------------------------
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*\(",
        r"\1*(",
        value
    )

    # ---------------------------------------------------------
    # Explicit multiplication after a closing parenthesis
    # Example: (x + 1)(x - 2) -> (x + 1)*(x - 2)
    # ---------------------------------------------------------
    value = re.sub(
        r"\)\s*\(",
        ")*(",
        value
    )

    # ---------------------------------------------------------
    # Explicit multiplication between ')' and x
    # Example: (x + 1)x -> (x + 1)*x
    # ---------------------------------------------------------
    value = re.sub(
        r"\)\s*[xX]\b",
        ")*x",
        value
    )

    # ---------------------------------------------------------
    # Explicit multiplication between x and '('
    # Example: x(x + 1) -> x*(x + 1)
    # ---------------------------------------------------------
    value = re.sub(
        r"\b[xX]\s*\(",
        "x*(",
        value
    )

    # ---------------------------------------------------------
    # Normalize X -> x
    # ---------------------------------------------------------
    value = re.sub(r"\bX\b", "x", value)

    # Remove spaces around operators.
    value = re.sub(r"\s*\*\s*", "*", value)
    value = re.sub(r"\s*\+\s*", "+", value)
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"\s*\^\s*", "^", value)

    return value.strip()


def _linear_coefficients(expression: str) -> Optional[Tuple[float, float]]:
    """Return (coefficient_of_x, constant) for a linear expression."""
    try:
        import sympy as sp

        x = sp.Symbol("x")
        expr = sp.sympify(_prepare_linear_expression(expression), locals={"x": x})
        if expr.has(x**2) or expr.has(x**3):
            return None
        poly = sp.Poly(sp.expand(expr), x)
        if poly.degree() > 1:
            return None
        a = float(poly.coeff_monomial(x))
        b = float(poly.coeff_monomial(1))
        return a, b
    except Exception:
        return None


def _validate_find_x_solution(left: str, right: str, x_value: float) -> bool:
    try:
        import sympy as sp
        x = sp.Symbol("x")
        lhs = sp.sympify(_prepare_linear_expression(left), locals={"x": x}).subs(x, x_value)
        rhs = sp.sympify(_prepare_linear_expression(right), locals={"x": x}).subs(x, x_value)
        return bool(sp.N(lhs - rhs) == 0 or abs(float(sp.N(lhs - rhs))) < 1e-8)
    except Exception:
        return False


def solve_symbolic_equation(text: str) -> Optional[Dict[str, Any]]:
    """Solve a safe univariate polynomial equation and validate every root."""
    equation = _extract_find_x_equation(text)
    if equation is None:
        return None

    left_text, right_text = equation
    left = _parse_safe_symbolic_expression(left_text)
    right = _parse_safe_symbolic_expression(right_text)
    if left is None or right is None:
        return None

    try:
        import sympy as sp

        difference = sp.expand(left - right)
        variable = _symbolic_variable(difference)
        if variable is None:
            return None

        polynomial = sp.Poly(difference, variable)
        degree = polynomial.degree()
        if degree < 1 or degree > 6:
            return None

        roots = sp.solve(sp.Eq(left, right), variable)
        if not isinstance(roots, list):
            return None

        validated = all(
            sp.simplify((left - right).subs(variable, root)) == 0
            for root in roots
        )
        factored = sp.factor(difference)

        if not roots:
            answer = "No solution"
            result_step = answer
        elif len(roots) == 1:
            answer = f"{variable} = {sp.sstr(roots[0])}"
            result_step = answer
        else:
            answer = ", ".join(
                f"{variable} = {sp.sstr(root)}"
                for root in roots
            )
            result_step = answer

        steps = [
            {"step": 1, "description": f"Equation: {sp.sstr(left)} = {sp.sstr(right)}"},
            {"step": 2, "description": f"Move all terms to one side: {sp.sstr(difference)} = 0"},
            {"step": 3, "description": f"Factor/solve: {sp.sstr(factored)} = 0"},
            {"step": 4, "description": f"Solutions: {result_step}", "result": result_step},
            {"step": 5, "description": f"Substitution validation: {'valid' if validated else 'could not validate'}"},
        ]
        return {
            "type": "solution",
            "subject": "find_x",
            "algorithm": "Safe Symbolic Equation Solver",
            "equation": f"{left_text} = {right_text}",
            "answer": answer,
            "confidence": 100.0,
            "solution_steps": steps,
            "validated": validated,
            "similarity_answer_copy": False,
        }
    except Exception:
        return None


def solve_find_x(
    text: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Solve a linear equation independently of the nearest dataset answer."""
    learning_context = dataset_learning or _get_dataset_method_learning(
        str(text or ""),
        subject="find_x",
        dataset_entries=dataset_entries,
    )
    if not text or "=" not in str(text):
        return None
    if not re.search(r"(?<![A-Za-z])x(?![A-Za-z])", str(text), re.I):
        return None

    equation = _extract_find_x_equation(text)
    if equation is None:
        return None

    left, right = equation
    left_coeff = _linear_coefficients(left)
    right_coeff = _linear_coefficients(right)

    if left_coeff is None or right_coeff is None:
        # The original linear route remains the fast path.  When the dataset
        # pattern says "equation" but the expression is quadratic or another
        # small polynomial, calculate it independently instead of returning an
        # empty no-match response.
        return solve_symbolic_equation(text)

    a, b = left_coeff
    c, d = right_coeff
    coefficient = a - c
    constant = d - b

    steps: List[Dict[str, Any]] = [
        {"step": 1, "description": f"Equation: {left} = {right}"},
        {"step": 2, "description": f"Collect x terms: ({_fmt_stat_number(a)})x − ({_fmt_stat_number(c)})x = {_fmt_stat_number(d)} − ({_fmt_stat_number(b)})"},
    ]

    if abs(coefficient) < 1e-12:
        if abs(constant) < 1e-12:
            steps.append({"step": 3, "description": "Both sides are identical, so infinitely many solutions exist."})
            return {
                "type": "solution", "subject": "find_x", "answer": "All real numbers",
                "confidence": 100.0, "solution_steps": steps,
                "dataset_assisted": bool(dataset_entries), "dataset_entries": len(dataset_entries or []),
                "similarity_answer_copy": False, "validated": True,
            }
        steps.append({"step": 3, "description": "The x coefficient is 0 but the constants are different, so there is no solution."})
        return {
            "type": "solution", "subject": "find_x", "answer": "No solution",
            "confidence": 100.0, "solution_steps": steps,
            "dataset_assisted": bool(dataset_entries), "dataset_entries": len(dataset_entries or []),
            "similarity_answer_copy": False, "validated": True,
        }

    x_value = constant / coefficient
    valid = _validate_find_x_solution(left, right, x_value)
    steps.extend([
        {"step": 3, "description": f"{_fmt_stat_number(coefficient)}x = {_fmt_stat_number(constant)}"},
        {"step": 4, "description": f"x = {_fmt_stat_number(constant)} / {_fmt_stat_number(coefficient)}"},
        {"step": 5, "description": f"x = {_format_number(x_value)}"},
        {"step": 6, "description": f"Validation: {'Valid' if valid else 'Could not validate'}"},
    ])

    return {
        "type": "solution",
        "subject": "find_x",
        "algorithm": "Dataset-Assisted Linear Equation Solver",
        "equation": f"{left} = {right}",
        "answer": _format_number(x_value),
        "x": x_value,
        "confidence": 100.0,
        "solution_steps": steps,
        "validated": valid,
        "dataset_assisted": bool(dataset_entries),
        "dataset_entries": len(dataset_entries or []),
        "dataset_learning": _dataset_learning_summary(dataset_entries, learning_context=learning_context, question=text, subject='find_x'),
        "similarity_answer_copy": False,
    }



# ════════════════════════════════════════════════════════════════════════════
# MATRIX SOLVER — USER INPUT FIRST, DATASET REFERENCE ONLY
# ════════════════════════════════════════════════════════════════════════════

def _matrix_extract_all(text: str) -> List[List[List[Any]]]:
    """Extract Python-style matrix literals from the user's actual question."""
    matrices = []
    for match in re.findall(r"\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])+\s*\]", str(text or "")):
        try:
            value = ast.literal_eval(match)
            if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
                if all(row and all(isinstance(x, (int, float, complex)) for x in row) for row in value):
                    widths = {len(row) for row in value}
                    if len(widths) == 1:
                        matrices.append(value)
        except Exception:
            continue
    return matrices


def _matrix_from_text(text: str, label: str) -> Optional[List[List[Any]]]:
    pattern = rf"\b{re.escape(label)}\s*=\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])+\s*\])"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
        if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
            if len({len(row) for row in value}) == 1 and all(row for row in value):
                return value
    except Exception:
        pass
    return None


def _matrix_to_text(matrix: Any) -> str:
    try:
        rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        return str(rows).replace("'", "")
    except Exception:
        return str(matrix)


def _matrix_clean_number(value: Any) -> Any:
    try:
        if hasattr(value, "is_Integer") and value.is_Integer:
            return int(value)
        if hasattr(value, "is_Rational") and value.q == 1:
            return int(value.p)
        if hasattr(value, "is_Float"):
            f = float(value)
            return int(f) if f.is_integer() else round(f, 10)
        return value
    except Exception:
        return value


def _matrix_answer(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [[_matrix_clean_number(x) for x in row] if isinstance(row, list) else _matrix_clean_number(row) for row in value]
    return _matrix_clean_number(value)


def solve_matrix(
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Dict[str, Any]:
    """Solve the user's matrix problem independently.

    Dataset entries may describe patterns/examples, but their matrices and answers
    are never substituted into the user's question.
    """
    text = str(question or "").strip()
    dataset_count = len(dataset_entries or [])
    base = {
        "subject": "matrix",
        "dataset_assisted": bool(dataset_entries),
        "dataset_entries": dataset_count,
        "similarity_answer_copy": False,
        "routing": "matrix_independent_solver",
        "dataset_learning_used": bool(
            isinstance(dataset_learning, dict)
            and dataset_learning.get("enabled")
        ),
        "learned_method": _learning_best_method(dataset_learning) or None,
    }
    try:
        import sympy as sp
    except ImportError:
        return {**base, "type": "error", "message": "SymPy is required for matrix solving. Install it with: pip install sympy"}

    matrices = _matrix_extract_all(text)
    A = _matrix_from_text(text, "A")
    B = _matrix_from_text(text, "B")
    if A is None and matrices:
        A = matrices[0]
    if B is None and len(matrices) > 1:
        B = matrices[1]

    low = normalize_solver_text(text).lower()
    nlp_operation = detect_operation_nlp(text)
    learning_context = _get_dataset_method_learning(
        text,
        subject="matrix",
        dataset_entries=dataset_entries,
        supplied_context=dataset_learning,
    )
    learned_matrix_method = _learning_best_method(learning_context)
    # Normalize OCR/operator tokens before keyword detection.
    # The matrix OCR can return forms such as `operation=+`, `A+B`, or `A + B`.
    normalized = re.sub(r"\s+", " ", low).strip()
    operation = None

    explicit_op = re.search(
        r"\boperation\s*=\s*(addition|add|subtraction|subtract|multiplication|multiply|[+\-×x*])",
        normalized,
    )
    if explicit_op:
        token = explicit_op.group(1)
        if token in {"addition", "add", "+"}:
            operation = "addition"
        elif token in {"subtraction", "subtract", "-"}:
            operation = "subtraction"
        elif token in {"multiplication", "multiply", "×", "x", "*"}:
            operation = "multiplication"

    # Handle compact matrix expressions produced by OCR, e.g. A+B.
    if operation is None:
        if re.search(r"\bA\s*\+\s*B\b", normalized):
            operation = "addition"
        elif re.search(r"\bA\s*[-−]\s*B\b", normalized):
            operation = "subtraction"
        elif re.search(r"\bA\s*[×x*]\s*B\b", normalized):
            operation = "multiplication"
    if any(k in low for k in ("eigenvector", "eigen vector")):
        operation = "eigenvectors"
    elif "eigenvalue" in low or "eigen value" in low:
        operation = "eigenvalues"
    elif "inverse" in low or "inv(" in low:
        operation = "inverse"
    elif "adjoint" in low or "adjugate" in low:
        operation = "adjoint"
    elif "rank" in low:
        operation = "rank"
    elif "cofactor" in low:
        operation = "cofactor"
    elif "minor" in low:
        operation = "minor"
    elif "skew-symmetric" in low or "skew symmetric" in low:
        operation = "skew_symmetric"
    elif "symmetric" in low:
        operation = "symmetric"
    elif "identity matrix" in low or low.strip() in {"identity", "find identity matrix"}:
        operation = "identity"
    elif "zero matrix" in low or low.strip() in {"zero matrix", "find zero matrix"}:
        operation = "zero"
    elif "diagonal matrix" in low or low.strip() in {"diagonal", "find diagonal matrix"}:
        operation = "diagonal"
    elif "trace" in low:
        operation = "trace"
    elif "transpose" in low:
        operation = "transpose"
    elif "determinant" in low or "det of matrix" in low or re.search(r"\bdet\b", low):
        operation = "determinant"
    elif "scalar" in low and ("multiply" in low or "multiplication" in low):
        operation = "scalar_multiplication"
    elif re.search(r"\bmultiply\s+(?:the\s+)?matrix\s+[A-Za-z]?\s*by\s+(?:a\s+)?scalar", low):
        operation = "scalar_multiplication"
    elif any(k in low for k in (
        "multiplication", "multiply", "multiply matrices", "multiply matrix",
        "product of matrices", "matrix product"
    )):
        operation = "multiplication"
    elif any(k in low for k in ("subtraction", "subtract", "difference of")):
        operation = "subtraction"
    elif any(k in low for k in ("addition", "add", "sum of", "sum")):
        operation = "addition"

    if nlp_operation is not None:
        operation = nlp_operation

    if operation in {"addition", "subtraction", "multiplication"} and (A is None or B is None):
        return {**base, "type": "needs_input", "message": "Please provide both matrices A and B. Example: A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]."}
    if operation in {"transpose", "determinant", "trace", "inverse", "adjoint", "rank", "eigenvalues", "eigenvectors", "symmetric", "skew_symmetric", "minor", "cofactor"} and A is None:
        return {**base, "type": "needs_input", "message": "Please provide the matrix. Example: A = [[1, 2], [3, 4]]."}
    if (
        operation in {"addition", "subtraction"}
        and A is not None
        and B is not None
        and (
            len(A) != len(B)
            or any(len(row_a) != len(row_b) for row_a, row_b in zip(A, B))
        )
    ):
        return {**base, "type": "error", "message": "Matrix addition/subtraction requires matrices of the same order."}

    try:
        mA = sp.Matrix(A) if A is not None else None
        mB = sp.Matrix(B) if B is not None else None
        steps = []
        answer = None
        formula = None
        if operation == "addition":
            answer = mA + mB
            formula = "R[i][j] = A[i][j] + B[i][j]"
            steps = ["Given A and B.", "Add corresponding entries: R[i][j] = A[i][j] + B[i][j].", f"R = {_matrix_to_text(answer)}"]
        elif operation == "subtraction":
            answer = mA - mB
            formula = "R[i][j] = A[i][j] − B[i][j]"
            steps = ["Given A and B.", "Subtract corresponding entries: R[i][j] = A[i][j] − B[i][j].", f"R = {_matrix_to_text(answer)}"]
        elif operation == "multiplication":
            if mA.cols != mB.rows:
                return {**base, "type": "error", "message": f"Matrix multiplication is not possible: A is {mA.rows}×{mA.cols} and B is {mB.rows}×{mB.cols}."}
            answer = mA * mB
            formula = "R[i][j] = Σ A[i][k]B[k][j]"
            steps = ["Given A and B.", "Multiply rows of A by columns of B.", f"R = {_matrix_to_text(answer)}"]
        elif operation == "scalar_multiplication":
            nums = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)
            if A is None or not nums:
                return {**base, "type": "needs_input", "message": "Please provide a matrix and scalar. Example: Multiply A = [[1, 2], [3, 4]] by 3."}
            scalar = sp.sympify(nums[-1])
            answer = scalar * mA
            formula = "R[i][j] = kA[i][j]"
            steps = [f"Given A and scalar k = {scalar}.", "Multiply every matrix entry by k.", f"R = {_matrix_to_text(answer)}"]
        elif operation == "transpose":
            answer = mA.T
            formula = "(Aᵀ)[i][j] = A[j][i]"
            steps = [f"Given A = {_matrix_to_text(mA)}.", "Interchange rows and columns.", f"Aᵀ = {_matrix_to_text(answer)}"]
        elif operation == "determinant":
            if mA.rows != mA.cols:
                return {**base, "type": "error", "message": "A determinant is defined only for a square matrix."}
            answer = mA.det()
            formula = "det(A)"
            steps = [f"Given A = {_matrix_to_text(mA)}.", "Compute the determinant of the square matrix.", f"det(A) = {_matrix_clean_number(answer)}"]
        elif operation == "trace":
            if mA.rows != mA.cols:
                return {**base, "type": "error", "message": "Trace is defined only for a square matrix."}
            answer = mA.trace()
            steps = [f"Given A = {_matrix_to_text(mA)}.", "Add the main-diagonal elements.", f"tr(A) = {_matrix_clean_number(answer)}"]
        elif operation == "inverse":
            if mA.rows != mA.cols:
                return {**base, "type": "error", "message": "A matrix must be square to have an inverse."}
            if mA.det() == 0:
                return {**base, "type": "error", "message": "The matrix is singular, so its inverse does not exist."}
            answer = mA.inv()
            steps = [f"Given A = {_matrix_to_text(mA)}.", "Compute A⁻¹ using exact matrix arithmetic.", f"A⁻¹ = {_matrix_to_text(answer)}"]
        elif operation == "adjoint":
            if mA.rows != mA.cols:
                return {**base, "type": "error", "message": "Adjoint/adjugate requires a square matrix."}
            answer = mA.adjugate()
            steps = [f"Given A = {_matrix_to_text(mA)}.", "Form the cofactor matrix and transpose it.", f"adj(A) = {_matrix_to_text(answer)}"]
        elif operation == "rank":
            answer = mA.rank()
            steps = [f"Given A = {_matrix_to_text(mA)}.", "Reduce the matrix and count the number of linearly independent rows/columns.", f"rank(A) = {answer}"]
        elif operation in {"symmetric", "skew_symmetric"}:
            if mA.rows != mA.cols:
                return {**base, "type": "error", "message": "Symmetric and skew-symmetric matrices must be square."}
            if operation == "symmetric":
                answer = bool(mA == mA.T)
                steps = [f"Given A = {_matrix_to_text(mA)}.", "Check whether A = Aᵀ.", f"A is {'symmetric' if answer else 'not symmetric'}. "]
            else:
                answer = bool(mA == -mA.T)
                steps = [f"Given A = {_matrix_to_text(mA)}.", "Check whether A = −Aᵀ.", f"A is {'skew-symmetric' if answer else 'not skew-symmetric'}. "]
        elif operation in {"minor", "cofactor"}:
            pos = re.search(r"a\s*([1-9])\s*([1-9])", low)
            if not pos or mA.rows < 2 or mA.cols < 2:
                return {**base, "type": "needs_input", "message": "Please specify the element, such as a12, together with a square matrix."}
            i, j = int(pos.group(1)), int(pos.group(2))
            if i > mA.rows or j > mA.cols:
                return {**base, "type": "error", "message": f"Element a{i}{j} is outside the matrix."}
            minor = mA.minor_submatrix(i-1, j-1).det()
            answer = minor if operation == "minor" else ((-1) ** (i+j)) * minor
            steps = [f"Given A and element a{i}{j}.", "Delete row i and column j and take the determinant of the remaining matrix.", f"M{i}{j} = {_matrix_clean_number(minor)}"]
            if operation == "cofactor":
                steps.append(f"C{i}{j} = (-1)^({i}+{j}) M{i}{j} = {_matrix_clean_number(answer)}")
        elif operation in {"eigenvalues", "eigenvectors"}:
            if mA.rows != mA.cols:
                return {**base, "type": "error", "message": "Eigenvalues and eigenvectors require a square matrix."}
            if operation == "eigenvalues":
                answer = {str(k): int(v) if getattr(v, 'is_Integer', False) else str(v) for k, v in mA.eigenvals().items()}
                steps = [f"Given A = {_matrix_to_text(mA)}.", "Solve det(A − λI) = 0.", f"Eigenvalues = {answer}"]
            else:
                answer = {str(k): [_matrix_answer(v) for v in vecs] for k, vecs in mA.eigenvects()}
                steps = [f"Given A = {_matrix_to_text(mA)}.", "Solve (A − λI)v = 0 for each eigenvalue.", f"Eigenvectors = {answer}"]
        elif operation in {"identity", "zero", "diagonal"}:
            if not matrices:
                return {**base, "type": "needs_input", "message": "Please provide the matrix or its order. Example: 3×3 identity matrix."}
            m = mA
            if operation == "identity": answer = sp.eye(m.rows) if m.rows == m.cols else None
            elif operation == "zero": answer = sp.zeros(m.rows, m.cols)
            else: answer = sp.diag(*m.diagonal())
            if answer is None:
                return {**base, "type": "error", "message": "Identity matrix must be square."}
            steps = [f"Given A = {_matrix_to_text(mA)}.", f"Construct the {operation} matrix from the requested order/data.", f"Result = {_matrix_to_text(answer)}"]
        else:
            return {**base, "type": "needs_input", "message": "Please specify the matrix operation (for example addition, subtraction, multiplication, transpose, determinant, inverse, rank, trace, minor, or cofactor) and provide the required matrix data."}

        return {
            **base,
            "type": "solution",
            "algorithm": "Independent Matrix Solver",
            "operation": operation,
            "formula": formula,
            "answer": _matrix_answer(answer),
            "confidence": 100.0,
            "solution_steps": [{"step": i + 1, "description": step} for i, step in enumerate(steps)],
            "validated": True,
        }
    except Exception as exc:
        logger.exception("Matrix solver failed: %s", exc)
        return {**base, "type": "error", "message": f"Could not solve the matrix question: {exc}"}


# ════════════════════════════════════════════════════════════════════════════
# TRIGONOMETRY SOLVER — USER INPUT FIRST, DATASET REFERENCE ONLY
# ════════════════════════════════════════════════════════════════════════════

_TRIG_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sec": lambda x: 1.0 / math.cos(x),
    "csc": lambda x: 1.0 / math.sin(x),
    "cot": lambda x: 1.0 / math.tan(x),
}


def _trig_clean_number(value: float) -> str:
    """Format trigonometric numerical results cleanly."""
    value = float(value)

    if abs(value) < 1e-12:
        value = 0.0

    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))

    return f"{value:.10f}".rstrip("0").rstrip(".")


def _trig_exact_value(function_name: str, angle: float) -> Optional[str]:
    """Return exact values for standard angles where possible."""

    normalized = angle % 360

    exact = {
        ("sin", 0): "0",
        ("sin", 30): "1/2",
        ("sin", 45): "√2/2",
        ("sin", 60): "√3/2",
        ("sin", 90): "1",
        ("sin", 120): "√3/2",
        ("sin", 135): "√2/2",
        ("sin", 150): "1/2",
        ("sin", 180): "0",
        ("sin", 210): "-1/2",
        ("sin", 225): "-√2/2",
        ("sin", 240): "-√3/2",
        ("sin", 270): "-1",
        ("sin", 300): "-√3/2",
        ("sin", 315): "-√2/2",
        ("sin", 330): "-1/2",
        ("sin", 360): "0",

        ("cos", 0): "1",
        ("cos", 30): "√3/2",
        ("cos", 45): "√2/2",
        ("cos", 60): "1/2",
        ("cos", 90): "0",
        ("cos", 120): "-1/2",
        ("cos", 135): "-√2/2",
        ("cos", 150): "-√3/2",
        ("cos", 180): "-1",
        ("cos", 210): "-√3/2",
        ("cos", 225): "-√2/2",
        ("cos", 240): "-1/2",
        ("cos", 270): "0",
        ("cos", 300): "1/2",
        ("cos", 315): "√2/2",
        ("cos", 330): "√3/2",
        ("cos", 360): "1",

        ("tan", 0): "0",
        ("tan", 30): "√3/3",
        ("tan", 45): "1",
        ("tan", 60): "√3",
        ("tan", 180): "0",
        ("tan", 210): "√3/3",
        ("tan", 225): "1",
        ("tan", 240): "√3",
        ("tan", 360): "0",
    }

    # Floating-point angles such as 45.0 should still match.
    for (func, standard_angle), value in exact.items():
        if func == function_name and abs(normalized - standard_angle) < 1e-10:
            return value

    return None


def _extract_trig_expression(question: str) -> Optional[Tuple[str, float]]:
    """
    Extract a trigonometric function and angle from natural language.

    Examples:
        sin 37°
        sin(37°)
        Find sin 37 degrees
        calculate cos(123 degrees)
        tan 45 deg
    """

    if not question:
        return None

    text = str(question).strip().lower()

    # Preserve the degree information while normalising common spellings.
    text = text.replace("°", " degrees ")
    text = re.sub(r"\bdegrees?\b", " degrees ", text)
    text = re.sub(r"\bdeg\b", " degrees ", text)
    text = re.sub(r"\s+", " ", text)

    function_pattern = (
        r"\b(sin|sine|cos|cosine|tan|tangent|"
        r"sec|secant|csc|cosecant|cot|cotangent)\b"
    )

    function_match = re.search(function_pattern, text, re.I)

    if not function_match:
        return None

    raw_function = function_match.group(1).lower()

    function_map = {
        "sin": "sin",
        "sine": "sin",
        "cos": "cos",
        "cosine": "cos",
        "tan": "tan",
        "tangent": "tan",
        "sec": "sec",
        "secant": "sec",
        "csc": "csc",
        "cosecant": "csc",
        "cot": "cot",
        "cotangent": "cot",
    }

    function_name = function_map.get(raw_function)

    if function_name is None:
        return None

    remaining = text[function_match.end():]

    angle_match = re.search(
        r"(?P<angle>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        remaining
    )

    if not angle_match:
        return None

    try:
        angle = float(angle_match.group("angle"))
    except Exception:
        return None

    return function_name, angle



def _trig_response(
    question: str,
    answer: str,
    steps: List[Dict[str, Any]],
    dataset_entries: Optional[List[dict]] = None,
    algorithm: str = "Independent Trigonometry Solver",
    result: Any = None,
) -> Dict[str, Any]:
    """Build a validated trig response without copying a dataset answer."""
    return {
        "type": "solution",
        "subject": "trigonometry",
        "algorithm": algorithm,
        "answer": answer,
        "result": result if result is not None else answer,
        "confidence": 100.0,
        "solution_steps": steps,
        "dataset_assisted": bool(dataset_entries),
        "dataset_entries": len(dataset_entries or []),
        "dataset_learning": _dataset_learning_summary(dataset_entries),
        "similarity_answer_copy": False,
        "validated": True,
    }


def _trig_given_ratio_expression_solver(question: str, dataset_entries=None):
    """Evaluate an expression from a given trig ratio and quadrant condition.

    This route is intentionally before equation solving.  A prompt such as
    "sin θ = 3/5, θ acute; find (1-cos θ)/(1+cos θ)" is an expression-value
    problem, not a request to list all angles satisfying sin θ = 3/5.
    """
    text = str(question or "")
    if not text:
        return None

    normalized = (
        text.lower().replace("θ", "theta").replace("−", "-")
        .replace("×", "*").replace("÷", "/").replace("²", "^2")
    )
    normalized = re.sub(r"\s+", " ", normalized)

    given = re.search(
        r"\b(sin|cos|tan)\s*(?:\(\s*theta\s*\)|theta\b)\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
        r"(?:/\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?",
        normalized,
        re.I,
    )
    target = re.search(
        r"\b(?:find|evaluate|calculate|determine)\b\s*"
        # Accept both "find the value of ..." and direct forms such as
        # "evaluate (1 - cos(theta)) / (1 + cos(theta))".  The strict
        # expression validation below prevents a prose fragment from being
        # treated as a mathematical target.
        r"(?:(?:the\s+)?(?:value|result)\s*(?:of\s*)?[:=]?\s*|(?:of\s+)?)"
        r"(.+?)\s*$",
        normalized,
        re.I | re.S,
    )
    if not given or not target:
        return None

    try:
        import sympy as sp

        function_name = given.group(1).lower()
        numerator = sp.Rational(given.group(2))
        denominator = sp.Rational(given.group(3)) if given.group(3) else sp.Integer(1)
        if denominator == 0:
            return None
        known_value = sp.simplify(numerator / denominator)
    except Exception:
        return None

    is_acute = bool(re.search(r"\bacute(?:\s+angle)?\b", normalized))
    if function_name in {"sin", "cos"} and abs(float(known_value)) > 1:
        return None

    values = {function_name: known_value}
    derived_steps = [
        {
            "step": 1,
            "description": f"Given {function_name}(θ) = {sp.sstr(known_value)}.",
        }
    ]

    # Derive the remaining primary ratios only when the acute-angle condition
    # fixes their signs.  Without a quadrant, e.g. sin θ = 3/5 does not give a
    # unique cos θ, so returning a numeric answer would be wrong.
    if is_acute:
        try:
            if function_name == "sin":
                values["cos"] = sp.simplify(sp.sqrt(1 - known_value ** 2))
                values["tan"] = sp.simplify(known_value / values["cos"])
                derived_steps.extend([
                    {"step": 2, "formula": "sin²θ + cos²θ = 1"},
                    {
                        "step": 3,
                        "description": "θ is acute, so cos(θ) is positive.",
                        "calculation": (
                            f"cos(θ) = √(1 - ({sp.sstr(known_value)})²) "
                            f"= {sp.sstr(values['cos'])}"
                        ),
                    },
                ])
            elif function_name == "cos":
                values["sin"] = sp.simplify(sp.sqrt(1 - known_value ** 2))
                values["tan"] = sp.simplify(values["sin"] / known_value)
                derived_steps.extend([
                    {"step": 2, "formula": "sin²θ + cos²θ = 1"},
                    {
                        "step": 3,
                        "description": "θ is acute, so sin(θ) is positive.",
                        "calculation": (
                            f"sin(θ) = √(1 - ({sp.sstr(known_value)})²) "
                            f"= {sp.sstr(values['sin'])}"
                        ),
                    },
                ])
            elif function_name == "tan":
                values["cos"] = sp.simplify(1 / sp.sqrt(1 + known_value ** 2))
                values["sin"] = sp.simplify(known_value * values["cos"])
                derived_steps.extend([
                    {"step": 2, "formula": "1 + tan²θ = sec²θ"},
                    {
                        "step": 3,
                        "description": "θ is acute, so sin(θ) and cos(θ) are positive.",
                    },
                ])
        except Exception:
            return None

    expression_text = target.group(1).strip().rstrip("?.!")
    expression_text = expression_text.replace("^", "**")
    replacement_map = {
        "sin": "sin_theta", "cos": "cos_theta", "tan": "tan_theta",
        "sec": "sec_theta", "csc": "csc_theta", "cot": "cot_theta",
    }
    for name, replacement in replacement_map.items():
        # Match only the trig reference itself: ``cos θ`` or ``cos(θ)``.
        # Do not make the closing parenthesis optional independently of an
        # opening parenthesis.  In ``(1 - cos θ) / (1 + cos θ)``, the older
        # pattern consumed the grouping ``)`` after each ``cos θ`` and left
        # an unbalanced expression for SymPy to parse.  That caused this
        # solver to fail and the later equation solver to return angle
        # solutions instead of the requested expression value.
        expression_text = re.sub(
            rf"\b{name}\s*(?:\(\s*theta\s*\)|theta\b)",
            replacement,
            expression_text,
            flags=re.I,
        )

    # The target is an arithmetic expression after trig terms have been
    # replaced. Reject prose/unknown identifiers before SymPy parses it.
    if not re.fullmatch(r"[0-9A-Za-z_+*/().\s-]+", expression_text):
        return None
    identifiers = set(re.findall(r"[A-Za-z_]+", expression_text))
    if not identifiers.issubset(set(replacement_map.values())):
        return None

    local_values = {
        "sin_theta": values.get("sin"),
        "cos_theta": values.get("cos"),
        "tan_theta": values.get("tan"),
    }
    if values.get("cos") not in (None, 0):
        local_values["sec_theta"] = sp.simplify(1 / values["cos"])
    if values.get("sin") not in (None, 0):
        local_values["csc_theta"] = sp.simplify(1 / values["sin"])
    if values.get("tan") not in (None, 0):
        local_values["cot_theta"] = sp.simplify(1 / values["tan"])

    if any(local_values.get(identifier) is None for identifier in identifiers):
        return {
            "type": "needs_input",
            "subject": "trigonometry",
            "message": (
                "The given ratio does not determine the required expression "
                "uniquely. Please state the quadrant or whether θ is acute."
            ),
            "similarity_answer_copy": False,
        }

    try:
        expression = sp.sympify(expression_text, locals=local_values)
        answer_value = sp.simplify(expression)
        if answer_value in (sp.zoo, sp.oo, -sp.oo, sp.nan):
            return None
        answer = sp.sstr(answer_value)
    except Exception:
        return None

    substitution = expression_text
    for name, value in local_values.items():
        if value is not None:
            substitution = substitution.replace(name, f"({sp.sstr(value)})")

    derived_steps.extend([
        {
            "step": len(derived_steps) + 1,
            "description": "Substitute the determined trigonometric value into the requested expression.",
            "calculation": f"{substitution} = {answer}",
        },
        {
            "step": len(derived_steps) + 2,
            "description": f"Final Answer: {answer}",
            "result": answer,
        },
        {
            "step": len(derived_steps) + 3,
            "description": "Validation: the result was evaluated exactly from the given ratio and acute-angle sign.",
        },
    ])
    return _trig_response(
        question,
        answer,
        derived_steps,
        dataset_entries,
        "Independent Given-Ratio Trigonometry Solver",
        answer_value,
    )


def _degree_radian_solver(question: str, dataset_entries=None):
    text = str(question).strip().lower().replace("π", "pi")

    # Parse a radian angle as a complete mathematical expression, rather than
    # taking only the first numeric token. This is essential for values such as
    # 5π/6, 3π/4, π/2 and -7π/3.
    value = None
    radian_expr = re.search(
        r"(?<![a-z0-9_])([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)\s*(?:\*\s*)?pi\s*(?:/\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?",
        text,
    )
    if radian_expr:
        coefficient = radian_expr.group(1)
        coefficient = -1.0 if coefficient == "-" else (1.0 if coefficient in ("", "+") else float(coefficient))
        denominator = float(radian_expr.group(2)) if radian_expr.group(2) else 1.0
        if denominator != 0:
            value = coefficient * math.pi / denominator

    # Plain numeric radians, e.g. 2 radians. Do this only when no pi expression
    # was found, so 5π/6 is never misread as the number 5.
    if value is None:
        nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if not nums:
            return None
        value = float(nums[0])

    # Degree -> radian conversion: require the requested target to be radians.
    if re.search(r"\b(?:degrees?|deg)\b|°", text) and re.search(r"\bto\s+radians?\b|\binto\s+radians?\b", text):
        radians = math.radians(value)
        try:
            from fractions import Fraction
            # Convert the displayed decimal degree value to a rational using
            # its decimal text, then reduce degrees/180 exactly.
            degree_text = format(value, ".12f").rstrip("0").rstrip(".")
            degree_fraction = Fraction(degree_text)
            frac = (degree_fraction / 180).limit_denominator(3600)
            if frac == 0:
                exact = "0"
            elif frac == 1:
                exact = "π"
            elif frac.denominator == 1:
                exact = f"{frac.numerator}π"
            elif frac.numerator == 1:
                exact = f"π/{frac.denominator}"
            elif frac.numerator == -1:
                exact = f"-π/{frac.denominator}"
            else:
                exact = f"{frac.numerator}π/{frac.denominator}"
        except Exception:
            exact = f"{_format_number(value)}π/180"
        return _trig_response(
            question, exact,
            [
                {"step": 1, "description": f"Given angle = {value:g}°."},
                {"step": 2, "formula": "radians = degrees × π / 180",
                 "calculation": f"{value:g} × π / 180 = {exact}"},
                {"step": 3, "description": f"Final Answer: {exact}", "result": exact},
            ],
            dataset_entries, "Independent Degree-to-Radian Solver", radians,
        )

    # Radian -> degree conversion: require the requested target to be degrees.
    if re.search(r"\bradians?\b", text) and re.search(r"\bto\s+degrees?\b|\binto\s+degrees?\b", text):
        degrees_value = math.degrees(value)
        answer = f"{_trig_clean_number(degrees_value)}°"
        return _trig_response(
            question, answer,
            [
                {"step": 1, "description": f"Given angle = {value / math.pi:g}π radians." if radian_expr else f"Given angle = {value:g} radians."},
                {"step": 2, "formula": "degrees = radians × 180 / π",
                 "calculation": f"{value / math.pi:g}π × 180 / π = {degrees_value:.10f}°" if radian_expr else f"{value:g} × 180 / π = {degrees_value:.10f}°"},
                {"step": 3, "description": f"Final Answer: {answer}", "result": answer},
            ],
            dataset_entries, "Independent Radian-to-Degree Solver", degrees_value,
        )
    return None


def _law_of_cosines_solver(question: str, dataset_entries=None):
    text = str(question).lower()
    # Support natural-language triangle questions even when the user does
    # not explicitly say "Law of Cosines".
    explicit_cosine_rule = any(
        x in text for x in ("law of cosines", "law of cosine", "cosine rule", "cos rule")
    )
    natural_triangle_form = bool(
        re.search(r"\btriangle\b", text)
        and re.search(r"\bsides?\s+[-+]?\d", text)
        and re.search(r"\bincluded\s+angle\b", text)
    )
    if not (explicit_cosine_rule or natural_triangle_form):
        return None

    angle_match = re.search(
        r"(?:included\s+angle|angle)\s*(?:is|=|of|:)?\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:°|degrees?|deg)?",
        text, re.I
    )
    if not angle_match:
        return None

    angle = float(angle_match.group(1))
    angle_start = angle_match.start(1)
    before = text[:angle_start]
    side_nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", before)

    # Prefer explicitly labelled a and b.
    labelled = {}
    for label in ("a", "b"):
        m = re.search(rf"\b{label}\s*=\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if m:
            labelled[label] = float(m.group(1))

    if len(labelled) == 2:
        a, b = labelled["a"], labelled["b"]
    elif len(side_nums) >= 2:
        a, b = float(side_nums[-2]), float(side_nums[-1])
    else:
        return None

    if a <= 0 or b <= 0 or not (0 < angle < 180):
        return None

    c2 = a*a + b*b - 2*a*b*math.cos(math.radians(angle))
    if c2 < -1e-10:
        return None
    c = math.sqrt(max(0.0, c2))
    answer = _trig_clean_number(c)

    return _trig_response(
        question, answer,
        [
            {"step": 1, "description": f"Given a = {a:g}, b = {b:g}, included angle C = {angle:g}°."},
            {"step": 2, "formula": "c² = a² + b² − 2ab cos(C)"},
            {"step": 3, "calculation": f"c² = {a:g}² + {b:g}² − 2({a:g})({b:g})cos({angle:g}°) = {c2:.10f}"},
            {"step": 4, "calculation": f"c = √{c2:.10f} = {answer}"},
            {"step": 5, "description": f"Final Answer: {answer}", "result": answer},
        ],
        dataset_entries, "Independent Law of Cosines Solver", c,
    )


def _law_of_sines_solver(question: str, dataset_entries=None):
    text = str(question).lower()
    if "law of sines" not in text and "sine rule" not in text:
        return None

    sides = {}
    angles = {}
    for side in ("a", "b", "c"):
        m = re.search(rf"\b{side}\s*=\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if m:
            sides[side] = float(m.group(1))
    for angle_name in ("A", "B", "C"):
        m = re.search(
            rf"\b{angle_name}\s*=\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:°|degrees?|deg)?",
            question, re.I
        )
        if m:
            angles[angle_name] = float(m.group(1))

    known = [
        (s, a) for s, a in (("a", "A"), ("b", "B"), ("c", "C"))
        if s in sides and a in angles and 0 < angles[a] < 180
    ]
    if not known:
        return None

    s0, a0 = known[0]
    ratio = sides[s0] / math.sin(math.radians(angles[a0]))

    for side, angle_name in (("a", "A"), ("b", "B"), ("c", "C")):
        if side in sides and angle_name not in angles:
            q = sides[side] / ratio
            if -1 <= q <= 1:
                angles[angle_name] = math.degrees(math.asin(q))
        elif angle_name in angles and side not in sides:
            sides[side] = ratio * math.sin(math.radians(angles[angle_name]))

    parts = []
    for side, angle_name in (("a", "A"), ("b", "B"), ("c", "C")):
        if side in sides:
            parts.append(f"{side}={_trig_clean_number(sides[side])}")
        if angle_name in angles:
            parts.append(f"{angle_name}={_trig_clean_number(angles[angle_name])}°")

    if len(parts) < 2:
        return None

    answer = ", ".join(parts)
    return _trig_response(
        question, answer,
        [
            {"step": 1, "description": "Identify a known side-angle pair."},
            {"step": 2, "formula": "a/sin A = b/sin B = c/sin C"},
            {"step": 3, "calculation": f"Apply the sine rule using the supplied values: {answer}."},
            {"step": 4, "description": f"Final Answer: {answer}", "result": answer},
        ],
        dataset_entries, "Independent Law of Sines Solver",
    )


def _inverse_trig_solver(question: str, dataset_entries=None):
    """Solve inverse trig expressions, including common radical fractions."""
    text = str(question or "").lower().replace("√", "sqrt")
    func_match = re.search(
        r"\b(arcsin|sin\s*\^-\s*1|inverse\s+sine|"
        r"arccos|cos\s*\^-\s*1|inverse\s+cosine|"
        r"arctan|tan\s*\^-\s*1|inverse\s+tangent)\b",
        text,
        re.I,
    )
    if not func_match:
        return None

    raw_func = re.sub(r"\s+", "", func_match.group(1).lower())
    remaining = text[func_match.end():].lstrip()
    if remaining.startswith("("):
        remaining = remaining[1:].lstrip()

    operand_match = re.match(
        r"(?P<radical>[+-]?sqrt\s*\(\s*(?P<radicand>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\))"
        r"(?:\s*/\s*(?P<radden>[-+]?(?:\d+(?:\.\d*)?|\.\d+)))?"
        r"|"
        r"(?P<radical_plain>[+-]?sqrt\s*(?P<radicand_plain>[-+]?(?:\d+(?:\.\d*)?|\.\d+)))"
        r"(?:\s*/\s*(?P<radden_plain>[-+]?(?:\d+(?:\.\d*)?|\.\d+)))?"
        r"|"
        r"(?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+))?)",
        remaining,
        re.I,
    )
    if not operand_match:
        return None

    try:
        if operand_match.group("radical"):
            radicand = float(operand_match.group("radicand"))
            denominator = float(operand_match.group("radden")) if operand_match.group("radden") else 1.0
            raw_display = f"√{_trig_clean_number(radicand)}" + (f"/{_trig_clean_number(denominator)}" if denominator != 1 else "")
            value = math.sqrt(radicand) / denominator
        elif operand_match.group("radical_plain"):
            radicand = float(operand_match.group("radicand_plain"))
            denominator = float(operand_match.group("radden_plain")) if operand_match.group("radden_plain") else 1.0
            raw_display = f"√{_trig_clean_number(radicand)}" + (f"/{_trig_clean_number(denominator)}" if denominator != 1 else "")
            value = math.sqrt(radicand) / denominator
        else:
            raw_numeric = operand_match.group("num").replace(" ", "")
            value = (
                float(raw_numeric.split("/")[0]) / float(raw_numeric.split("/")[1])
                if "/" in raw_numeric else float(raw_numeric)
            )
            raw_display = raw_numeric
    except Exception:
        return None

    if "cos" in raw_func:
        fn, name = math.acos, "arccos"
    elif "tan" in raw_func:
        fn, name = math.atan, "arctan"
    else:
        fn, name = math.asin, "arcsin"

    if (name in ("arcsin", "arccos")) and not -1 <= value <= 1:
        return None

    try:
        radians = fn(value)
    except (ValueError, OverflowError):
        return None

    degrees_value = math.degrees(radians)
    answer = f"{_trig_clean_number(degrees_value)}°"
    return _trig_response(
        question, answer,
        [
            {"step": 1, "description": f"Evaluate {name}({raw_display})."},
            {"step": 2, "formula": f"θ = {name}({raw_display})"},
            {"step": 3, "calculation": f"θ = {radians:.10f} radians = {degrees_value:.10f}°"},
            {"step": 4, "description": f"Final Answer: {answer}", "result": answer},
        ],
        dataset_entries, "Independent Inverse Trigonometry Solver", degrees_value,
    )


def _trig_polynomial_equation_solver(question: str, dataset_entries=None):
    """Solve simple quadratic trig equations such as a*sin²θ+b*sinθ+c=0."""
    text = str(question).lower().replace(" ", "").replace("−", "-").replace("–", "-")
    # Accept sin/cos with Unicode superscript ² or ^2, and require =0.
    for fn in ("sin", "cos"):
        pattern = rf"([+-]?\d*\.?\d*)\*?{fn}(?:θ|x)?²(?:θ|x)?([+-]\d*\.?\d*)\*?{fn}(?:θ|x)?([+-]\d*\.?\d*)=0"
        m = re.search(pattern, text)
        if not m:
            continue
        try:
            a = float(m.group(1) or ("-1" if m.group(1) == "-" else "1"))
            b = float(m.group(2))
            c = float(m.group(3) or 0.0)
        except Exception:
            continue
        if abs(a) < 1e-15:
            continue
        disc = b*b - 4*a*c
        if disc < -1e-12:
            return _trig_response(question, "No real solutions", [
                {"step": 1, "description": f"Let u = {fn}(θ), giving a quadratic equation."},
                {"step": 2, "calculation": f"Discriminant = {disc:g} < 0."},
                {"step": 3, "description": "No real trigonometric solutions exist."},
                {"step": 4, "description": "Final Answer: No real solutions", "result": "No real solutions"},
            ], dataset_entries, "Independent Trigonometric Equation Solver")
        roots = [(-b + math.sqrt(max(0.0, disc)))/(2*a), (-b - math.sqrt(max(0.0, disc)))/(2*a)]
        roots = sorted({round(r, 12) for r in roots if -1 - 1e-12 <= r <= 1 + 1e-12})
        if not roots:
            return _trig_response(question, "No solutions in the real domain", [
                {"step": 1, "description": f"Let u = {fn}(θ)."},
                {"step": 2, "description": "The quadratic roots are outside the valid range for the selected trigonometric function."},
                {"step": 3, "description": "Final Answer: No solutions in the real domain", "result": "No solutions in the real domain"},
            ], dataset_entries, "Independent Trigonometric Equation Solver")

        angles = []
        for value in roots:
            if abs(value) > 1 + 1e-12:
                continue
            principal = math.degrees(math.asin(value) if fn == "sin" else math.acos(value))
            if fn == "sin":
                candidates = [principal, 180 - principal]
            else:
                candidates = [principal, 360 - principal]
            for angle in candidates:
                angle %= 360
                if abs(angle - 360) < 1e-10:
                    angle = 0.0
                angles.append(round(angle, 10))
        solutions = sorted(set(angles))
        answer = ", ".join(f"{_trig_clean_number(x)}°" for x in solutions)
        return _trig_response(question, answer, [
            {"step": 1, "description": f"Let u = {fn}(θ). Then solve {a:g}u² {b:+g}u {c:+g} = 0."},
            {"step": 2, "calculation": "Solve the quadratic equation for u."},
            {"step": 3, "calculation": f"Valid {fn}(θ) values = {', '.join(_trig_clean_number(r) for r in roots)}."},
            {"step": 4, "description": f"Find all angles from 0° to 360° for those {fn} values."},
            {"step": 5, "description": f"Final Answer: {answer}", "result": answer},
        ], dataset_entries, "Independent Trigonometric Equation Solver")
    return None


def _trig_equation_solver(question: str, dataset_entries=None):
    text = str(question).lower()
    m = re.search(
        r"\b(sin|cos|tan)\s*(?:\(?\s*(?:θ|x)\s*\)?)?\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+))?)",
        text
    )
    if not m:
        return None

    fn, raw = m.group(1), m.group(2).replace(" ", "")
    try:
        value = float(raw.split("/")[0]) / float(raw.split("/")[1]) if "/" in raw else float(raw)
    except Exception:
        return None

    if fn in ("sin", "cos") and not -1 <= value <= 1:
        return None

    principal = math.degrees({"sin": math.asin, "cos": math.acos, "tan": math.atan}[fn](value))
    if fn == "sin":
        candidates = [principal, 180 - principal]
    elif fn == "cos":
        candidates = [principal, 360 - principal]
    else:
        candidates = [principal, principal + 180]

    solutions = sorted({round(x % 360, 10) for x in candidates})
    answer = ", ".join(f"{_trig_clean_number(x)}°" for x in solutions)

    return _trig_response(
        question, answer,
        [
            {"step": 1, "description": f"Given {fn}(θ) = {raw}."},
            {"step": 2, "description": f"Find the principal angle using inverse {fn}."},
            {"step": 3, "calculation": f"Principal angle = {_trig_clean_number(principal)}°."},
            {"step": 4, "description": f"Solutions in 0° to 360°: {answer}"},
            {"step": 5, "description": f"Final Answer: {answer}", "result": answer},
        ],
        dataset_entries, "Independent Trigonometric Equation Solver",
    )


def _right_triangle_ratio_solver(question: str, dataset_entries=None):
    text = str(question).lower()
    values = {}
    for name in ("opposite", "adjacent", "hypotenuse"):
        m = re.search(rf"\b{name}\s*=\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if m:
            values[name] = float(m.group(1))

    target = re.search(r"\b(?:find|calculate|evaluate)\s+(sin|cos|tan|sec|csc|cot)\b", text)
    if len(values) < 2 or not target:
        return None

    fn = target.group(1)
    if fn in ("sin", "csc") and not {"opposite", "hypotenuse"} <= values.keys():
        return None
    if fn in ("cos", "sec") and not {"adjacent", "hypotenuse"} <= values.keys():
        return None
    if fn in ("tan", "cot") and not {"opposite", "adjacent"} <= values.keys():
        return None

    base = {
        "sin": values["opposite"] / values["hypotenuse"],
        "cos": values["adjacent"] / values["hypotenuse"],
        "tan": values["opposite"] / values["adjacent"],
    }
    value = base[fn] if fn in base else 1 / base[{"sec": "cos", "csc": "sin", "cot": "tan"}[fn]]
    answer = _trig_clean_number(value)

    return _trig_response(
        question, answer,
        [
            {"step": 1, "description": "Identify the triangle side lengths."},
            {"step": 2, "description": f"Use the definition of {fn}(θ)."},
            {"step": 3, "calculation": f"{fn}(θ) = {answer}"},
            {"step": 4, "description": f"Final Answer: {answer}", "result": answer},
        ],
        dataset_entries, "Independent Right-Triangle Trigonometry Solver", value,
    )


def _trig_identity_solver(question: str, dataset_entries=None):
    text = str(question).lower().replace(" ", "")
    text = text.replace("²", "^2")
    identities = {
        "1+tan^2θ": "sec²θ",
        "1+tan^2x": "sec²x",
        "1+cot^2θ": "csc²θ",
        "1+cot^2x": "csc²x",
        "sin^2θ+cos^2θ": "1",
        "sin^2x+cos^2x": "1",
    }
    for key, answer in identities.items():
        if key in text:
            return _trig_response(
                question, answer,
                [
                    {"step": 1, "description": f"Identify the identity: {key}."},
                    {"step": 2, "formula": f"{key} = {answer}"},
                    {"step": 3, "description": f"Final Answer: {answer}", "result": answer},
                ],
                dataset_entries, "Independent Trigonometric Identity Solver",
            )
    return None


def _compound_angle_solver(question: str, dataset_entries=None):
    text = str(question).lower()
    m = re.search(
        r"\b(sin|cos|tan)\s*\(\s*(\d+(?:\.\d*)?)\s*(?:°|degrees?)\s*"
        r"([+-])\s*(\d+(?:\.\d*)?)\s*(?:°|degrees?)\s*\)", text
    )
    if not m:
        return None

    fn, a, sign, b = m.group(1), float(m.group(2)), m.group(3), float(m.group(4))
    angle = a + b if sign == "+" else a - b
    value = {"sin": math.sin, "cos": math.cos, "tan": math.tan}[fn](math.radians(angle))
    exact = _trig_exact_value(fn, angle)
    answer = exact if exact is not None else _trig_clean_number(value)

    formula = {
        "sin": "sin(A±B)=sinA cosB ± cosA sinB",
        "cos": "cos(A±B)=cosA cosB ∓ sinA sinB",
        "tan": "tan(A±B)=(tanA±tanB)/(1∓tanA tanB)",
    }[fn]

    return _trig_response(
        question, answer,
        [
            {"step": 1, "description": f"A = {a:g}°, B = {b:g}°."},
            {"step": 2, "formula": formula},
            {"step": 3, "calculation": f"{fn}({angle:g}°) = {answer}"},
            {"step": 4, "description": f"Final Answer: {answer}", "result": answer},
        ],
        dataset_entries, "Independent Compound-Angle Solver", value,
    )


def _double_angle_solver(question: str, dataset_entries=None):
    """
    Solve genuine numeric double-angle expressions only.

    The `2` must be a separate double-angle token. A normal multi-digit
    angle such as `240°` must never be interpreted as `2 × 40°`.
    """
    text = str(question).lower()
    m = re.search(
        r"\b(sin|cos|tan)\s*\(\s*2\s*"
        r"(?:[×x*]\s*|\s+)"
        r"(\d+(?:\.\d*)?)\s*(?:°|degrees?)\s*\)",
        text
    )
    if not m:
        return None

    fn = m.group(1)
    angle = 2 * float(m.group(2))
    value = {"sin": math.sin, "cos": math.cos, "tan": math.tan}[fn](math.radians(angle))
    exact = _trig_exact_value(fn, angle)
    answer = exact if exact is not None else _trig_clean_number(value)

    formula = {
        "sin": "sin(2A)=2sinA cosA",
        "cos": "cos(2A)=cos²A−sin²A",
        "tan": "tan(2A)=2tanA/(1−tan²A)",
    }[fn]

    return _trig_response(
        question, answer,
        [
            {"step": 1, "description": f"A = {float(m.group(2)):g}°."},
            {"step": 2, "formula": formula},
            {"step": 3, "calculation": f"{fn}(2A) = {fn}({angle:g}°) = {answer}"},
            {"step": 4, "description": f"Final Answer: {answer}", "result": answer},
        ],
        dataset_entries, "Independent Double-Angle Solver", value,
    )


def solve_trigonometry(
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Solve new trigonometry questions from their actual values.

    Dataset examples are context only. No stored dataset answer is copied.
    """
    if not question:
        return None

    # A given-ratio expression must be evaluated before any generic equation
    # solver sees its "sin θ = ..." clause.  Otherwise it can incorrectly
    # return angle solutions instead of the requested expression value.
    given_ratio_result = _trig_given_ratio_expression_solver(
        question,
        dataset_entries=dataset_entries,
    )
    if given_ratio_result is not None:
        return given_ratio_result

    # Direct numeric angle evaluation has priority over learned method routing.
    # This prevents a learned `double_angle` label from misreading 240° as 2×40°.
    direct_match = re.search(
        r"\b(?:evaluate|calculate|compute|find|what\s+is)\b.*?"
        r"\b(sin|cos|tan|sec|csc|cot)\s*\(\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:°|degrees?|deg)\s*\)",
        str(question or "").lower(),
        re.I,
    )
    if direct_match:
        function_name = direct_match.group(1)
        angle = float(direct_match.group(2))
        exact_value = _trig_exact_value(function_name, angle)
        if exact_value is not None:
            learning_context = _get_dataset_method_learning(
                question,
                subject="trigonometry",
                dataset_entries=dataset_entries,
                supplied_context=dataset_learning,
            )
            direct_result = _trig_response(
                question,
                exact_value,
                [
                    {"step": 1, "description": f"Identify the angle: {angle:g}°."},
                    {
                        "step": 2,
                        "description": f"Use the standard-angle value for {function_name}.",
                        "formula": f"{function_name}({angle:g}°)",
                    },
                    {
                        "step": 3,
                        "calculation": f"{function_name}({angle:g}°) = {exact_value}",
                    },
                    {
                        "step": 4,
                        "description": f"Final Answer: {exact_value}",
                        "result": exact_value,
                    },
                ],
                dataset_entries,
                "Independent Trigonometry Solver",
            )
            if isinstance(direct_result, dict):
                direct_result["dataset_learning"] = _dataset_learning_summary(
                    dataset_entries,
                    learning_context=learning_context,
                    question=question,
                    subject="trigonometry",
                )
                direct_result["dataset_learning_used"] = False
                direct_result["learned_method"] = None
                direct_result["similarity_answer_copy"] = False
            return direct_result

        function = _TRIG_FUNCTIONS.get(function_name)
        if function is not None:
            radians_value = math.radians(angle)
            try:
                value = function(radians_value)
            except (ValueError, ZeroDivisionError, OverflowError):
                value = None
            if value is not None and math.isfinite(value):
                result_text = _trig_clean_number(value)
                learning_context = _get_dataset_method_learning(
                    question,
                    subject="trigonometry",
                    dataset_entries=dataset_entries,
                    supplied_context=dataset_learning,
                )
                direct_result = _trig_response(
                    question,
                    result_text,
                    [
                        {"step": 1, "description": f"Identify the angle: {angle:g}°."},
                        {
                            "step": 2,
                            "formula": "radians = degrees × π / 180",
                            "calculation": f"{angle:g} × π / 180 = {radians_value:.10f}",
                        },
                        {
                            "step": 3,
                            "description": f"Evaluate {function_name} using the actual user-provided angle.",
                            "calculation": f"{function_name}({radians_value:.10f})",
                        },
                        {
                            "step": 4,
                            "description": f"{function_name}({angle:g}°) ≈ {result_text}",
                            "result": result_text,
                        },
                        {
                            "step": 5,
                            "description": f"Final Answer: {result_text}",
                            "result": result_text,
                        },
                    ],
                    dataset_entries,
                    "Independent Numerical Trigonometry Solver",
                    value,
                )
                if isinstance(direct_result, dict):
                    direct_result["dataset_learning"] = _dataset_learning_summary(
                        dataset_entries,
                        learning_context=learning_context,
                        question=question,
                        subject="trigonometry",
                    )
                    direct_result["dataset_learning_used"] = False
                    direct_result["learned_method"] = None
                    direct_result["similarity_answer_copy"] = False
                return direct_result

    learning_context = _get_dataset_method_learning(
        question,
        subject="trigonometry",
        dataset_entries=dataset_entries,
        supplied_context=dataset_learning,
    )
    best_method = _learning_best_method(learning_context)

    # The dataset can influence solver ORDER, but never the final numeric result.
    # This turns learned examples into method selection rather than answer lookup.
    specialized_map = {
        "degree_radian": _degree_radian_solver,
        "degree_to_radian": _degree_radian_solver,
        "radian_to_degree": _degree_radian_solver,
        "law_of_cosines": _law_of_cosines_solver,
        "law_of_sines": _law_of_sines_solver,
        "inverse_trig": _inverse_trig_solver,
        "trig_equation": _trig_polynomial_equation_solver,
        "trigonometric_equation": _trig_polynomial_equation_solver,
        "right_triangle": _right_triangle_ratio_solver,
        "right_triangle_ratio": _right_triangle_ratio_solver,
        "trig_identity": _trig_identity_solver,
        "compound_angle": _compound_angle_solver,
        "double_angle": _double_angle_solver,
    }

    ordered = []
    preferred = specialized_map.get(best_method)
    if preferred is not None:
        ordered.append(preferred)

    for fn in (
        _degree_radian_solver,
        _law_of_cosines_solver,
        _law_of_sines_solver,
        _inverse_trig_solver,
        _trig_polynomial_equation_solver,
        _trig_equation_solver,
        _right_triangle_ratio_solver,
        _trig_identity_solver,
        _compound_angle_solver,
        _double_angle_solver,
    ):
        if fn not in ordered:
            ordered.append(fn)

    specialized = tuple(ordered)

    for solver_fn in specialized:
        try:
            result = solver_fn(question, dataset_entries)
            if result is not None:
                if isinstance(result, dict):
                    result["dataset_learning"] = _dataset_learning_summary(
                        dataset_entries,
                        learning_context=learning_context,
                        question=question,
                        subject="trigonometry",
                    )
                    result["dataset_learning_used"] = bool(
                        learning_context.get("enabled")
                    )
                    result["learned_method"] = best_method or None
                    result["similarity_answer_copy"] = False
                return result
        except Exception as exc:
            logger.debug("Specialized trig solver failed: %s", exc)

    extracted = _extract_trig_expression(question)
    if extracted is None:
        return None

    function_name, angle = extracted
    text = str(question).lower()
    if not ("°" in text or re.search(r"\bdegrees?\b", text) or re.search(r"\bdeg\b", text)):
        return None

    exact_value = _trig_exact_value(function_name, angle)
    if exact_value is not None:
        return _trig_response(
            question, exact_value,
            [
                {"step": 1, "description": f"Identify the angle: {angle:g}°."},
                {"step": 2, "description": f"Use the standard-angle value for {function_name}.",
                 "formula": f"{function_name}({angle:g}°)"},
                {"step": 3, "description": f"{function_name}({angle:g}°) = {exact_value}",
                 "calculation": exact_value},
                {"step": 4, "description": f"Final Answer: {exact_value}", "result": exact_value},
            ],
            dataset_entries, "Independent Trigonometry Solver",
        )

    function = _TRIG_FUNCTIONS.get(function_name)
    if function is None:
        return None

    radians_value = math.radians(angle)
    try:
        result = function(radians_value)
    except (ValueError, ZeroDivisionError, OverflowError):
        return {
            "type": "error",
            "subject": "trigonometry",
            "message": f"{function_name}({angle:g}°) cannot be calculated for this angle.",
            "confidence": 100.0,
            "dataset_assisted": bool(dataset_entries),
            "dataset_entries": len(dataset_entries or []),
            "similarity_answer_copy": False,
        }

    if not math.isfinite(result):
        return {
            "type": "error",
            "subject": "trigonometry",
            "message": f"{function_name}({angle:g}°) is undefined.",
            "confidence": 100.0,
            "dataset_assisted": bool(dataset_entries),
            "dataset_entries": len(dataset_entries or []),
            "similarity_answer_copy": False,
        }

    result_text = _trig_clean_number(result)
    return _trig_response(
        question, result_text,
        [
            {"step": 1, "description": f"Identify the angle: {angle:g}°."},
            {"step": 2, "description": "Convert degrees to radians.",
             "formula": "radians = degrees × π / 180",
             "calculation": f"{angle:g} × π / 180 = {radians_value:.10f}"},
            {"step": 3, "description": f"Evaluate {function_name} using the actual user-provided angle.",
             "formula": f"{function_name}({angle:g}°)",
             "calculation": f"{function_name}({radians_value:.10f})"},
            {"step": 4, "description": f"{function_name}({angle:g}°) ≈ {result_text}", "result": result_text},
            {"step": 5, "description": f"Final Answer: {result_text}", "result": result_text},
        ],
        dataset_entries, "Independent Numerical Trigonometry Solver", result,
    )


# ════════════════════════════════════════════════════════════════════════════
# DATASET-LEARNED KNN SOLVER
# ════════════════════════════════════════════════════════════════════════════

def _extract_knn_instance(question: str) -> Optional[Dict[str, float]]:
    """Extract CGPA, Assessment and Project values from a KNN question."""
    text = str(question or "")
    patterns = {
        "CGPA": r"\b(?:CGPA|C\.?G\.?P\.?A\.?|grade\s*point)\s*(?:['\"]\s*)?[:=]?\s*(-?(?:\d+(?:\.\d*)?|\.\d+))",
        "Assessment": r"\b(?:Assessment|assessment\s*score|assessment\s*marks?)\s*(?:['\"]\s*)?[:=]?\s*(-?(?:\d+(?:\.\d*)?|\.\d+))",
        "Project": r"\b(?:Project|project\s*score|project\s*marks?)\s*(?:['\"]\s*)?[:=]?\s*(-?(?:\d+(?:\.\d*)?|\.\d+))",
    }
    values = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.I)
        if m:
            values[key] = float(m.group(1))
    return values if len(values) == 3 else None


def _extract_knn_label(entry: dict) -> Optional[str]:
    """Extract a Pass/Fail label from a training record."""
    if not isinstance(entry, dict):
        return None

    candidates = [
        entry.get("label"),
        entry.get("class"),
        entry.get("target"),
        entry.get("result"),
        entry.get("answer"),
        entry.get("solution"),
    ]

    for value in candidates:
        if isinstance(value, str):
            match = re.search(r"\b(pass|fail)\b", value, re.I)
            if match:
                return match.group(1).title()
        elif isinstance(value, list):
            match = re.search(r"\b(pass|fail)\b", " ".join(map(str, value)), re.I)
            if match:
                return match.group(1).title()
        elif isinstance(value, dict):
            match = re.search(r"\b(pass|fail)\b", " ".join(map(str, value.values())), re.I)
            if match:
                return match.group(1).title()

    return None


def _extract_knn_k(question: str) -> int:
    m = re.search(r"\bk\s*=\s*(\d+)", str(question), re.I)
    if not m:
        m = re.search(r"\bk\s*(?:is|of)\s*(\d+)", str(question), re.I)
    try:
        k = int(m.group(1)) if m else 5
    except Exception:
        k = 5
    return max(1, min(k, 25))


def solve_knn(
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Train a KNN classifier from the supplied labeled dataset examples,
    then classify the new CGPA/Assessment/Project instance.
    """
    learning_context = _get_dataset_method_learning(
        question,
        subject="knn",
        dataset_entries=dataset_entries,
        supplied_context=dataset_learning,
    )

    if not question:
        return None

    text = re.sub(r"\bk\s*-\s*nn\b", "knn", str(question).lower())
    if not ("knn" in text or "k-nearest" in text or "nearest neighbor" in text):
        return None

    instance = _extract_knn_instance(question)
    if instance is None:
        return None

    training = []
    for entry in dataset_entries or []:
        values = _extract_knn_instance(str(entry.get("question", "")))
        label = _extract_knn_label(entry)
        if values and label:
            distance = math.sqrt(
                sum(
                    (values[key] - instance[key]) ** 2
                    for key in ("CGPA", "Assessment", "Project")
                )
            )
            training.append((distance, label, values))

    if not training:
        return {
            "type": "no_match",
            "subject": "knn",
            "message": "KNN was detected, but no labeled KNN training examples were available.",
            "confidence": 0.0,
            "similarity_answer_copy": False,
        }

    training.sort(key=lambda item: item[0])
    k = min(_extract_knn_k(question), len(training))
    nearest = training[:k]
    counts = Counter(label for _, label, _ in nearest)
    predicted = counts.most_common(1)[0][0]
    confidence = 100.0 * counts[predicted] / k

    return {
        "type": "solution",
        "subject": "knn",
        "algorithm": "Dataset-Learned KNN (Euclidean Distance)",
        "answer": predicted,
        "prediction": predicted,
        "k": k,
        "training_examples": len(training),
        "nearest_neighbors": [
            {"distance": round(distance, 6), "label": label, "features": values}
            for distance, label, values in nearest
        ],
        "confidence": round(confidence, 2),
        "solution_steps": [
            {"step": 1, "description": f"New instance: CGPA={instance['CGPA']}, Assessment={instance['Assessment']}, Project={instance['Project']}."},
            {"step": 2, "formula": "d = √((CGPA−CGPAᵢ)² + (Assessment−Assessmentᵢ)² + (Project−Projectᵢ)²)"},
            {"step": 3, "description": f"Compute Euclidean distance to {len(training)} labeled training examples."},
            {"step": 4, "description": f"Select the nearest k = {k} examples."},
            {"step": 5, "description": f"Nearest-neighbor vote: {dict(counts)}."},
            {"step": 6, "description": f"Final predicted class: {predicted}", "result": predicted},
        ],
        "dataset_assisted": True,
        "dataset_entries": len(dataset_entries or []),
        "dataset_learning": {
            "training_examples": len(training),
            "features": ["CGPA", "Assessment", "Project"],
            "method": "Euclidean distance + majority vote",
            "answer_copied_from_dataset": False,
        },
        "similarity_answer_copy": False,
        "validated": True,
    }


def solve_by_subject(
    subject: str,
    question: str,
    dataset_entries: Optional[List[dict]] = None,
    dataset_learning: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Route supported dataset subjects to independent mathematical solvers."""
    key = str(subject or "").strip().lower().replace("-", "_").replace(" ", "_")

    if key in {"mch", "statistics", "statistical"}:
        return solve_statistics(
            question,
            dataset_entries=dataset_entries,
            dataset_learning=dataset_learning,
        )

    if key in {"find_x", "findx"}:
        return solve_find_x(
            question,
            dataset_entries=dataset_entries,
            dataset_learning=dataset_learning,
        )

    if key in {"matrix", "matrices"}:
        return solve_matrix(
            question,
            dataset_entries=dataset_entries,
            dataset_learning=dataset_learning,
        )

    if key in {"trigonometry", "trig", "trigonometric"}:
        return solve_trigonometry(
            question,
            dataset_entries=dataset_entries,
            dataset_learning=dataset_learning,
        )

    if key in {"knn", "k_nearest", "nearest_neighbor", "nearest_neighbour"}:
        return solve_knn(
            question,
            dataset_entries=dataset_entries,
            dataset_learning=dataset_learning,
        )

    return None
