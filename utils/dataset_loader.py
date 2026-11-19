"""
dataset_loader.py
-----------------

Central dataset loader for the Multi-Module Mathematical Engine.

Responsibilities:
- Discover JSON / JSONL datasets
- Normalize dataset subjects
- Treat x_value.json as the Find-X dataset
- Load common dataset formats
- Remove duplicate records
- Provide dataset statistics
- Support hot reload
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".json",
    ".jsonl",
}

FIND_X_SUBJECT = "find_x"
MATRIX_SUBJECT = "matrix"
KNN_SUBJECT = "knn"
TRIG_SUBJECT = "trigonometry"
STATISTICS_SUBJECT = "mch"

KNN_FILENAMES = {
    "knn", "knn_dataset", "knn_data", "knn_questions", "knn_problems",
    "k_nearest", "k_nearest_neighbor", "k_nearest_neighbors",
    "nearest_neighbor", "nearest_neighbors",
}

TRIGONOMETRY_FILENAMES = {
    "trigonometry", "trigonometric", "trigonometry_dataset",
    "trigonometric_dataset", "trig", "trig_dataset",
}

STATISTICS_FILENAMES = {
    "mch", "mch_dataset", "math", "statistics", "statistics_dataset",
    "statistic", "stats", "stats_dataset",
}

MATRIX_FILENAMES = {
    "matrix",
    "matrices",
    "matrix_dataset",
    "matrix_data",
    "matrix_questions",
    "matrix_problems",
    "matrices_dataset",
}

# IMPORTANT:
# Your x_value.json contains the Find-X dataset.
FIND_X_FILENAMES = {
    "x_value",
    "x_value_dataset",
    "find_x",
    "find_x_dataset",
    "findx",
    "findx_dataset",
}


# Lightweight mathematical vocabulary for automatic dataset pattern learning.
# This metadata helps later modules select a solver; it never supplies the
# final answer to a new question.
_PATTERN_KEYWORDS = {
    "linear_equation": (
        "linear equation", "solve for x", "find x", "determine x",
        "calculate x", "equation", "linear"
    ),
    "quadratic": (
        "quadratic", "quadratic equation", "roots of", "solve ax^2",
        "discriminant"
    ),
    "covariance": ("covariance", "covariance of"),
    "correlation": ("correlation", "correlation coefficient", "pearson"),
    "mean": ("mean", "average", "arithmetic mean"),
    "median": ("median",),
    "mode": ("mode",),
    "variance": ("variance",),
    "standard_deviation": ("standard deviation", "std deviation"),
    "probability": ("probability", "probabilities"),
    "derivative": ("derivative", "differentiate", "d/dx", "d dx"),
    "integral": ("integral", "integrate", "integration", "∫"),
    "trigonometry": (
        "sin", "cos", "tan", "sine", "cosine", "tangent",
        "sec", "csc", "cot", "secant", "cosecant", "cotangent",
        "trigonometric", "trigonometry", "radian", "degree",
        "unit circle", "inverse trig", "arcsin", "arccos", "arctan"
    ),
    "law_of_cosines": (
        "law of cosines", "law of cosine", "included angle",
        "third side of a triangle", "third side"
    ),
    "law_of_sines": ("law of sines", "law of sine", "sine rule"),
    "right_triangle": (
        "right triangle", "right-angled triangle", "hypotenuse",
        "opposite", "adjacent"
    ),
    "trig_identity": (
        "trig identity", "trigonometric identity", "simplify",
        "pythagorean identity", "tan²", "sin²", "cos²", "sec²", "csc²"
    ),
    "compound_angle": (
        "compound angle", "sum of angles", "difference of angles",
        "sin(a+b)", "cos(a+b)", "tan(a+b)"
    ),
    "double_angle": (
        "double angle", "sin 2", "cos 2", "tan 2"
    ),
    "matrix": (
        "matrix", "matrices", "determinant", "eigenvalue", "inverse matrix"
    ),
    "knn": (
        "knn", "k-nearest", "k nearest", "nearest neighbor",
        "nearest neighbours", "nearest neighbour", "euclidean distance",
        "majority vote", "distance based classification"
    ),
}

def _normalise_pattern_text(value: Any) -> str:
    """Normalize text before pattern analysis."""
    value = str(value or "").lower()
    value = value.replace("²", "^2").replace("³", "^3")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _infer_pattern_from_question(question: str) -> List[str]:
    """Infer descriptive mathematical patterns from one example."""
    text = _normalise_pattern_text(question)
    labels = []

    for label, keywords in _PATTERN_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)

    if "=" in text and re.search(r"\bx\b", text):
        if "linear_equation" not in labels and "quadratic" not in labels:
            labels.append("equation")

    has_list = bool(re.search(r"\[[^\]]+\]", text))

    # Lists are useful structural information, but `list_based` is deliberately
    # suppressed for explicit statistics questions. Otherwise a question such
    # as "Find mean of [1,2,3,4]" can acquire a weak generic pattern that later
    # competes with the real statistics solver.
    statistical_intent = any(
        keyword in text
        for keyword in (
            "mean", "average", "median", "mode", "variance",
            "standard deviation", "std deviation", "covariance",
            "correlation", "pearson",
        )
    )
    if has_list and not statistical_intent:
        labels.append("list_based")

    if re.search(r"\b(?:x|y)\s*=", text):
        labels.append("variable_data")

    if re.search(r"\b(?:cgpa|assessment|project)\b", text) and (
        "knn" in labels or "classification" in text
    ):
        labels.append("knn_features")

    if re.search(r"\bk\s*=\s*\d+\b|\bk\s*[- ]?nearest\b", text):
        if "knn" not in labels:
            labels.append("knn")
        labels.append("knn_parameter")

    if re.search(r"\b(?:x|y)\s*=\s*\[[^\]]+\]", text):
        if statistical_intent:
            labels.append("paired_numeric_lists")

    if re.search(r"[+\-*/^]", text) and not statistical_intent:
        labels.append("arithmetic_expression")

    return labels or ["general_math"]


def _dataset_pattern_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create compact, JSON-friendly learned metadata for a dataset."""
    pattern_counts = Counter()
    question_lengths = []

    for record in records:
        question = record.get("question", "")
        if not isinstance(question, str) or not question.strip():
            continue

        pattern_counts.update(_infer_pattern_from_question(question))
        question_lengths.append(len(question))

    total = len(records)

    return {
        "patterns": [
            {
                "name": name,
                "examples": count,
                "frequency": round(count / total, 4) if total else 0.0,
            }
            for name, count in pattern_counts.most_common()
        ],
        "question_count": total,
        "average_question_length": (
            round(sum(question_lengths) / len(question_lengths), 2)
            if question_lengths else 0.0
        ),
        "learning_type": "pattern_reference",
        "answer_copy_allowed": False,
    }



# ============================================================
# Dataset Learning Engine
# ============================================================

_FORMULA_KEYS = (
    "formula", "equation", "rule", "method_formula", "formula_used"
)

_STEP_KEYS = (
    "steps", "solution_steps", "reasoning_steps", "working",
    "derivation", "method_steps"
)

_METHOD_KEYS = (
    "method", "method_name", "algorithm", "technique",
    "approach", "problem_type", "type", "category"
)

_VARIABLE_PATTERN = re.compile(
    r"\b(?:[a-zA-Z](?:_[a-zA-Z0-9]+)?|theta|alpha|beta|gamma)"
    r"(?:\s*\^\s*[0-9]+)?\b"
)


def _first_text(record: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    """Return the first useful textual field from a record."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if parts:
                return " | ".join(parts)
    return ""


def _normalise_learned_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("π", "pi").replace("θ", "theta")
    text = text.replace("²", "^2").replace("³", "^3")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_method_label(record: Dict[str, Any], question: str) -> str:
    """Infer a reusable problem-solving method from explicit fields or question."""
    explicit = _first_text(record, _METHOD_KEYS)
    if explicit:
        explicit_norm = _normalise_learned_text(explicit)
        # Keep useful explicit names while removing noisy generic labels.
        if explicit_norm not in {"math", "mathematics", "general", "general math"}:
            return explicit.strip()

    patterns = _infer_pattern_from_question(question)
    priority = (
        "law_of_cosines", "law_of_sines", "right_triangle",
        "trig_identity", "compound_angle", "double_angle",
        "quadratic", "linear_equation", "covariance", "correlation",
        "mean", "median", "mode", "variance", "standard_deviation",
        "probability", "derivative", "integral", "matrix", "knn"
    )
    for label in priority:
        if label in patterns:
            return label
    return patterns[0] if patterns else "general_math"


def _extract_formula_reference(record: Dict[str, Any]) -> str:
    """Extract a formula/rule reference when the dataset provides one."""
    value = _first_text(record, _FORMULA_KEYS)
    return value


def _extract_step_references(record: Dict[str, Any]) -> List[str]:
    """Extract reusable method steps while excluding a raw final answer."""
    for key in _STEP_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [line.strip() for line in re.split(r"\n+|\s*;\s*", value) if line.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
    return []


def _extract_variable_hints(question: str, formula: str = "") -> List[str]:
    """Extract symbolic variable hints from a question/formula."""
    text = f"{question} {formula}"
    candidates = _VARIABLE_PATTERN.findall(text)
    stop = {
        "a", "an", "the", "to", "of", "in", "by", "is", "if",
        "for", "and", "or", "find", "value", "given", "let",
        "sin", "cos", "tan", "sec", "csc", "cot", "pi"
    }
    result = []
    for item in candidates:
        normalized = item.lower().strip()
        if normalized in stop:
            continue
        if normalized not in result:
            result.append(normalized)
    return result[:20]


def _content_tokens(value: Any) -> set[str]:
    """Return useful normalized content tokens for lightweight semantic matching."""
    text = _normalise_learned_text(value)
    tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", text))
    stop = {
        "find", "calculate", "determine", "value", "solve", "given",
        "using", "show", "that", "the", "and", "with", "from", "for",
        "what", "this", "where", "when", "into", "degrees", "radians",
        "question", "problem", "answer", "step", "steps"
    }
    return {token for token in tokens if token not in stop}


def _question_shape(question: str) -> Dict[str, Any]:
    """Create numerical/structural features for similarity and pattern learning."""
    text = _normalise_learned_text(question)
    numbers = []
    for token in re.findall(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", text):
        try:
            numbers.append(float(token))
        except ValueError:
            pass

    function_hits = []
    for fn in ("sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan"):
        if re.search(rf"\b{fn}\b", text):
            function_hits.append(fn)

    return {
        "token_count": len(text.split()),
        "number_count": len(numbers),
        "numbers": numbers[:30],
        "has_equals": "=" in text,
        "has_list": bool(re.search(r"\[[^\]]+\]", text)),
        "has_angle": bool(re.search(r"\b(?:degrees?|radians?|degree|radian)\b|(?:°|pi)", text)),
        "trig_functions": function_hits,
        "has_unknown_x": bool(re.search(r"\bx\b", text)),
    }


def _build_training_record(record: Dict[str, Any], subject: str) -> Optional[Dict[str, Any]]:
    """Build a privacy-safe training representation without storing final-answer targets."""
    question = record.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return None

    formula = _extract_formula_reference(record)
    method = _extract_method_label(record, question)
    patterns = _infer_pattern_from_question(question)
    steps = _extract_step_references(record)

    return {
        "subject": subject,
        "problem_type": method,
        "patterns": patterns,
        "formula_reference": formula,
        "step_references": steps,
        "variable_hints": _extract_variable_hints(question, formula),
        "features": _question_shape(question),
        "content_tokens": sorted(_content_tokens(question))[:40],
        "answer_target": None,
        "answer_copy_allowed": False,
    }


def _build_dataset_learning(records: List[Dict[str, Any]], subject: str) -> Dict[str, Any]:
    """Train a lightweight symbolic pattern/method index from dataset examples."""
    training_records = []
    method_counts = Counter()
    pattern_counts = Counter()
    formula_counts = Counter()
    feature_profile = {
        "number_count_mean": 0.0,
        "token_count_mean": 0.0,
        "angle_question_ratio": 0.0,
        "equation_question_ratio": 0.0,
    }

    for record in records:
        learned = _build_training_record(record, subject)
        if learned is None:
            continue

        training_records.append(learned)
        method_counts.update([learned["problem_type"]])
        pattern_counts.update(learned["patterns"])
        formula = _normalise_learned_text(learned["formula_reference"])
        if formula:
            formula_counts.update([formula])

    if training_records:
        n = len(training_records)
        feature_profile["number_count_mean"] = round(
            sum(item["features"]["number_count"] for item in training_records) / n, 4
        )
        feature_profile["token_count_mean"] = round(
            sum(item["features"]["token_count"] for item in training_records) / n, 4
        )
        feature_profile["angle_question_ratio"] = round(
            sum(bool(item["features"]["has_angle"]) for item in training_records) / n, 4
        )
        feature_profile["equation_question_ratio"] = round(
            sum(bool(item["features"]["has_equals"]) for item in training_records) / n, 4
        )

    # Build compact method profiles. Do not persist answer/response fields.
    method_profiles = []
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for item in training_records:
        by_method.setdefault(item["problem_type"], []).append(item)

    for method, items in sorted(
        by_method.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        formulas = [
            x["formula_reference"] for x in items
            if x.get("formula_reference")
        ]
        all_patterns = Counter()
        variables = Counter()
        vocabulary = Counter()
        for item in items:
            all_patterns.update(item.get("patterns", []))
            variables.update(item.get("variable_hints", []))
            vocabulary.update(item.get("content_tokens", []))

        number_mean = sum(
            item["features"].get("number_count", 0) for item in items
        ) / len(items)
        token_mean = sum(
            item["features"].get("token_count", 0) for item in items
        ) / len(items)

        method_profiles.append({
            "method": method,
            "examples": len(items),
            "frequency": round(len(items) / len(training_records), 6)
                if training_records else 0.0,
            "patterns": [name for name, _ in all_patterns.most_common(10)],
            "formula_references": list(dict.fromkeys(formulas))[:10],
            "variable_hints": [name for name, _ in variables.most_common(15)],
            "vocabulary": [name for name, _ in vocabulary.most_common(30)],
            "number_count_mean": round(number_mean, 4),
            "token_count_mean": round(token_mean, 4),
        })

    return {
        "subject": subject,
        "training_examples": len(training_records),
        "methods": method_profiles,
        "method_counts": dict(method_counts),
        "pattern_counts": dict(pattern_counts),
        "formula_references": [
            {"formula": formula, "examples": count}
            for formula, count in formula_counts.most_common(20)
        ],
        "feature_profile": feature_profile,
        "learning_type": "method_formula_pattern_learning",
        "answer_copy_allowed": False,
        "training_records": training_records,
    }


def _score_learned_method(
    question: str,
    method_profile: Dict[str, Any],
) -> float:
    """Score a new question against learned method metadata."""
    text = _normalise_learned_text(question)
    detected = set(_infer_pattern_from_question(question))
    score = 0.0

    for pattern in method_profile.get("patterns", []):
        if pattern in detected:
            score += 3.0

    method_text = _normalise_learned_text(method_profile.get("method", ""))
    if method_text and (
        method_text in text
        or method_text.replace("_", " ") in text
    ):
        score += 4.0

    question_tokens = _content_tokens(question)
    learned_vocabulary = set(method_profile.get("vocabulary", []))
    if question_tokens and learned_vocabulary:
        overlap = len(question_tokens & learned_vocabulary)
        score += min(3.0, overlap * 0.35)

    variable_hints = set(method_profile.get("variable_hints", []))
    variables = set(_extract_variable_hints(question))
    score += min(len(variable_hints & variables), 4) * 0.5

    shape = _question_shape(question)

    expected_numbers = method_profile.get("number_count_mean")
    if expected_numbers is not None:
        distance = abs(shape["number_count"] - float(expected_numbers))
        score += max(0.0, 1.5 - 0.25 * distance)

    if shape["has_angle"] and any(
        pattern in detected
        for pattern in ("trigonometry", "law_of_cosines", "law_of_sines", "right_triangle")
    ):
        score += 1.0

    return round(score, 6)


def get_learned_method_candidates(
    question: str,
    subject: Optional[str] = None,
    min_score: float = 0.75,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return methods/formula references learned from dataset examples.

    IMPORTANT:
        This function never returns a stored answer and never selects one as
        the final answer. It only recommends a mathematical method/pattern.
    """
    load_all()
    normalized_subject = normalize_subject_name(subject) if subject else None

    candidate_subjects = (
        [normalized_subject] if normalized_subject in _DATASET_LEARNING
        else list(_DATASET_LEARNING.keys())
    )
    results = []

    for subj in candidate_subjects:
        metadata = _DATASET_LEARNING.get(subj, {})
        methods = metadata.get("methods", [])
        for method in methods:
            score = _score_learned_method(question, method)
            if score < min_score:
                continue
            confidence = round(
                min(0.99, 1.0 - math.exp(-max(score, 0.0) / 4.5)),
                4,
            )
            results.append({
                "subject": subj,
                "method": method.get("method"),
                "score": score,
                "confidence": confidence,
                "patterns": method.get("patterns", []),
                "formula_references": method.get("formula_references", []),
                "variable_hints": method.get("variable_hints", []),
                "training_examples": method.get("examples", 0),
                "learning_type": "dataset_learned_method_selection",
                "answer_copy_allowed": False,
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max(1, int(top_k))]


def get_learned_training_records(
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return answer-free learned representations for training/reference modules.
    """
    load_all()
    if subject is None:
        records = []
        for metadata in _DATASET_LEARNING.values():
            records.extend(metadata.get("training_records", []))
        return records

    normalized = normalize_subject_name(subject)
    return list(
        _DATASET_LEARNING.get(normalized, {}).get("training_records", [])
    )


def get_learning_model(
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    """Return rich learned method/formula metadata."""
    load_all()
    if subject is None:
        return {key: dict(value) for key, value in _DATASET_LEARNING.items()}

    normalized = normalize_subject_name(subject)
    return dict(_DATASET_LEARNING.get(normalized, {}))


# ============================================================
# Internal cache
# ============================================================

_DATASETS: Dict[str, List[Dict[str, Any]]] = {}
_DATASET_PATHS: Dict[str, Path] = {}
_DATASET_MTIMES: Dict[str, float] = {}
_DATASET_PATTERNS: Dict[str, Dict[str, Any]] = {}
# Rich learned representations used by model.py/solver.py.
# These are method/pattern/formula references, never final-answer lookup tables.
_DATASET_LEARNING: Dict[str, Dict[str, Any]] = {}

_LAST_SCAN_TIME = 0.0
_SCAN_INTERVAL = 1.0


# ============================================================
# Dataset directory
# ============================================================

def _datasets_dir() -> Path:
    """
    Return the configured datasets directory.
    """
    path = Path(config.DATASETS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# Subject normalization
# ============================================================

def _stem_to_subject(filename: str) -> str:
    """
    Convert a dataset filename into a normalized subject name.

    IMPORTANT:
        x_value.json -> find_x

    Examples:
        knn_dataset.json              -> knn
        mch.json                      -> mch
        x_value.json                  -> find_x
        find_x_dataset.json           -> find_x
        find-x-dataset.json           -> find_x
    """

    stem = Path(filename).stem.lower().strip()

    # Files copied/downloaded from ChatGPT/Windows often get a numeric
    # duplicate suffix such as "knn_dataset(5)" or "matrix(1)".
    # That suffix is not part of the dataset subject. Remove it BEFORE
    # subject detection so all datasets are discovered consistently.
    normalized = stem.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"\(\d+\)$", "", normalized)


    # --------------------------------------------------------
    # Find-X detection
    # --------------------------------------------------------

    if normalized in FIND_X_FILENAMES:
        return FIND_X_SUBJECT

    # Explicit KNN dataset names.
    if normalized in KNN_FILENAMES or normalized.startswith("knn_"):
        return KNN_SUBJECT

    # Explicit trigonometry dataset names.
    if normalized in TRIGONOMETRY_FILENAMES or normalized.startswith("trig_"):
        return TRIG_SUBJECT

    # Explicit Matrix dataset names.
    # Keep Matrix separate from Algebra / Find-X.
    if normalized in MATRIX_FILENAMES:
        return MATRIX_SUBJECT

    # Explicit statistics/MCH dataset names.
    if normalized in STATISTICS_FILENAMES or normalized.startswith("mch_"):
        return STATISTICS_SUBJECT

    if normalized.startswith("matrix_") or normalized.startswith("matrices_"):
        return MATRIX_SUBJECT

    # Handle names such as:
    # find_x_dataset_1000_verified
    # findx_1000
    # x_value_1000_verified

    if normalized.startswith("find_x"):
        return FIND_X_SUBJECT

    if normalized.startswith("findx"):
        return FIND_X_SUBJECT

    if normalized.startswith("x_value"):
        return FIND_X_SUBJECT

    # --------------------------------------------------------
    # Remove common dataset suffixes
    # --------------------------------------------------------

    suffixes = (
        "_dataset",
        "_data",
        "_questions",
        "_problems",
        "_qa",
    )

    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[
                : -len(suffix)
            ]
            break

    return normalized


# ============================================================
# JSON reading
# ============================================================

def _read_json(path: Path) -> Any:
    """
    Read a normal JSON file.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Any]:
    """
    Read JSON Lines file.
    """

    records = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as exc:
                logger.warning(
                    "Invalid JSONL line %s in %s: %s",
                    line_number,
                    path.name,
                    exc,
                )

    return records


# ============================================================
# Extract records from JSON
# ============================================================

def _extract_records(data: Any) -> List[Dict[str, Any]]:
    """
    Convert different JSON structures into a list of records.

    Supported:

    [
        {...},
        {...}
    ]

    or:

    {
        "data": [
            {...}
        ]
    }

    or:

    {
        "questions": [
            {...}
        ]
    }

    or:

    {
        "problems": [
            {...}
        ]
    }

    or:

    {
        "items": [
            {...}
        ]
    }
    """

    # --------------------------------------------------------
    # Direct list
    # --------------------------------------------------------

    if isinstance(data, list):

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    # --------------------------------------------------------
    # Dictionary wrapper
    # --------------------------------------------------------

    if isinstance(data, dict):

        possible_keys = (
            "data",
            "questions",
            "problems",
            "items",
            "records",
            "examples",
            "dataset",
            "samples",
        )

        for key in possible_keys:

            value = data.get(key)

            if isinstance(value, list):

                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

        # ----------------------------------------------------
        # Single record
        # ----------------------------------------------------

        if any(
            key in data
            for key in (
                "question",
                "problem",
                "query",
                "input",
            )
        ):
            return [data]

    return []


# ============================================================
# Normalize record
# ============================================================

def _normalize_record(
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Normalize a dataset record.

    The original fields are preserved.
    """

    if not isinstance(record, dict):
        return None

    cleaned = dict(record)

    # --------------------------------------------------------
    # Normalize question field
    # --------------------------------------------------------

    question = None

    for key in (
        "question",
        "problem",
        "query",
        "input",
        "prompt",
        "text",
    ):

        value = cleaned.get(key)

        if isinstance(value, str) and value.strip():
            question = value.strip()
            break

    if question is None:
        return None

    # Ensure a standard question field exists
    cleaned["question"] = question

    # --------------------------------------------------------
    # Normalize solution if available
    # --------------------------------------------------------

    if "solution" not in cleaned:

        for key in (
            "answer",
            "response",
            "explanation",
            "steps",
        ):

            if key in cleaned:
                cleaned["solution"] = cleaned[key]
                break

    return cleaned


# ============================================================
# Deduplicate records
# ============================================================

def _deduplicate(
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    result = []
    seen = set()

    for record in records:

        question = str(
            record.get("question", "")
        ).strip().lower()

        if not question:
            continue

        if question in seen:
            continue

        seen.add(question)
        result.append(record)

    return result


# ============================================================
# Load single dataset
# ============================================================

def _load_file(
    path: Path,
) -> Tuple[str, List[Dict[str, Any]]]:

    subject = _stem_to_subject(path.name)

    logger.info(
        "Loading dataset: %s -> subject=%s",
        path.name,
        subject,
    )

    try:

        if path.suffix.lower() == ".jsonl":
            raw_data = _read_jsonl(path)

        else:
            raw_data = _read_json(path)

    except Exception as exc:

        logger.exception(
            "Failed to read dataset %s: %s",
            path,
            exc,
        )

        return subject, []

    records = _extract_records(raw_data)

    normalized_records = []

    for record in records:

        normalized = _normalize_record(record)

        if normalized is not None:
            normalized_records.append(normalized)

    if getattr(config, "DATASET_DEDUPLICATE_QUESTIONS", False):
        normalized_records = _deduplicate(normalized_records)

    max_records = max(
        1,
        int(getattr(config, "MAX_DATASET_RECORDS_PER_SUBJECT", 50000)),
    )
    normalized_records = normalized_records[:max_records]

    logger.info(
        "Loaded %s: %d records",
        subject,
        len(normalized_records),
    )

    # Special Find-X logging
    if subject == FIND_X_SUBJECT:

        logger.info(
            "============================================"
        )

        logger.info(
            "FIND-X DATASET LOADED"
        )

        logger.info(
            "File: %s",
            path,
        )

        logger.info(
            "Entries: %d",
            len(normalized_records),
        )

        logger.info(
            "============================================"
        )

    return subject, normalized_records


# ============================================================
# Discover dataset files
# ============================================================

def _discover_files() -> Dict[str, Path]:
    """
    Discover all supported dataset files.

    Returns:
        {
            "knn": Path(...),
            "mch": Path(...),
            "find_x": Path(...)
        }
    """

    directory = _datasets_dir()

    discovered: Dict[str, Path] = {}

    if not directory.exists():
        logger.warning(
            "Dataset directory does not exist: %s",
            directory,
        )
        return discovered

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    logger.info(
        "Scanning dataset directory: %s",
        directory,
    )

    for path in files:

        subject = _stem_to_subject(
            path.name
        )

        # ----------------------------------------------------
        # Find-X
        # ----------------------------------------------------

        if subject == FIND_X_SUBJECT:

            logger.info(
                "Find-X dataset detected: %s",
                path.name,
            )

            # If multiple Find-X files exist,
            # prefer the explicitly named x_value file.
            existing = discovered.get(
                FIND_X_SUBJECT
            )

            if existing is None:

                discovered[
                    FIND_X_SUBJECT
                ] = path

            else:

                existing_stem = (
                    existing.stem.lower()
                )

                current_stem = (
                    path.stem.lower()
                )

                if current_stem == "x_value":

                    discovered[
                        FIND_X_SUBJECT
                    ] = path

            continue

        # ----------------------------------------------------
        # Normal datasets
        # ----------------------------------------------------

        if subject not in discovered:

            discovered[subject] = path

    # --------------------------------------------------------
    # Final Find-X check
    # --------------------------------------------------------

    if FIND_X_SUBJECT not in discovered:

        logger.warning(
            "Find-X dataset was NOT discovered."
        )

    else:

        logger.info(
            "Find-X dataset path: %s",
            discovered[FIND_X_SUBJECT],
        )

    return discovered


def _discover_all_files() -> Dict[str, List[Path]]:
    """Discover every supported dataset file grouped by normalized subject."""
    directory = _datasets_dir()
    grouped: Dict[str, List[Path]] = {}
    if not directory.exists():
        return grouped
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        subject = _stem_to_subject(path.name)
        grouped.setdefault(subject, []).append(path)
    if FIND_X_SUBJECT in grouped:
        grouped[FIND_X_SUBJECT].sort(
            key=lambda p: (p.stem.lower() != "x_value", p.name.lower())
        )
    return grouped


# ============================================================
# Load all datasets
# ============================================================

def load_all(
    force: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover and load all datasets.

    Args:
        force:
            Force reload even if datasets are already cached.

    Returns:
        Dictionary containing all loaded datasets.
    """

    global _LAST_SCAN_TIME

    now = time.time()

    if (
        not force
        and _DATASETS
        and now - _LAST_SCAN_TIME
        < _SCAN_INTERVAL
    ):
        return _DATASETS

    _LAST_SCAN_TIME = now

    grouped = _discover_all_files()
    discovered = _discover_files()

    # Remove subjects whose contributing files disappeared.
    for subject in list(_DATASETS.keys()):
        if subject not in grouped:
            _DATASETS.pop(subject, None)
            _DATASET_PATHS.pop(subject, None)
            _DATASET_MTIMES.pop(subject, None)
            _DATASET_PATTERNS.pop(subject, None)
            _DATASET_LEARNING.pop(subject, None)

    # Load/merge every file belonging to each subject.
    for subject, paths in grouped.items():
        mtimes = []
        for path in paths:
            try:
                mtimes.append(path.stat().st_mtime_ns)
            except OSError:
                mtimes.append(0)

        combined_mtime = float(max(mtimes) if mtimes else 0)
        should_load = (
            force
            or subject not in _DATASETS
            or _DATASET_MTIMES.get(subject) != combined_mtime
        )
        if not should_load:
            continue

        combined_records: List[Dict[str, Any]] = []
        for path in paths:
            loaded_subject, records = _load_file(path)
            if loaded_subject == subject:
                combined_records.extend(records)

        if getattr(config, "DATASET_DEDUPLICATE_QUESTIONS", False):
            combined_records = _deduplicate(combined_records)

        max_records = max(
            1,
            int(getattr(config, "MAX_DATASET_RECORDS_PER_SUBJECT", 50000)),
        )
        records = combined_records[:max_records]
        _DATASETS[subject] = records

        canonical_path = discovered.get(subject) or paths[0]
        _DATASET_PATHS[subject] = canonical_path
        _DATASET_MTIMES[subject] = combined_mtime
        _DATASET_PATTERNS[subject] = _dataset_pattern_summary(records)
        _DATASET_LEARNING[subject] = _build_dataset_learning(records, subject)

        logger.info(
            "Learned patterns for %s from %d file(s): %s",
            subject,
            len(paths),
            [item["name"] for item in _DATASET_PATTERNS[subject]["patterns"]],
        )

    # --------------------------------------------------------
    # Explicit Find-X status
    # --------------------------------------------------------

    if FIND_X_SUBJECT in _DATASETS:

        logger.info(
            "Find-X loaded successfully: %d entries",
            len(
                _DATASETS[
                    FIND_X_SUBJECT
                ]
            ),
        )

    else:

        logger.warning(
            "Find-X dataset was NOT discovered."
        )

    return _DATASETS


# ============================================================
# Hot reload
# ============================================================

def reload_if_changed() -> bool:
    """
    Reload datasets if any dataset file changed.

    Returns:
        True if a reload occurred.
    """

    discovered = _discover_files()

    changed = False

    # New / changed files
    for subject, path in discovered.items():

        try:
            mtime = float(path.stat().st_mtime_ns)

        except OSError:
            continue

        # load_all() stores the combined nanosecond mtime. Keep reload_if_changed()
        # on the same time basis so a dataset change is not missed.
        if (
            subject not in _DATASET_MTIMES
            or _DATASET_MTIMES[subject]
            != mtime
        ):
            changed = True
            break

    # Removed files
    if not changed:

        for subject in _DATASETS:

            if subject not in discovered:

                changed = True
                break

    if changed:

        load_all(force=True)
        return True

    return False


# ============================================================
# Subject alias normalization
# ============================================================

def normalize_subject_name(subject: Any) -> str:
    """Normalize public subject names consistently across the engine.

    This function is intentionally conservative: Matrix and Find-X remain
    separate subjects and are never silently converted into Algebra.
    """
    normalized = (
        str(subject or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in FIND_X_FILENAMES or normalized in {
        "xvalue", "findx", "find_x_dataset"
    }:
        return FIND_X_SUBJECT

    if normalized in KNN_FILENAMES or normalized.startswith("knn_"):
        return KNN_SUBJECT

    if normalized in TRIGONOMETRY_FILENAMES or normalized in {
        "trig", "trigonometry", "trigonometric"
    } or normalized.startswith("trig_"):
        return TRIG_SUBJECT

    if normalized in MATRIX_FILENAMES or normalized in {
        "matrix", "matrices"
    }:
        return MATRIX_SUBJECT

    if normalized.startswith("matrix_") or normalized.startswith("matrices_"):
        return MATRIX_SUBJECT

    if normalized in STATISTICS_FILENAMES or normalized.startswith("mch_"):
        return STATISTICS_SUBJECT

    return normalized


def is_statistics_question(question: Any) -> bool:
    """
    Return True for explicit descriptive/inferential statistics questions.

    This is an intent guard, not an answer lookup.  It exists so callers can
    route statistics before broad structural detectors such as matrix/list
    detection.
    """
    text = _normalise_pattern_text(question)
    statistical_terms = (
        r"\bmean\b",
        r"\baverage\b",
        r"\bmedian\b",
        r"\bmode\b",
        r"\bvariance\b",
        r"\bstandard\s+deviation\b",
        r"\bstd\s+deviation\b",
        r"\bcovariance\b",
        r"\bcorrelation\b",
        r"\bpearson\b",
    )
    return any(re.search(pattern, text) for pattern in statistical_terms)


# ============================================================
# Learned dataset patterns
# ============================================================

def get_dataset_patterns(
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return automatically learned pattern metadata.

    If subject is None, return metadata for every loaded dataset.
    """
    load_all()

    if subject is None:
        return {
            name: dict(metadata)
            for name, metadata in _DATASET_PATTERNS.items()
        }

    normalized = normalize_subject_name(subject)

    return dict(_DATASET_PATTERNS.get(normalized, {}))


def get_pattern_candidates(
    question: str,
    min_frequency: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Find candidate dataset patterns for a new question.

    This only detects patterns. It does NOT return a dataset answer.
    The solver must independently calculate the final answer.
    """
    load_all()

    detected = _infer_pattern_from_question(question)
    candidates = []

    for subject, metadata in _DATASET_PATTERNS.items():
        learned = {
            item["name"]: item
            for item in metadata.get("patterns", [])
        }

        matched = [
            learned[name]
            for name in detected
            if name in learned
            and learned[name].get("frequency", 0.0) >= min_frequency
        ]

        if not matched:
            continue

        score = sum(
            item.get("frequency", 0.0)
            for item in matched
        )

        candidates.append({
            "subject": subject,
            "score": round(score, 6),
            "matched_patterns": [
                item["name"] for item in matched
            ],
            "dataset_entries": metadata.get(
                "question_count", 0
            ),
            "learning_type": "pattern_reference",
        })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates


# ============================================================
# Get subject
# ============================================================

def get_subject(
    subject: str,
) -> List[Dict[str, Any]]:
    """
    Return records for a subject.

    Example:
        get_subject("find_x")
    """

    load_all()

    normalized = normalize_subject_name(subject)

    return _DATASETS.get(
        normalized,
        [],
    )


# ============================================================
# Get subjects
# ============================================================

def get_subjects() -> List[str]:
    """
    Return all loaded subject names.
    """

    load_all()

    return sorted(
        _DATASETS.keys()
    )


# ============================================================
# Get dataset path
# ============================================================

def get_dataset_path(
    subject: str,
) -> Optional[str]:
    """
    Return dataset file path for a subject.
    """

    load_all()

    normalized = normalize_subject_name(subject)

    path = _DATASET_PATHS.get(
        normalized
    )

    if path is None:
        return None

    return str(path)


# ============================================================
# Dataset statistics
# ============================================================

def dataset_stats() -> Dict[str, Dict[str, Any]]:
    """
    Return statistics for all datasets.
    """

    load_all()

    stats = {}

    for subject, records in _DATASETS.items():

        with_answers = 0
        with_solutions = 0

        for record in records:

            if record.get("answer") not in (
                None,
                "",
                [],
                {},
            ):
                with_answers += 1

            if record.get("solution") not in (
                None,
                "",
                [],
                {},
            ):
                with_solutions += 1

        path = _DATASET_PATHS.get(
            subject
        )

        pattern_meta = _DATASET_PATTERNS.get(subject, {})
        grouped_files = _discover_all_files().get(subject, [])

        stats[subject] = {
            "entries": len(records),
            "with_answers": with_answers,
            "with_solutions": with_solutions,
            "file": (
                path.name
                if path
                else None
            ),
            "files": [p.name for p in grouped_files],
            "file_count": len(grouped_files),
            "learned_pattern_count": len(
                pattern_meta.get("patterns", [])
            ),
            "learning_type": pattern_meta.get(
                "learning_type", "pattern_reference"
            ),
            "answer_copy_allowed": False,
        }

    return stats


# ============================================================
# Dataset learning helpers
# ============================================================

def get_training_records(subject: str) -> List[Dict[str, Any]]:
    """
    Return answer-free training representations.

    The raw dataset records remain available through get_subject(); this
    public training API deliberately exposes learned features/methods without
    promoting a stored final answer.
    """
    return get_learned_training_records(subject)


def get_dataset_learning_info(
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return learned dataset metadata without selecting a stored answer.
    """
    load_all()

    def _public_model(name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        learned = _DATASET_LEARNING.get(name, {})
        return {
            "subject": name,
            "entries": len(records),
            "training_examples": learned.get("training_examples", 0),
            "patterns": _DATASET_PATTERNS.get(name, {}).get("patterns", []),
            "methods": learned.get("methods", []),
            "method_counts": learned.get("method_counts", {}),
            "formula_references": learned.get("formula_references", []),
            "feature_profile": learned.get("feature_profile", {}),
            "learning_type": learned.get(
                "learning_type", "method_formula_pattern_learning"
            ),
            "answer_copy_allowed": False,
        }

    if subject is None:
        return {
            name: _public_model(name, records)
            for name, records in _DATASETS.items()
        }

    normalized = normalize_subject_name(subject)
    return _public_model(
        normalized,
        _DATASETS.get(normalized, []),
    )


# ============================================================
# Summary
# ============================================================

def get_summary() -> Dict[str, Any]:
    """
    Return complete dataset loader summary.
    """

    load_all()

    find_x_records = _DATASETS.get(
        FIND_X_SUBJECT,
        [],
    )

    find_x_path = _DATASET_PATHS.get(
        FIND_X_SUBJECT
    )

    return {
        "datasets_dir": str(
            _datasets_dir()
        ),

        "subjects": get_subjects(),

        "dataset_count": len(
            _DATASETS
        ),

        "stats": dataset_stats(),

        "learned_patterns": get_dataset_patterns(),
        "learned_methods": get_learning_model(),

        "find_x_loaded": (
            FIND_X_SUBJECT
            in _DATASETS
        ),

        "find_x_entries": len(
            find_x_records
        ),

        "find_x_path": (
            str(find_x_path)
            if find_x_path
            else None
        ),
    }
try:
    load_all()
except Exception as exc:
    logger.exception(
        "Dataset loader initialization failed: %s",
        exc,
    )
