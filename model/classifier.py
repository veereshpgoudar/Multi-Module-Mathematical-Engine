"""
M3_ENGINE — Subject Classifier
(model/classifier.py)

Classifies a mathematical question into one of the loaded subjects.

Classification pipeline:
    Pass 1 → High-confidence keyword / phrase detection
    Pass 2 → TF-IDF cosine similarity
    Pass 3 → Confidence validation

Important:
    The classifier never blindly selects the first dataset when the
    question has insufficient evidence.

Works with dynamically loaded datasets.
"""

import re
import math
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from utils import dataset_loader


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Minimum TF-IDF similarity required before returning a confident subject.
DEFAULT_MIN_SIMILARITY = 0.12

# If the best and second-best subjects are too close, classification is
# considered ambiguous.
DEFAULT_MIN_SCORE_GAP = 0.025

# Minimum keyword confidence.
DEFAULT_KEYWORD_CONFIDENCE = 0.60


# ════════════════════════════════════════════════════════════════════════════
# STOP WORDS
# ════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "calculate",
    "can",
    "find",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "solve",
    "the",
    "this",
    "to",
    "value",
    "what",
    "with",
}


# ════════════════════════════════════════════════════════════════════════════
# SUBJECT KEYWORD BANKS
# ════════════════════════════════════════════════════════════════════════════

SUBJECT_KEYWORDS: Dict[str, List[str]] = {

    "algebra": [
        "equation",
        "linear equation",
        "linear",
        "quadratic",
        "quadratic equation",
        "polynomial",
        "factor",
        "factorization",
        "expand",
        "simplify",
        "expression",
        "variable",
        "coefficient",
        "solve for",
        "system of equations",
        "exponent",
        "exponential",
        "logarithm",
        "log",
        "arithmetic sequence",
        "geometric sequence",
        "binomial",
        "inequality",
        "absolute value",
        "roots",
        "root",
        "zeroes",
        "zeros",
    ],

    "matrix": [
        "matrix",
        "matrices",
        "matrix addition",
        "matrix subtraction",
        "matrix multiplication",
        "matrix product",
        "transpose of matrix",
        "determinant of matrix",
        "inverse of matrix",
        "adjoint of matrix",
        "rank of matrix",
        "trace of matrix",
        "eigenvalue of matrix",
        "eigenvector of matrix",
        "symmetric matrix",
        "skew symmetric matrix",
        "cofactor matrix",
        "minor of matrix",
        "row matrix",
        "column matrix",
        "square matrix",
        "identity matrix",
        "zero matrix",
    ],

    "calculus": [
        "derivative",
        "differentiate",
        "differentiation",
        "integration",
        "integral",
        "integrate",
        "limit",
        "limits",
        "continuous",
        "continuity",
        "differentiable",
        "chain rule",
        "product rule",
        "quotient rule",
        "implicit differentiation",
        "related rates",
        "optimization",
        "maximum",
        "minimum",
        "inflection",
        "concave",
        "convex",
        "taylor",
        "maclaurin",
        "series",
        "convergence",
        "divergence",
        "partial derivative",
        "double integral",
        "triple integral",
        "surface",
        "vector field",
        "gradient",
        "curl",
        "divergence",
        "divergence theorem",
        "stokes theorem",
    ],

    "statistics": [
        "mean",
        "median",
        "mode",
        "variance",
        "standard deviation",
        "probability",
        "distribution",
        "normal distribution",
        "binomial distribution",
        "poisson",
        "hypothesis",
        "hypothesis testing",
        "confidence interval",
        "regression",
        "linear regression",
        "correlation",
        "covariance",
        "scatter",
        "scatter plot",
        "z score",
        "z-score",
        "t test",
        "t-test",
        "chi square",
        "chi-square",
        "anova",
        "p value",
        "p-value",
        "sample",
        "population",
        "random variable",
        "expected value",
        "bayes",
        "permutation",
        "combination",
        "pearson",
        "spearman",
        "frequency",
        "frequency distribution",
    ],

    "trigonometry": [
        "sin",
        "cos",
        "tan",
        "sine",
        "cosine",
        "tangent",
        "angle",
        "radian",
        "radians",
        "degree",
        "degrees",
        "unit circle",
        "inverse trigonometry",
        "inverse trig",
        "arcsin",
        "arccos",
        "arctan",
        "sec",
        "csc",
        "cot",
        "secant",
        "cosecant",
        "cotangent",
        "pythagorean",
        "pythagorean identity",
        "law of sines",
        "law of cosines",
        "amplitude",
        "period",
        "phase shift",
        "trigonometric identity",
        "identity",
        "double angle",
        "half angle",
        "sum to product",
        "product to sum",
    ],

    # These entries are useful when the corresponding datasets exist.
    "knn": [
        "knn",
        "k nearest neighbor",
        "k nearest neighbours",
        "nearest neighbor",
        "nearest neighbours",
        "classification",
        "classifier",
        "euclidean distance",
        "distance based classification",
        "majority vote",
    ],

    "mch": [
        "covariance",
        "covariance matrix",
        "mean",
        "variance",
        "correlation coefficient",
        "statistical relationship",
        "covariance calculation",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# PRE-COMPILED KEYWORD PATTERNS
# ════════════════════════════════════════════════════════════════════════════

_KW_PATTERNS: Dict[str, List[re.Pattern]] = {}


for subject, keywords in SUBJECT_KEYWORDS.items():

    patterns = []

    for keyword in keywords:

        escaped = re.escape(keyword)

        # Permit spaces, hyphens and underscores to behave similarly.
        escaped = escaped.replace(
            r"\ ",
            r"(?:[\s_-]+)"
        )

        try:

            patterns.append(
                re.compile(
                    r"(?<![a-z0-9])"
                    + escaped
                    + r"(?![a-z0-9])",
                    re.I,
                )
            )

        except re.error:

            logger.warning(
                "Could not compile keyword pattern: %s",
                keyword
            )

    _KW_PATTERNS[subject] = patterns


# ════════════════════════════════════════════════════════════════════════════
# TOKENIZATION
# ════════════════════════════════════════════════════════════════════════════

def _normalise_text(text: str) -> str:
    """
    Normalize text before classification.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Normalize mathematical Unicode characters.
    replacements = {
        "×": " * ",
        "÷": " / ",
        "−": " - ",
        "–": " - ",
        "—": " - ",
        "√": " sqrt ",
        "^": " power ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize hyphens/underscores.
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def _tokenise(text: str) -> List[str]:
    """
    Tokenize text into useful words/numbers.

    Common English stop words are removed to make similarity more
    dependent on mathematical terminology.
    """

    text = _normalise_text(text)

    if not text:
        return []

    tokens = re.findall(
        r"[a-z]+(?:[0-9]+)?|[0-9]+(?:\.[0-9]+)?",
        text
    )

    return [
        token
        for token in tokens
        if token not in _STOP_WORDS
    ]


# ════════════════════════════════════════════════════════════════════════════
# TERM FREQUENCY
# ════════════════════════════════════════════════════════════════════════════

def _tf(tokens: List[str]) -> Dict[str, float]:
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
# COSINE SIMILARITY
# ════════════════════════════════════════════════════════════════════════════

def _cosine(
    vec_a: Dict[str, float],
    vec_b: Dict[str, float]
) -> float:
    """
    Calculate cosine similarity between two sparse vectors.
    """

    if not vec_a or not vec_b:
        return 0.0

    dot = sum(
        vec_a.get(word, 0.0) * value
        for word, value in vec_b.items()
    )

    norm_a = math.sqrt(
        sum(
            value * value
            for value in vec_a.values()
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in vec_b.values()
        )
    )

    if not norm_a or not norm_b:
        return 0.0

    return dot / (
        norm_a * norm_b
    )


# ════════════════════════════════════════════════════════════════════════════
# TF-IDF CORPUS
# ════════════════════════════════════════════════════════════════════════════

class TFIDFCorpus:
    """
    Lightweight TF-IDF index built from a list of documents.
    """

    def __init__(
        self,
        documents: List[str]
    ) -> None:

        self._docs = list(
            documents or []
        )

        self._tokens = [
            _tokenise(document)
            for document in self._docs
        ]

        self._tfs = [
            _tf(tokens)
            for tokens in self._tokens
        ]

        self._idf = self._build_idf()

        self._tfidf = [
            self._apply_idf(tf)
            for tf in self._tfs
        ]

        self._norms = [
            math.sqrt(
                sum(
                    value * value
                    for value in vector.values()
                )
            )
            for vector in self._tfidf
        ]

    def _build_idf(self) -> Dict[str, float]:
        """
        Build inverse document frequency values.
        """

        n = len(self._docs)

        if n == 0:
            return {}

        document_frequency: Dict[str, int] = {}

        for tf in self._tfs:

            for word in tf:

                document_frequency[word] = (
                    document_frequency.get(word, 0) + 1
                )

        return {
            word: math.log(
                (n + 1) / (count + 1)
            ) + 1.0

            for word, count
            in document_frequency.items()
        }

    def _apply_idf(
        self,
        tf: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply IDF weights to a TF vector.
        """

        return {
            word: value * self._idf.get(
                word,
                1.0
            )

            for word, value
            in tf.items()
        }

    def best_match(
        self,
        query: str
    ) -> Tuple[int, float]:
        """
        Return:

            (document_index, similarity_score)
        """

        query_tokens = _tokenise(query)

        if not query_tokens:
            return -1, 0.0

        query_tf = _tf(
            query_tokens
        )

        query_tfidf = self._apply_idf(
            query_tf
        )

        query_norm = math.sqrt(
            sum(
                value * value
                for value
                in query_tfidf.values()
            )
        )

        if not query_norm:
            return -1, 0.0

        best_index = -1
        best_score = 0.0

        for index, (
            vector,
            vector_norm
        ) in enumerate(
            zip(
                self._tfidf,
                self._norms
            )
        ):

            if not vector_norm:
                continue

            dot = sum(
                value * vector.get(
                    word,
                    0.0
                )
                for word, value
                in query_tfidf.items()
            )

            score = dot / (
                query_norm * vector_norm
            )

            if score > best_score:

                best_score = score
                best_index = index

        return best_index, best_score


# ════════════════════════════════════════════════════════════════════════════
# SUBJECT CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

class SubjectClassifier:
    """
    Dynamic mathematical subject classifier.

    Pipeline:

        1. Keyword / phrase detection
        2. TF-IDF similarity
        3. Confidence validation
        4. Unknown / ambiguous result when evidence is weak
    """

    def __init__(self) -> None:

        self._corpora: Dict[
            str,
            Optional[TFIDFCorpus]
        ] = {}

        self._doc_lists: Dict[
            str,
            List[str]
        ] = {}

    # ════════════════════════════════════════════════════════════════════════
    # CORPUS MANAGEMENT
    # ════════════════════════════════════════════════════════════════════════

    def update_corpus(
        self,
        subject: str,
        questions: List[str]
    ) -> None:
        """
        Build/rebuild the TF-IDF corpus for a subject.

        The corpus is rebuilt only when the questions actually change.
        """

        if not subject:
            return

        subject = str(
            subject
        ).strip().lower()

        cleaned_questions = [
            str(question).strip()
            for question in (
                questions or []
            )
            if str(question).strip()
        ]

        old_questions = self._doc_lists.get(
            subject
        )

        if old_questions == cleaned_questions:
            return

        logger.debug(
            "Rebuilding TF-IDF corpus for '%s' (%d documents).",
            subject,
            len(cleaned_questions)
        )

        self._doc_lists[
            subject
        ] = list(
            cleaned_questions
        )

        self._corpora[
            subject
        ] = (
            TFIDFCorpus(
                cleaned_questions
            )
            if cleaned_questions
            else None
        )

    def remove_subject(
        self,
        subject: str
    ) -> None:
        """
        Remove a subject and its corpus.
        """

        subject = str(
            subject
        ).strip().lower()

        self._corpora.pop(
            subject,
            None
        )

        self._doc_lists.pop(
            subject,
            None
        )

    def clear(self) -> None:
        """
        Clear all loaded corpora.
        """

        self._corpora.clear()
        self._doc_lists.clear()

    def _active_subjects(self) -> List[str]:
        """
        Return subjects having usable corpora.
        """

        return [
            subject
            for subject, corpus
            in self._corpora.items()
            if corpus is not None
        ]

    # ════════════════════════════════════════════════════════════════════════
    # KEYWORD SCORING
    # ════════════════════════════════════════════════════════════════════════

    def _keyword_score(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Calculate weighted keyword scores.

        Exact multi-word phrases receive more weight than single words.
        """

        if not text:
            return {}

        normalised = _normalise_text(
            text
        )

        scores: Dict[str, float] = {}

        for subject, patterns in _KW_PATTERNS.items():

            # Only score subjects that are actually loaded.
            if subject not in self._corpora:
                continue

            if self._corpora.get(subject) is None:
                continue

            score = 0.0

            for pattern in patterns:

                matches = pattern.findall(
                    normalised
                )

                if not matches:
                    continue

                phrase = pattern.pattern

                # Multi-word phrases are stronger signals.
                if (
                    r"(?:[\s_-]+)" in phrase
                    or " " in phrase
                ):
                    weight = 2.5
                else:
                    weight = 1.0

                score += weight * min(
                    len(matches),
                    3
                )

            if score > 0:
                scores[subject] = score

        return scores

    def _keyword_confidence(
        self,
        scores: Dict[str, float]
    ) -> Tuple[Optional[str], float, float]:
        """
        Return:

            (best_subject, confidence, score_gap)
        """

        if not scores:
            return None, 0.0, 0.0

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_subject, best_score = ranked[0]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        gap = (
            best_score - second_score
        )

        # Convert keyword score to a bounded confidence.
        confidence = min(
            1.0,
            0.45 + (
                best_score * 0.08
            )
        )

        return (
            best_subject,
            confidence,
            gap
        )

    # ════════════════════════════════════════════════════════════════════════
    # TF-IDF SUBJECT SCORING
    # ════════════════════════════════════════════════════════════════════════

    def _tfidf_scores(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Calculate the best document similarity for every loaded subject.
        """

        scores: Dict[str, float] = {}

        for subject in self._active_subjects():

            corpus = self._corpora.get(
                subject
            )

            if corpus is None:
                continue

            _, score = corpus.best_match(
                text
            )

            scores[subject] = float(
                score
            )

        return scores

    # ════════════════════════════════════════════════════════════════════════
    # LEARNED DATASET PATTERNS
    # ════════════════════════════════════════════════════════════════════════

    def _learned_pattern_scores(
        self,
        text: str,
    ) -> Dict[str, float]:
        """
        Read pattern metadata learned by dataset_loader.

        This is routing evidence only. Dataset answers are never read or
        returned by the classifier.
        """
        detector = getattr(
            dataset_loader,
            "get_pattern_candidates",
            None,
        )

        if not callable(detector):
            return {}

        try:
            candidates = detector(
                str(text or "")
            )
        except Exception as exc:
            logger.warning(
                "Learned dataset pattern detection failed: %s",
                exc,
            )
            return {}

        scores: Dict[str, float] = {}

        if not isinstance(candidates, list):
            return scores

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            subject = str(
                candidate.get("subject", "")
            ).strip().lower()

            if not subject or subject not in self._corpora:
                continue

            patterns = {
                str(value).strip()
                for value in candidate.get(
                    "matched_patterns",
                    [],
                )
                if str(value).strip()
            }

            # Generic structural labels are not enough to select a solver.
            specific_patterns = patterns - {
                "general_math",
                "equation",
                "arithmetic_expression",
                "list_based",
                "variable_data",
            }

            if not specific_patterns:
                continue

            try:
                candidate_score = float(
                    candidate.get("score", 0.0)
                )
            except (TypeError, ValueError):
                candidate_score = 0.0

            if candidate_score > 0:
                scores[subject] = min(
                    scores.get(subject, 0.0)
                    + candidate_score,
                    1.0,
                )

        return scores

    # ════════════════════════════════════════════════════════════════════════
    # CLASSIFICATION
    # ════════════════════════════════════════════════════════════════════════


    def _learned_method_scores(
        self,
        text: str,
    ) -> Dict[str, float]:
        """
        Score subjects using method/formula representations learned from the
        datasets. This is stronger than raw keyword matching but remains
        routing evidence only.

        The dataset loader never supplies a stored final answer here.
        """
        getter = getattr(
            dataset_loader,
            "get_learned_method_candidates",
            None,
        )
        if not callable(getter):
            return {}

        try:
            candidates = getter(
                str(text or ""),
                min_score=0.75,
                top_k=8,
            )
        except TypeError:
            try:
                candidates = getter(
                    str(text or ""),
                    top_k=8,
                )
            except Exception as exc:
                logger.warning(
                    "Learned method candidate lookup failed: %s",
                    exc,
                )
                return {}
        except Exception as exc:
            logger.warning(
                "Learned method candidate lookup failed: %s",
                exc,
            )
            return {}

        scores: Dict[str, float] = {}
        if not isinstance(candidates, list):
            return scores

        active = set(self._active_subjects())

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            subject = str(candidate.get("subject", "")).strip().lower()
            if not subject or subject not in active:
                continue

            try:
                raw_score = float(candidate.get("score", 0.0))
            except (TypeError, ValueError):
                raw_score = 0.0

            try:
                candidate_confidence = float(
                    candidate.get("confidence", 0.0)
                )
            except (TypeError, ValueError):
                candidate_confidence = 0.0

            normalized_score = min(
                1.0,
                max(
                    0.0,
                    0.55 * candidate_confidence
                    + 0.45 * min(raw_score / 6.0, 1.0),
                ),
            )

            if normalized_score > scores.get(subject, 0.0):
                scores[subject] = normalized_score

        return scores

    def _looks_like_matrix_text(self, text: str) -> bool:
        """Detect strong matrix structure before generic subject scoring."""
        if not text:
            return False
        raw = str(text).strip().lower()
        normalised = _normalise_text(raw)
        if re.search(r"\bmatrices?\b", raw):
            return True
        if re.search(r"\[\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\])+\s*\]", raw):
            return True
        has_labels = bool(re.search(r"\b[ab]\s*=", raw))
        row_shape = bool(re.search(r"\[[^\]]+\]\s*,\s*\[[^\]]+\]", raw))
        if has_labels and row_shape:
            return True
        numeric_rows = re.findall(r"(?:^|[\n;])\s*[-+]?\d+(?:\s+[-+]?\d+){1,}\s*(?=$|[\n;])", normalised)
        return len(numeric_rows) >= 2

    def _is_statistics_text(self, text: str) -> bool:
        """Detect explicit statistics intent before generic matrix/ML routing."""
        value = str(text or "").strip().lower()
        if not value:
            return False

        patterns = (
            r"\bmean\b",
            r"\baverage\b",
            r"\bmedian\b",
            r"\bmode\b",
            r"\bvariance\b",
            r"\bstandard\s+deviation\b",
            r"\bstd(?:\\.?)\s*dev(?:iation)?\b",
            r"\bcovariance\b",
            r"\bcorrelation(?:\s+coefficient)?\b",
            r"\bpearson\b",
            r"\bspearman\b",
            r"\bregression\b",
            r"\bprobability\b",
        )
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)

    def _explicit_subject(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Detect unambiguous mathematical intents before TF-IDF.

        This is routing evidence only. It never reads or returns a dataset
        answer. Explicit solver names/patterns are preferred because a new
        question should reach the independent solver instead of being
        misclassified by a similar training example.
        """
        value = str(text or "").strip().lower()
        if not value:
            return None

        active = set(self._active_subjects())

        # Explicit statistics intent must beat generic list/matrix structure.
        # Example: covariance of X=[...] and Y=[...] is two vectors, not a matrix.
        if "statistics" in active or "mch" in active:
            if self._is_statistics_text(value):
                subject = "statistics" if "statistics" in active else "mch"
                return subject, 0.99

        # Matrix has the strongest structural signal.
        if "matrix" in value or "matrices" in value:
            if "matrix" in active:
                return "matrix", 0.99

        # KNN must be recognized before generic words such as
        # "classification", which can appear in unrelated contexts.
        knn_patterns = (
            r"\bk\s*[- ]?nearest\b",
            r"\bknn\b",
            r"\bnearest\s+neighbor(?:s)?\b",
            r"\beuclidean\s+distance\b.*\b(?:classif|neighbor)",
            r"\bmajority\s+vote\b.*\b(?:classif|neighbor)",
        )
        if "knn" in active and any(
            re.search(pattern, value, re.IGNORECASE)
            for pattern in knn_patterns
        ):
            return "knn", 0.99

        # Trigonometric formulas/questions should reach solve_trigonometry().
        trig_patterns = (
            r"\blaw\s+of\s+(?:sines|cosines)\b",
            r"\b(?:sin|cos|tan|sec|csc|cot)\b",
            r"\b(?:sine|cosine|tangent|secant|cosecant|cotangent)\b",
            r"\b(?:arcsin|arccos|arctan)\b",
            r"\b(?:trigonometry|trigonometric|unit\s+circle)\b",
        )
        if "trigonometry" in active and any(
            re.search(pattern, value, re.IGNORECASE)
            for pattern in trig_patterns
        ):
            return "trigonometry", 0.98

        # Find-X is normally handled earlier by model.py. Keep this here as
        # classifier support when classify() is called directly.
        find_x_pattern = (
            r"\b(?:find|solve|determine|calculate)\b.{0,80}"
            r"\b(?:the\s+)?(?:value\s+of\s+)?x\b"
        )
        if "find_x" in active and re.search(find_x_pattern, value):
            return "find_x", 0.98

        return None

    def classify(
        self,
        text: str
    ) -> Tuple[str, float]:
        """
        Classify a question.

        Order:
            0. Explicit solver-specific intent / structure
            1. Strong explicit keywords
            2. Learned dataset patterns
            3. TF-IDF similarity
            4. Weak keyword evidence
            5. Unknown when evidence is insufficient

        The classifier chooses a subject only. It never returns a dataset
        answer for a new question.
        """
        if not text or not str(text).strip():
            return (
                "unknown",
                0.0
            )

        # PASS 0 — explicit mathematical intent.
        # This prevents strong solver-specific wording from being displaced
        # by a superficially similar dataset question.
        explicit = self._explicit_subject(str(text).strip())
        if explicit is not None:
            return explicit

        # MATRIX GUARD — structural evidence wins over generic algebra.
        # Explicit statistics intent takes precedence over two-list structure.
        if (
            not self._is_statistics_text(str(text).strip())
            and self._looks_like_matrix_text(str(text).strip())
        ):
            return ("matrix", 0.99)

        active = self._active_subjects()

        if not active:
            logger.warning(
                "No datasets loaded — cannot classify."
            )
            return (
                "unknown",
                0.0
            )

        text = str(text).strip()

        # PASS 1 — explicit keyword evidence
        keyword_scores = self._keyword_score(
            text
        )

        if keyword_scores:
            (
                keyword_subject,
                keyword_confidence,
                keyword_gap
            ) = self._keyword_confidence(
                keyword_scores
            )

            if (
                keyword_subject is not None
                and keyword_confidence >= DEFAULT_KEYWORD_CONFIDENCE
                and keyword_gap >= 1.0
            ):
                return (
                    keyword_subject,
                    round(
                        keyword_confidence,
                        4
                    )
                )

        # PASS 2 — learned dataset patterns
        learned_scores = self._learned_pattern_scores(
            text
        )

        if learned_scores:
            ranked_learned = sorted(
                learned_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )

            learned_subject, learned_score = ranked_learned[0]
            second_learned_score = (
                ranked_learned[1][1]
                if len(ranked_learned) > 1
                else 0.0
            )

            learned_gap = (
                learned_score
                - second_learned_score
            )

            # Require useful pattern evidence and avoid selecting a nearly
            # tied learned subject.
            if (
                learned_score >= 0.20
                and (
                    len(ranked_learned) == 1
                    or learned_gap >= 0.05
                )
            ):
                confidence = min(
                    0.95,
                    0.50 + learned_score * 0.45
                )

                return (
                    learned_subject,
                    round(
                        confidence,
                        4
                    )
                )

        # PASS 3 — learned method / formula selection
        # Do not let generic arithmetic/list patterns hijack direct arithmetic.
        # Dataset learning is still used when the question contains a
        # subject-specific signal (statistics, trig, matrix, KNN, etc.).
        plain_arithmetic = bool(
            re.fullmatch(
                r"[\\s0-9.+*/()^%\\-]+",
                _normalise_text(text),
            )
        )
        method_scores = (
            {}
            if plain_arithmetic
            else self._learned_method_scores(text)
        )

        if method_scores:
            ranked_methods = sorted(
                method_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            method_subject, method_score = ranked_methods[0]
            second_method_score = (
                ranked_methods[1][1]
                if len(ranked_methods) > 1
                else 0.0
            )
            method_gap = method_score - second_method_score

            if (
                method_score >= 0.40
                and (
                    len(ranked_methods) == 1
                    or method_gap >= 0.06
                )
            ):
                return (
                    method_subject,
                    round(
                        min(0.97, 0.55 + method_score * 0.42),
                        4,
                    ),
                )

        # PASS 4 — TF-IDF similarity
        tfidf_scores = self._tfidf_scores(
            text
        )

        if not tfidf_scores:
            return (
                "unknown",
                0.0
            )

        ranked = sorted(
            tfidf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_subject, best_score = ranked[0]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        score_gap = (
            best_score
            - second_score
        )

        # PASS 4 — confidence validation
        if best_score <= 0.0:
            if keyword_scores:
                (
                    weak_subject,
                    _,
                    _
                ) = self._keyword_confidence(
                    keyword_scores
                )

                if weak_subject in active:
                    return (
                        weak_subject,
                        0.15
                    )

            return (
                "unknown",
                0.0
            )

        if best_score >= DEFAULT_MIN_SIMILARITY:
            if (
                len(ranked) > 1
                and score_gap < DEFAULT_MIN_SCORE_GAP
                and best_score < 0.30
            ):
                logger.debug(
                    "Ambiguous classification: %s=%.4f, "
                    "%s=%.4f",
                    best_subject,
                    best_score,
                    ranked[1][0],
                    second_score
                )

                return (
                    "unknown",
                    round(
                        best_score,
                        4
                    )
                )

            return (
                best_subject,
                round(
                    min(best_score, 1.0),
                    4
                )
            )

        # Weak TF-IDF score with useful keyword evidence.
        if keyword_scores:
            (
                weak_subject,
                keyword_confidence,
                keyword_gap
            ) = self._keyword_confidence(
                keyword_scores
            )

            if (
                weak_subject in active
                and keyword_gap >= 1.0
            ):
                return (
                    weak_subject,
                    round(
                        min(
                            keyword_confidence,
                            0.50
                        ),
                        4
                    )
                )

        return (
            "unknown",
            round(
                best_score,
                4
            )
        )

    # ════════════════════════════════════════════════════════════════════════
    # DETAILED CLASSIFICATION
    # ════════════════════════════════════════════════════════════════════════

    def classify_detailed(
        self,
        text: str
    ) -> Dict:
        """
        Return detailed classification information.

        Useful for debugging and for the frontend.
        """

        subject, confidence = self.classify(
            text
        )

        keyword_scores = self._keyword_score(
            text
        )

        tfidf_scores = self._tfidf_scores(
            text
        )

        learned_scores = self._learned_pattern_scores(
            text
        )

        method_scores = self._learned_method_scores(
            text
        )

        return {
            "subject": subject,

            "confidence": confidence,

            "keyword_scores": {
                key: round(
                    value,
                    4
                )
                for key, value
                in keyword_scores.items()
            },

                        "learned_method_scores": {
                key: round(
                    value,
                    4
                )
                for key, value in method_scores.items()
            },

"learned_pattern_scores": {
                key: round(
                    value,
                    4
                )
                for key, value
                in learned_scores.items()
            },

            "tfidf_scores": {
                key: round(
                    value,
                    4
                )
                for key, value
                in tfidf_scores.items()
            },

            "active_subjects": self._active_subjects(),
        }

    # ════════════════════════════════════════════════════════════════════════
    # ALL SCORES
    # ════════════════════════════════════════════════════════════════════════

    def classify_all_scores(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Return TF-IDF similarity scores for every loaded subject.

        Useful for debugging.
        """

        return {
            subject: round(
                score,
                4
            )

            for subject, score
            in self._tfidf_scores(
                text
            ).items()
        }


# ════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ════════════════════════════════════════════════════════════════════════════

classifier = SubjectClassifier()