"""
M3_ENGINE — Model Orchestrator
(model/model.py)

Central orchestration layer for the Multi-Module Mathematical Engine.

Pipeline:

    Input
      ↓
    OCR (if image supplied)
      ↓
    Find-X detection
      ↓
    Find-X dataset learning/reference
      ↓
    Find-X equation solving + validation
      ↓
    KNN dataset-learned classification
      ↓
    Trigonometry detection + solving
      ↓
    Normal arithmetic detection
      ↓
    Numerical algorithm detection
      ↓
    Subject classification
      ↓
    Subject-specific independent solver
      ↓
    Safe no-copy fallback / no-match response

Important Find-X behavior:

    - x_value.json is loaded as the "find_x" dataset.
    - Dataset examples are used as training/reference examples.
    - A new equation does NOT need to exactly exist in the dataset.
    - The actual equation is solved by solver.py.
    - The generated result must be validated.
    - A similar dataset record must NEVER be blindly returned as
      the answer to a different equation.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import re
import time

from typing import Dict, Optional, Set


from utils import dataset_loader, preprocessor
from model import classifier as clf_module, solver

import config


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DEFAULT SETTINGS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MATCH_THRESHOLD = 0.55

FIND_X_SUBJECT = "find_x"
KNN_SUBJECT = "knn"


# ════════════════════════════════════════════════════════════════════════════
# MODEL ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

class M3Model:
    """
    Stateful orchestrator for the Multi-Module Mathematical Engine.

    Responsibilities:

        1. Load datasets
        2. Keep classifier corpora synchronized
        3. Process OCR input
        4. Detect Find-X equations
        5. Use Find-X dataset as mathematical reference/training data
        6. Generate and validate Find-X solutions
        7. Detect direct arithmetic problems
        8. Detect numerical algorithms
        9. Classify mathematical subject
        10. Use dataset-assisted independent solvers
        11. Safely handle unknown/ambiguous questions without copying answers
    """

    def __init__(self) -> None:

        self._classifier = clf_module.classifier

        # Runtime diagnostics: these paths make it obvious which modules are
        # actually imported by the running server. This is important when the
        # project contains multiple copies of model/preprocessor files.
        logger.info(
            "M3 runtime modules: model=%s | preprocessor=%s | solver=%s | classifier=%s",
            __file__,
            getattr(preprocessor, "__file__", "unknown"),
            getattr(solver, "__file__", "unknown"),
            getattr(clf_module, "__file__", "unknown"),
        )

        self._synced_subjects: Set[str] = set()

        self._last_subjects: Set[str] = set()

        # Rich dataset-learning metadata cache.  This is method/formula/pattern
        # knowledge only; stored answers are never used as final answers.
        self._last_learning_signature = None

        self._boot()

    # ════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ════════════════════════════════════════════════════════════════════════

    def _boot(self) -> None:
        """
        Load all datasets and initialize classifier corpora.
        """

        logger.info(
            "Initializing M3 mathematical engine..."
        )

        dataset_loader.load_all()

        self._sync_corpora()

        logger.info(
            "M3 engine initialized with subjects: %s",
            dataset_loader.get_subjects()
        )

    # ════════════════════════════════════════════════════════════════════════
    # DATASET → CLASSIFIER SYNCHRONIZATION
    # ════════════════════════════════════════════════════════════════════════

    def _sync_corpora(self) -> None:
        """
        Push current dataset question lists into the classifier.

        Also removes classifier corpora for datasets that no longer exist.
        """

        current_subjects = set(
            dataset_loader.get_subjects()
        )

        for subject in sorted(
            current_subjects
        ):

            entries = dataset_loader.get_subject(
                subject
            )

            questions = [
                str(
                    entry.get(
                        "question",
                        ""
                    )
                ).strip()

                for entry in entries

                if isinstance(entry, dict)

                and str(
                    entry.get(
                        "question",
                        ""
                    )
                ).strip()
            ]

            self._classifier.update_corpus(
                subject,
                questions
            )

        removed_subjects = (
            self._synced_subjects
            - current_subjects
        )

        for subject in removed_subjects:

            logger.info(
                "Removing deleted dataset subject '%s'.",
                subject
            )

            if hasattr(
                self._classifier,
                "remove_subject"
            ):

                self._classifier.remove_subject(
                    subject
                )

        self._synced_subjects = set(
            current_subjects
        )

        self._last_subjects = set(
            current_subjects
        )

    # ════════════════════════════════════════════════════════════════════════
    # HOT RELOAD
    # ════════════════════════════════════════════════════════════════════════

    def reload(self) -> Dict[str, int]:
        """
        Force dataset reload and classifier synchronization.
        """

        logger.info(
            "Reloading datasets..."
        )

        dataset_loader.load_all(
            force=True
        )

        self._sync_corpora()

        if hasattr(
            solver,
            "clear_match_cache"
        ):

            solver.clear_match_cache()

        stats = dataset_loader.dataset_stats()

        logger.info(
            "Dataset reload complete: %s",
            stats
        )

        return stats

    def _maybe_hot_reload(self) -> None:
        """
        Refresh datasets when hot reload is enabled.
        """

        if not getattr(
            config,
            "ENABLE_HOT_RELOAD",
            False
        ):

            return

        try:

            dataset_loader.load_all()

            current_subjects = set(
                dataset_loader.get_subjects()
            )

            if current_subjects != self._last_subjects:

                self._sync_corpora()

                if hasattr(
                    solver,
                    "clear_match_cache"
                ):

                    solver.clear_match_cache()

                return

            self._sync_corpora()

        except Exception as exc:

            logger.exception(
                "Hot reload failed: %s",
                exc
            )

    # ════════════════════════════════════════════════════════════════════════
    # CONFIGURATION HELPERS
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _match_threshold() -> float:
        """
        Safely read MATCH_THRESHOLD from config.py.
        """

        try:

            value = float(
                getattr(
                    config,
                    "MATCH_THRESHOLD",
                    DEFAULT_MATCH_THRESHOLD
                )
            )

            return max(
                0.0,
                min(
                    value,
                    1.0
                )
            )

        except Exception:

            return DEFAULT_MATCH_THRESHOLD

    @staticmethod
    def _normalize_subject(
        subject: Optional[str]
    ) -> Optional[str]:

        if subject is None:
            return None

        subject = str(
            subject
        ).strip().lower()

        subject = subject.replace(
            "-",
            "_"
        )

        subject = subject.replace(
            " ",
            "_"
        )

        if subject in {
            "x_value",
            "xvalue",
            "findx",
            "find_x_dataset",
            "find_x"
        }:

            return FIND_X_SUBJECT

        if subject in {
            "statistics",
            "statistic",
            "stats",
            "statistical",
        }:
            return "mch"

        return subject or None

    # ════════════════════════════════════════════════════════════════════════
    # FIND-X DETECTION
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_find_x_question(
        text: str
    ) -> bool:
        """
        Detect whether a question is a Find-X / solve-for-x problem.

        Examples:

            Find the value of x: 4x + 9 = 2x + 25
            Find x: 1250x + 3750 = 5000x - 8750
            Solve for x: 5x - 10 = 20
            What is x if 3x + 4 = 19?

        This is intentionally conservative so that ordinary arithmetic
        questions such as "Find the square root of 144" are NOT routed
        to Find-X.
        """

        if not text:
            return False

        value = str(
            text
        ).strip().lower()

        # IMPORTANT: statistical questions commonly contain
        # assignments such as ``X = [..]`` and ``Y = [..]``.
        # Those are data definitions, NOT equations to solve for x.
        # Route covariance/correlation/variance questions to their
        # statistical subject solver instead of Find-X.
        is_statistical_data_question = bool(
            re.search(
                r"\b(?:covariance|correlation|variance|standard\s+deviation|"
                r"covariance\s+matrix|correlation\s+coefficient)\b",
                value
            )
            and re.search(r"\bx\s*=\s*\[", value)
            and re.search(r"\by\s*=\s*\[", value)
        )

        if is_statistical_data_question:
            return False

        # Explicit Find-X language.
        explicit_find_x = bool(
            re.search(
                r"\b(?:find|solve|determine)\b", value
            )
            and re.search(
                r"(?<![A-Za-z])x(?![A-Za-z])", value
            )
            and "=" in value
        )

        solve_for_x = bool(
            re.search(
                r"\bsolve\s+for\s+x\b",
                value
            )
        )

        what_is_x = bool(
            re.search(
                r"\bwhat\s+is\s+x\b",
                value
            )
        )

        # Equation itself contains variable x.
        has_x_variable = bool(
            re.search(
                r"(?<![A-Za-z])x(?![A-Za-z])",
                value
            )
        )

        return (
            has_x_variable
            and (
                explicit_find_x
                or solve_for_x
                or what_is_x
            )
        )

    # ════════════════════════════════════════════════════════════════════════
    # FIND-X DATASET REFERENCE
    # ════════════════════════════════════════════════════════════════════════

    def _get_find_x_dataset(
        self
    ):

        """
        Return the Find-X dataset.

        x_value.json is mapped to find_x by dataset_loader.py.
        """

        entries = dataset_loader.get_subject(
            FIND_X_SUBJECT
        )

        if entries:

            logger.info(
                "Find-X dataset reference loaded: %d examples.",
                len(entries)
            )

        else:

            logger.warning(
                "Find-X dataset reference is empty."
            )

        return entries

    def _find_x_dataset_context(
        self,
        question: str
    ) -> dict:
        """
        Build lightweight dataset context.

        The dataset is used as a reference/training source.

        IMPORTANT:
            We do NOT return the solution of the nearest question.

        The purpose of this context is to allow solver.py to use the
        examples/patterns if it supports dataset-assisted solving.
        """

        entries = self._get_find_x_dataset()

        if not entries:

            return {
                "loaded": False,
                "entries": 0,
                "examples": []
            }

        # Find a few structurally similar examples only for reference.
        examples = []

        try:

            for entry in entries[:50]:

                if not isinstance(
                    entry,
                    dict
                ):
                    continue

                example_question = str(
                    entry.get(
                        "question",
                        ""
                    )
                ).strip()

                if not example_question:
                    continue

                examples.append(
                    {
                        "question": example_question,
                        "solution": entry.get(
                            "solution"
                        )
                    }
                )

                if len(examples) >= 5:
                    break

        except Exception as exc:

            logger.warning(
                "Unable to prepare Find-X dataset context: %s",
                exc
            )

        return {
            "loaded": True,
            "entries": len(entries),
            "examples": examples
        }

    # ════════════════════════════════════════════════════════════════════════
    # FIND-X SOLUTION
    # ════════════════════════════════════════════════════════════════════════

    def _solve_find_x(
        self,
        question: str,
        ocr_text: Optional[str],
        started_at: float
    ) -> dict:
        """
        Handle Find-X questions.

        Priority:

            1. Confirm Find-X dataset is loaded.
            2. Pass the equation to solver.py.
            3. Provide dataset context when supported.
            4. Validate generated solution when supported.
            5. NEVER use an unrelated dataset answer.

        This method intentionally does not call:
            - try_arithmetic()
            - solve_with_algorithms()

        because Find-X has its own solving route.
        """

        # Never invent an equation when the user only says "Find x".
        if "=" not in str(question or ""):
            return {
                "type": "needs_input",
                "message": "Please provide an equation containing x. Example: 2x + 5 = 17",
                "subject": FIND_X_SUBJECT,
                "input_text": question,
                "ocr_text": ocr_text,
                "dataset_assisted": False,
                "dataset_entries": 0,
                "similarity_answer_copy": False,
                "routing": "find_x_missing_equation",
                "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }

        dataset_entries = self._get_find_x_dataset()
        learning_context = self._get_dataset_learning_context(
            question,
            subject=FIND_X_SUBJECT,
        )

        if not dataset_entries:

            return {
                "type": "no_match",

                "message": (
                    "Find-X was detected, but the Find-X dataset "
                    "is not loaded."
                ),

                "subject": FIND_X_SUBJECT,

                "input_text": question,

                "ocr_text": ocr_text,

                "dataset_only": False,

                "routing": "find_x_dataset_missing",

                "processing_time_ms": round(
                    (
                        time.perf_counter()
                        - started_at
                    ) * 1000,
                    2
                ),
            }

        context = self._find_x_dataset_context(
            question
        )

        logger.info(
            "Find-X question detected."
        )

        logger.info(
            "Using %d Find-X dataset examples as reference.",
            context["entries"]
        )

        # ------------------------------------------------------------
        # Preferred new solver interface
        # ------------------------------------------------------------

        result = None

        # ------------------------------------------------------------
        # Preferred Find-X solver
        # ------------------------------------------------------------
        # The Flask process may keep an older imported solver.py module
        # in memory after the file has been replaced. Refresh the module
        # before deciding that solve_find_x() is unavailable.
        if not callable(
            getattr(
                solver,
                "solve_find_x",
                None
            )
        ):
            try:
                importlib.invalidate_caches()
                importlib.reload(solver)

                logger.info(
                    "Refreshed solver module for Find-X: %s",
                    getattr(
                        solver,
                        "__file__",
                        "unknown"
                    )
                )

            except Exception as exc:
                logger.exception(
                    "Unable to refresh solver module: %s",
                    exc
                )

        if callable(
            getattr(
                solver,
                "solve_find_x",
                None
            )
        ):

            try:

                result = solver.solve_find_x(
                    question,
                    dataset_entries=dataset_entries,
                    dataset_learning=learning_context,
                )

            except TypeError:

                # Backward-compatible call if solver.py does not accept
                # dataset_entries yet.

                result = solver.solve_find_x(
                    question
                )

            except Exception as exc:

                logger.exception(
                    "Find-X solver failed: %s",
                    exc
                )

                result = None

        # ------------------------------------------------------------
        # Backward compatibility
        # ------------------------------------------------------------

        elif callable(
            getattr(
                solver,
                "solve_linear_equation",
                None
            )
        ):

            try:

                result = solver.solve_linear_equation(
                    question
                )

            except Exception as exc:

                logger.exception(
                    "Linear Find-X solver failed: %s",
                    exc
                )

                result = None

        # ------------------------------------------------------------
        # If solver.py does not yet have Find-X support
        # ------------------------------------------------------------

        if not result:

            return {
                "type": "no_match",

                "message": (
                    "Find-X was detected and the dataset was loaded, "
                    "but the currently imported solver.py does not "
                    "provide solve_find_x(). Restart the application "
                    "and verify the imported solver path."
                ),

                "solver_module": getattr(
                    solver,
                    "__file__",
                    "unknown"
                ),

                "solver_has_find_x": callable(
                    getattr(
                        solver,
                        "solve_find_x",
                        None
                    )
                ),

                "subject": FIND_X_SUBJECT,

                "input_text": question,

                "ocr_text": ocr_text,

                "dataset_entries": len(
                    dataset_entries
                ),

                "dataset_only": False,

                "routing": "find_x_solver_missing",

                "processing_time_ms": round(
                    (
                        time.perf_counter()
                        - started_at
                    ) * 1000,
                    2
                ),
            }

        # ------------------------------------------------------------
        # Normalize result
        # ------------------------------------------------------------

        if not isinstance(
            result,
            dict
        ):

            result = {
                "type": "solution",
                "answer": result
            }

        result.setdefault(
            "type",
            "solution"
        )

        result["subject"] = FIND_X_SUBJECT

        result["input_text"] = question

        result["ocr_text"] = ocr_text

        result["routing"] = (
            "find_x_dataset_assisted_solver"
        )

        result["dataset_assisted"] = True
        result["dataset_learning_used"] = bool(learning_context.get("enabled"))
        result["dataset_learning"] = learning_context

        result["dataset_entries"] = len(
            dataset_entries
        )

        result["direct_arithmetic_used"] = False

        result["numerical_algorithm_used"] = False

        result["similarity_answer_copy"] = False

        result["processing_time_ms"] = round(
            (
                time.perf_counter()
                - started_at
            ) * 1000,
            2
        )

        return result

    # ════════════════════════════════════════════════════════════════════════
    # AUTOMATIC DATASET PATTERN ROUTING
    # ---------------------------------
    def _automatic_dataset_route(
        self,
        question: str,
    ) -> Optional[dict]:
        """
        Detect a specific mathematical pattern learned from the datasets.

        This method only chooses a possible solver subject. It never returns
        or copies an answer from a dataset example.
        """
        learning_context = self._get_dataset_learning_context(question)
        candidates = learning_context.get("candidates", [])

        if not candidates:
            detector = getattr(
                dataset_loader,
                "get_pattern_candidates",
                None,
            )
            if not callable(detector):
                return None
            try:
                candidates = detector(question)
            except Exception as exc:
                logger.warning(
                    "Automatic dataset pattern detection failed: %s",
                    exc,
                )
                return None

        if not isinstance(candidates, list):
            return None

        # These patterns are too generic to safely select a mathematical
        # solver by themselves.
        generic_patterns = {
            "general_math",
            "equation",
            "arithmetic_expression",
            "list_based",
            "variable_data",
        }

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            patterns = {
                str(value).strip()
                for value in candidate.get(
                    "matched_patterns",
                    [],
                )
                if str(value).strip()
            }

            specific_patterns = patterns - generic_patterns

            if not specific_patterns:
                continue

            subject = self._normalize_subject(
                candidate.get("subject")
            )

            if not subject:
                continue

            result = dict(candidate)
            result["subject"] = subject
            result["specific_patterns"] = sorted(
                specific_patterns
            )
            result["learning_type"] = candidate.get(
                "learning_type",
                "dataset_learned_method_selection",
            )
            result["answer_copy_allowed"] = False

            logger.info(
                "Automatic dataset pattern: subject=%s patterns=%s score=%s",
                subject,
                result["specific_patterns"],
                result.get("score"),
            )

            return result

        return None

    def _exact_dataset_solution(
        self,
        question: str,
        subjects: list,
        subject_hint: Optional[str] = None,
    ) -> Optional[dict]:
        """Return a stored solution only for an exact dataset question.

        This is a coverage fallback for the user's own datasets.  It is never
        a similarity lookup, so a different problem cannot receive a copied
        answer merely because it looks similar.
        """
        matcher = getattr(solver, "find_exact_dataset_match", None)
        formatter = getattr(solver, "format_exact_dataset_solution", None)
        if not callable(matcher) or not callable(formatter):
            return None

        hint = self._normalize_subject(subject_hint)
        ordered_subjects = []
        if hint and hint in subjects:
            ordered_subjects.append(hint)
        ordered_subjects.extend(
            subject for subject in subjects
            if subject not in ordered_subjects
        )

        for subject in ordered_subjects:
            entries = dataset_loader.get_subject(subject)
            entry = matcher(question, entries)
            if entry is None:
                continue
            try:
                result = formatter(entry, subject)
            except Exception as exc:
                logger.warning(
                    "Could not format exact dataset record for %s: %s",
                    subject,
                    exc,
                )
                continue

            if not isinstance(result, dict):
                continue
            result["dataset_assisted"] = True
            result["dataset_entries"] = len(entries)
            result["dataset_learning"] = self._get_dataset_learning_context(
                question,
                subject=subject,
            )
            result["dataset_learning_used"] = True
            result["routing"] = "exact_dataset_solution"
            return result

        return None

    def _exact_dataset_conflicts(
        self,
        question: str,
        subjects: list,
        subject_hint: Optional[str] = None,
    ) -> list:
        """Find exact questions whose duplicate dataset rows disagree.

        This is deliberately separate from similarity matching.  It prevents
        the response path from picking an arbitrary answer when the user's
        own dataset contains identical questions with different final answers.
        """
        status_getter = getattr(solver, "get_exact_dataset_match_status", None)
        if not callable(status_getter):
            return []

        hint = self._normalize_subject(subject_hint)
        ordered_subjects = []
        if hint and hint in subjects:
            ordered_subjects.append(hint)
        ordered_subjects.extend(
            subject for subject in subjects
            if subject not in ordered_subjects
        )

        conflicts = []
        for subject in ordered_subjects:
            try:
                status = status_getter(
                    question,
                    dataset_loader.get_subject(subject),
                )
            except Exception as exc:
                logger.warning(
                    "Exact dataset conflict check failed for %s: %s",
                    subject,
                    exc,
                )
                continue

            if not isinstance(status, dict) or not status.get("ambiguous"):
                continue

            conflicts.append({
                "subject": subject,
                "record_count": int(status.get("record_count", 0)),
                "answers": list(status.get("answers", [])),
            })

        return conflicts

    # GENERIC DATASET-AWARE SOLVER HOOK
    # ════════════════════════════════════════════════════════════════════════

    def _solve_with_dataset_solver(
        self,
        subject: str,
        question: str,
        entries: list,
        learning_context: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Try a subject-specific mathematical solver before generic
        similarity matching.

        Supported optional solver interfaces in solver.py are:
            solve_<subject>(question, dataset_entries=entries)
            solve_by_subject(subject, question, dataset_entries=entries)

        The dataset is supplied as reference/context. A solver result is
        accepted only when it returns a dictionary with a meaningful
        answer/result/solution_steps field. Otherwise the normal similarity
        matcher is used as a safe fallback.
        """

        if not subject or not question:
            return None

        # Do not route Find-X through this generic hook.
        if self._normalize_subject(subject) == FIND_X_SUBJECT:
            return None

        subject_key = self._normalize_subject(subject)
        if not subject_key:
            return None

        if learning_context is None:
            learning_context = self._get_dataset_learning_context(
                question,
                subject=subject_key,
            )

        # 1. Prefer solve_<subject>() when available.
        method_name = f"solve_{subject_key}"
        method = getattr(solver, method_name, None)

        if callable(method):
            try:
                try:
                    result = method(
                        question,
                        dataset_entries=entries,
                        dataset_learning=learning_context,
                    )
                except TypeError:
                    try:
                        result = method(
                            question,
                            dataset_entries=entries,
                        )
                    except TypeError:
                        result = method(question)

                if isinstance(result, dict):
                    if (
                        result.get("type") in {"needs_input", "error"}
                        or any(
                            key in result
                            for key in (
                                "answer",
                                "result",
                                "solution_steps",
                            )
                        )
                    ):
                        result.setdefault("subject", subject_key)
                        result.setdefault("type", "solved")
                        result["dataset_assisted"] = True
                        result["dataset_entries"] = len(entries)
                        result["similarity_answer_copy"] = False
                        result["dataset_learning_used"] = bool(learning_context.get("enabled"))
                        result["dataset_learning"] = learning_context
                        result["routing"] = "dataset_assisted_solver"
                        return result

            except Exception as exc:
                logger.exception(
                    "Subject solver '%s' failed: %s",
                    method_name,
                    exc,
                )

        # 2. Optional central dispatcher.
        dispatcher = getattr(
            solver,
            "solve_by_subject",
            None,
        )

        if callable(dispatcher):
            try:
                try:
                    result = dispatcher(
                        subject_key,
                        question,
                        dataset_entries=entries,
                        dataset_learning=learning_context,
                    )
                except TypeError:
                    try:
                        result = dispatcher(
                            subject_key,
                            question,
                            dataset_entries=entries,
                        )
                    except TypeError:
                        result = dispatcher(
                            subject_key,
                            question,
                        )

                if isinstance(result, dict):
                    if (
                        result.get("type") in {"needs_input", "error"}
                        or any(
                            key in result
                            for key in (
                                "answer",
                                "result",
                                "solution_steps",
                            )
                        )
                    ):
                        result.setdefault("subject", subject_key)
                        result.setdefault("type", "solved")
                        result["dataset_assisted"] = True
                        result["dataset_entries"] = len(entries)
                        result["similarity_answer_copy"] = False
                        result["dataset_learning_used"] = bool(learning_context.get("enabled"))
                        result["dataset_learning"] = learning_context
                        result["routing"] = "dataset_assisted_solver"
                        return result

            except Exception as exc:
                logger.exception(
                    "Generic dataset solver failed for '%s': %s",
                    subject_key,
                    exc,
                )

        return None

    # ════════════════════════════════════════════════════════════════════════
    # DATASET METHOD / FORMULA LEARNING
    # ════════════════════════════════════════════════════════════════════════

    def _get_dataset_learning_context(
        self,
        question: str,
        subject: Optional[str] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve the learned method/formula representation for a new question.

        The dataset contributes:
            - problem/method patterns
            - formula references
            - variable hints
            - learned vocabulary/features

        It NEVER contributes a stored final answer.
        """
        if not question:
            return {
                "enabled": False,
                "learning_type": "method_formula_pattern_learning",
                "answer_copy_allowed": False,
                "candidates": [],
            }

        detector = getattr(
            dataset_loader,
            "get_learned_method_candidates",
            None,
        )

        if not callable(detector):
            # Backward compatibility with older dataset_loader versions.
            detector = getattr(
                dataset_loader,
                "get_pattern_candidates",
                None,
            )
            if not callable(detector):
                return {
                    "enabled": False,
                    "learning_type": "unavailable",
                    "answer_copy_allowed": False,
                    "candidates": [],
                }

            try:
                candidates = detector(question)
            except Exception as exc:
                logger.warning("Dataset pattern learning failed: %s", exc)
                candidates = []

            return {
                "enabled": bool(candidates),
                "learning_type": "pattern_reference",
                "answer_copy_allowed": False,
                "candidates": candidates[:max(1, int(top_k))]
                if isinstance(candidates, list) else [],
            }

        try:
            candidates = detector(
                question,
                subject=subject,
                top_k=max(1, int(top_k)),
            )
        except TypeError:
            # Compatibility with a simpler loader signature.
            try:
                candidates = detector(question, subject=subject)
            except Exception as exc:
                logger.warning("Dataset method learning failed: %s", exc)
                candidates = []
        except Exception as exc:
            logger.warning("Dataset method learning failed: %s", exc)
            candidates = []

        if not isinstance(candidates, list):
            candidates = []

        # Do not allow any answer-like field to cross the learning boundary.
        safe_candidates = []
        for candidate in candidates[:max(1, int(top_k))]:
            if not isinstance(candidate, dict):
                continue
            safe = {
                "subject": candidate.get("subject"),
                "method": candidate.get("method"),
                "score": candidate.get("score", 0.0),
                "confidence": candidate.get("confidence", 0.0),
                "patterns": list(candidate.get("patterns", []) or []),
                "formula_references": list(
                    candidate.get("formula_references", []) or []
                ),
                "variable_hints": list(
                    candidate.get("variable_hints", []) or []
                ),
                "training_examples": candidate.get("training_examples", 0),
                "learning_type": candidate.get(
                    "learning_type",
                    "dataset_learned_method_selection",
                ),
                "answer_copy_allowed": False,
            }
            safe_candidates.append(safe)

        best = safe_candidates[0] if safe_candidates else None
        return {
            "enabled": bool(safe_candidates),
            "learning_type": "dataset_learned_method_selection",
            "answer_copy_allowed": False,
            "candidate_count": len(safe_candidates),
            "best_method": best.get("method") if best else None,
            "best_subject": best.get("subject") if best else None,
            "best_confidence": best.get("confidence") if best else 0.0,
            "candidates": safe_candidates,
        }

    @staticmethod
    def _learning_subject_from_context(context: dict) -> Optional[str]:
        """Return the highest-confidence learned subject, if any."""
        if not isinstance(context, dict):
            return None
        candidates = context.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            subject = str(candidate.get("subject") or "").strip()
            try:
                confidence = float(candidate.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if not subject or confidence < 0.50:
                continue
            # This helper is static, so it cannot use ``self``. Prefer the
            # dataset loader's canonical subject normalization and fall back
            # to the same conservative normalization rules used by the model.
            try:
                normalizer = getattr(dataset_loader, "normalize_subject_name", None)
                if callable(normalizer):
                    normalized = normalizer(subject)
                    if normalized:
                        return normalized
            except Exception:
                pass

            normalized = (
                subject.lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if normalized in {"statistics", "statistical"}:
                normalized = "mch"
            return normalized or None
        return None

    # ════════════════════════════════════════════════════════════════════════
    # OCR
    # ════════════════════════════════════════════════════════════════════════

    def _extract_text_from_image(
        self,
        image_bytes: Optional[bytes]
    ) -> Optional[str]:
        """Backward-compatible OCR entry point.

        Older M3_ENGINE versions called pytesseract directly from model.py.
        OCR is now centralized in utils.preprocessor so Tesseract can be
        detected automatically and configured safely on Windows.
        """
        return self._extract_ocr(image_bytes)

    def _extract_ocr(
        self,
        image_bytes: Optional[bytes]
    ) -> Optional[str]:

        if not image_bytes:
            return None

        try:

            logger.info(
                "Running OCR on uploaded image..."
            )

            ocr_text = preprocessor.ocr_from_bytes(
                image_bytes
            )

            if ocr_text:

                ocr_text = str(
                    ocr_text
                ).strip()

                logger.info(
                    "OCR extracted %d characters.",
                    len(ocr_text)
                )

                return ocr_text

            logger.warning(
                "OCR returned no usable text."
            )

            return None

        except Exception as exc:

            logger.exception(
                "OCR processing failed: %s",
                exc
            )

            return None

    # ════════════════════════════════════════════════════════════════════════
    # INPUT TEXT COMBINATION
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _combine_text(
        text: Optional[str],
        ocr_text: Optional[str]
    ) -> str:

        parts = []

        if text:

            typed = str(
                text
            ).strip()

            if typed:
                parts.append(
                    typed
                )

        if ocr_text:

            ocr = str(
                ocr_text
            ).strip()

            if ocr:
                parts.append(
                    ocr
                )

        return " ".join(
            parts
        ).strip()


    @staticmethod
    def _normalize_natural_language(text: Optional[str]) -> str:
        """Strong but conservative NLP normalization for mathematical queries.

        English/math wording is normalized while matrix literals, numbers,
        equations, variable names and operators are protected from aggressive
        correction.
        """
        value = str(text or "").strip()
        if not value:
            return ""

        # Unicode math symbols.
        value = value.translate(str.maketrans({
            "−": "-", "–": "-", "—": "-", "×": "*", "÷": "/",
            "∕": "/", "∗": "*", "²": "^2", "³": "^3", "π": "pi",
            "°": " degrees",
        }))

        # Natural-language abbreviations/phrases.
        phrases = [
            (r"\bwhat(?:'s| is)\s+the\s+det\b", "find the determinant"),
            (r"\bdet\s*\(\s*([a-z])\s*\)", r"determinant of \1"),
            (r"\bdet\s+of\b", "determinant of"),
            (r"\bfind\s+(?:the\s+)?det\b", "find determinant"),
            (r"\bcalculate\s+(?:the\s+)?det\b", "calculate determinant"),
            (r"\bpls\b", "please"),
            (r"\bplz\b", "please"),
            (r"\bcan\s+u\b", "can you"),
            (r"\bwhat's\b", "what is"),
            (r"\bmatrices\b", "matrix"),
        ]
        for pattern, replacement in phrases:
            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

        # Common mathematical spelling/OCR errors.
        replacements = {
            "deminant": "determinant", "determinan": "determinant",
            "determininan": "determinant", "determininant": "determinant",
            "determinent": "determinant", "determinat": "determinant",
            "determinatn": "determinant", "determinnt": "determinant",
            "determinantt": "determinant", "detminant": "determinant",
            "determinantion": "determinant", "matrx": "matrix",
            "matix": "matrix", "matirx": "matrix", "matrice": "matrix",
            "transpos": "transpose", "transopse": "transpose",
            "invers": "inverse", "invese": "inverse",
            "adjointt": "adjoint", "adjugent": "adjugate",
            "multiplcation": "multiplication", "multiplicaton": "multiplication",
            "substract": "subtract", "substraction": "subtraction",
            "subtracion": "subtraction", "addtion": "addition", "additon": "addition",
            "equasion": "equation", "equaton": "equation", "equatoin": "equation",
            "caluclate": "calculate", "calculte": "calculate", "calculat": "calculate",
            "solv": "solve", "determinee": "determine", "derivitive": "derivative",
            "intergral": "integral", "integal": "integral", "averge": "average",
            "meadian": "median", "varaince": "variance", "covaraince": "covariance",
            "correlaton": "correlation", "eigenvlaue": "eigenvalue",
            "eigenvektor": "eigenvector", "gausian": "gaussian", "bisecton": "bisection",
            "simpons": "simpson", "lagrangee": "lagrange", "trapezoidalr": "trapezoidal",
        }

        def replace_word(match: re.Match) -> str:
            word = match.group(0)
            return replacements.get(word.lower(), word)

        # Only complete alphabetic words are corrected. Matrix literals remain untouched.
        value = re.sub(r"(?<![A-Za-z])[A-Za-z]+(?![A-Za-z])", replace_word, value)

        # Conservative fuzzy correction for mathematical vocabulary.
        from difflib import get_close_matches
        math_vocab = {
            "determinant", "matrix", "inverse", "transpose", "addition",
            "subtraction", "multiplication", "division", "equation", "calculate",
            "solve", "determine", "derivative", "integral", "average", "median",
            "variance", "covariance", "correlation", "eigenvalue", "eigenvector",
            "gaussian", "bisection", "secant", "newton", "lagrange", "simpson",
            "trapezoidal", "interpolation", "euler", "runge", "kutta", "cofactor",
            "adjoint", "adjugate", "identity", "diagonal", "symmetric", "trace", "rank",
        }

        def fuzzy_word(match: re.Match) -> str:
            word = match.group(0)
            low = word.lower()
            if len(low) < 5 or low in math_vocab:
                return word
            matches = get_close_matches(low, math_vocab, n=1, cutoff=0.86)
            if matches:
                corrected = matches[0]
                logger.info("NLP fuzzy correction: %s -> %s", word, corrected)
                return corrected
            return word

        value = re.sub(r"(?<![A-Za-z])[A-Za-z]+(?![A-Za-z])", fuzzy_word, value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _underspecified_math_message(text: str) -> Optional[str]:
        """Detect clearly incomplete command-style math requests.

        Dataset examples are never used to fill missing values, equations,
        functions, angles, or data. This guard is deliberately conservative.
        """
        value = re.sub(r"\s+", " ", str(text or "").strip())
        if not value:
            return None
        low = value.lower().rstrip("?.!:")
        checks = [
            (r"(?:find|solve|calculate|determine)\s+(?:the\s+)?(?:mean|average)$", "Please provide the data values. Example: mean of [10, 20, 30]"),
            (r"(?:find|calculate|determine)\s+(?:the\s+)?(?:median|mode)$", "Please provide the data values. Example: median of [10, 20, 30]"),
            (r"(?:find|calculate|determine)\s+(?:the\s+)?(?:variance|standard\s+deviation)$", "Please provide the data values needed for the calculation."),
            (r"(?:find|calculate|determine)\s+(?:the\s+)?(?:covariance|correlation)$", "Please provide the X and Y data values. Example: X=[1,2,3], Y=[4,5,6]"),
            (r"(?:find|calculate|determine)\s+(?:the\s+)?(?:derivative|integral)$", "Please provide the function. Example: derivative of x^2 + 3x"),
            (r"(?:find|calculate|determine)\s+(?:the\s+)?(?:sin|cos|tan|sine|cosine|tangent)$", "Please provide the angle. Example: sin 30°"),
            (r"(?:find|calculate|determine)\s+(?:the\s+)?(?:factorial|square root|sqrt)$", "Please provide the number. Example: factorial of 5"),
        ]
        for pattern, message in checks:
            if re.fullmatch(pattern, low):
                return message
        return None

    @staticmethod
    def _image_fingerprint(image_bytes: Optional[bytes]) -> Optional[str]:
        """Return a short fingerprint for server-side image diagnostics."""
        if not image_bytes:
            return None
        try:
            return hashlib.sha256(image_bytes).hexdigest()[:16]
        except Exception:
            return None

    @staticmethod
    def _is_statistics_question(text: str) -> bool:
        """Detect explicit statistics intent before generic matrix routing."""
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
            r"\bstd(?:\.?\s*)dev(?:iation)?\b",
            r"\bcovariance\b",
            r"\bcorrelation(?:\s+coefficient)?\b",
            r"\bpearson\b",
            r"\bspearman\b",
            r"\bregression\b",
            r"\bprobability\b",
        )
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _is_trigonometry_question(text: str) -> bool:
        """
        Detect direct trigonometry questions before arithmetic and
        dataset-similarity fallback.

        This is intentionally conservative. It only identifies questions
        containing a trigonometric function name or common trigonometric
        terminology. The actual calculation is delegated to solver.py.

        IMPORTANT:
            A trig dataset example must never be returned as the answer to
            a different user angle/problem. solver.py receives the user's
            actual question and performs the calculation independently.
        """
        value = str(text or "").strip().lower()
        if not value:
            return False

        trig_patterns = (
            r"\b(?:sin|cos|tan|sec|csc|cot)\b",
            r"\b(?:sine|cosine|tangent|secant|cosecant|cotangent)\b",
            r"\b(?:arcsin|arccos|arctan)\b",
            r"\bsin\s*(?:\^-?1|\-?1)\b",
            r"\bcos\s*(?:\^-?1|\-?1)\b",
            r"\btan\s*(?:\^-?1|\-?1)\b",
            r"\b(?:trigonometry|trigonometric|unit\s+circle)\b",
            r"\b(?:radian|radians|degree|degrees)\b",
        )

        return any(
            re.search(pattern, value, flags=re.IGNORECASE)
            for pattern in trig_patterns
        )

    @staticmethod
    def _is_knn_question(text: str) -> bool:
        """Detect KNN classification requests before generic numeric routing."""
        value = str(text or "").strip().lower()
        if not value:
            return False

        has_knn_term = bool(re.search(
            r"\b(?:k\s*(?:-\s*)?nn|k\s*(?:-\s*)?nearest|"
            r"nearest\s*(?:-\s*)?neighbor|nearest\s*(?:-\s*)?neighbour)\b",
            value,
        ))
        if has_knn_term:
            return True

        # Dataset-specific feature classification phrasing.
        has_features = all(
            re.search(pattern, value)
            for pattern in (
                r"\bcgpa\b",
                r"\bassessment\b",
                r"\bproject\b",
            )
        )
        return bool(has_features and re.search(
            r"\b(?:classif|predict|pass|fail|instance|student)\b",
            value,
        ))

    @staticmethod
    def _text_explicitly_looks_like_matrix(text: Optional[str]) -> bool:
        """Detect explicit matrix syntax without trying to infer values."""
        value = str(text or "").strip()
        if not value:
            return False
        low = value.lower()
        if "matrix" in low or "matrices" in low:
            return True
        # A labelled matrix such as A=[[1,2],[3,4]] / B = [[...]].
        if re.search(r"\b[ab]\s*=\s*\[\s*\[", low):
            return True
        # Two bracketed rows are also a strong explicit matrix signal.
        if len(re.findall(r"\[\s*[-+]?\d", value)) >= 2 and value.count("]") >= 2:
            return True
        return False

    @staticmethod
    def _matrix_image_error(
        *,
        ocr_text: Optional[str],
        input_text: str,
        routing: str,
        started_at: float,
        fingerprint: Optional[str] = None,
    ) -> dict:
        """Create the single safe response used when matrix OCR is uncertain."""
        result = {
            "type": "needs_input",
            "message": (
                "A matrix was detected in the uploaded image, but its handwritten "
                "values could not be read reliably. Please upload a clearer image "
                "or type the matrices, for example: "
                "A=[[1,2],[3,4]], B=[[5,6],[7,8]]."
            ),
            "subject": "matrix",
            "input_text": input_text,
            "ocr_text": ocr_text,
            "dataset_assisted": False,
            "similarity_answer_copy": False,
            "routing": routing,
            "image_matrix_detected": True,
            "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        if fingerprint:
            result["image_fingerprint"] = fingerprint
        return result

    # ════════════════════════════════════════════════════════════════════════
    # MAIN SOLVER
    # ════════════════════════════════════════════════════════════════════════

    def solve(
        self,
        text: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        subject_hint: Optional[str] = None,
    ) -> dict:

        started_at = time.perf_counter()
        image_fingerprint = self._image_fingerprint(image_bytes)

        # ════════════════════════════════════════════════════════════════
        # STEP 0 — DATASET HOT RELOAD
        # ════════════════════════════════════════════════════════════════

        self._maybe_hot_reload()

        # ════════════════════════════════════════════════════════════════
        # STEP 1 — IMAGE ROUTING / OCR
        # ════════════════════════════════════════════════════════════════

        # A matrix image gets a dedicated route before generic OCR. In addition,
        # an explicit matrix subject hint is treated as a safety signal: generic
        # OCR must never be allowed to convert a matrix photo into a Find-X
        # equation.
        matrix_image_detected = False
        matrix_ocr_text = None
        matrix_subject_hint = (
            self._normalize_subject(subject_hint) in {"matrix", "matrices"}
        )

        if image_bytes:
            try:
                detector = getattr(preprocessor, "looks_like_matrix_image", None)
                if callable(detector):
                    matrix_image_detected = bool(detector(image_bytes))
            except Exception as exc:
                logger.warning(
                    "Matrix image detection failed for image %s: %s",
                    image_fingerprint or "unknown",
                    exc,
                )

        matrix_route_locked = bool(image_bytes and (matrix_image_detected or matrix_subject_hint))

        if matrix_route_locked:
            try:
                matrix_ocr = getattr(preprocessor, "matrix_ocr_from_bytes", None)
                if callable(matrix_ocr):
                    matrix_ocr_text = matrix_ocr(image_bytes)
                else:
                    logger.error(
                        "matrix_ocr_from_bytes() is missing from runtime preprocessor: %s",
                        getattr(preprocessor, "__file__", "unknown"),
                    )
            except Exception as exc:
                logger.warning("Targeted matrix OCR failed: %s", exc)
            ocr_text = matrix_ocr_text
        else:
            # Generic OCR is permitted only when the image is not on the matrix
            # route. This is the critical protection against false Find-X OCR.
            ocr_text = self._extract_ocr(image_bytes)

        if matrix_route_locked:
            ocr_text = matrix_ocr_text

        raw_combined = self._combine_text(
            text,
            ocr_text
        )

        # Normalize natural-language spelling/OCR variants before ANY
        # intent, matrix, algorithm, or subject routing.
        combined = self._normalize_natural_language(raw_combined)

        if not combined:

            if matrix_route_locked:
                return self._matrix_image_error(
                    ocr_text=ocr_text,
                    input_text="",
                    routing="matrix_handwriting_unreadable",
                    started_at=started_at,
                    fingerprint=image_fingerprint,
                )

            return {
                "type": "error",
                "message": (
                    "No question text could be extracted. "
                    "Please type a question or upload a clearer image."
                ),
                "ocr_text": ocr_text,
                "input_text": "",
                "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }

        logger.info(
            "Processing question: %r",
            combined[:200]
        )

        # ════════════════════════════════════════════════════════════════
        # STEP 2.5 — EXACT DATASET COVERAGE
        # ════════════════════════════════════════════════════════════════
        # Give an identical record from the user's datasets its own stored
        # worked solution before routing.  This guarantees coverage for every
        # valid supplied dataset question, including methods that do not yet
        # have a dedicated calculator.  The comparison is strict after only
        # harmless OCR normalization; it is never a fuzzy answer lookup.
        exact_subjects = [
            self._normalize_subject(subject)
            for subject in dataset_loader.get_subjects()
        ]
        exact_subjects = [subject for subject in exact_subjects if subject]
        # Keep the original text for exact dataset identity lookup.  General
        # routing normalization deliberately rewrites harmless wording such
        # as "matrices" → "matrix" and "π" → "pi"; using only that rewritten
        # text made valid dataset prompts miss their own record.  We first
        # try normalized text (useful for OCR typography) and then the raw
        # text, with both paths still requiring strict equality.
        exact_queries = [combined]
        if raw_combined and raw_combined != combined:
            exact_queries.append(raw_combined)

        exact_result = None
        for exact_query in exact_queries:
            exact_result = self._exact_dataset_solution(
                exact_query,
                exact_subjects,
                subject_hint=subject_hint,
            )
            if exact_result is not None:
                break
        if exact_result is not None:
            exact_result["ocr_text"] = ocr_text
            exact_result["input_text"] = combined
            exact_result["processing_time_ms"] = round(
                (time.perf_counter() - started_at) * 1000,
                2,
            )
            return exact_result

        # Identical KNN prompts must have a single verified label.  Unlike a
        # numerical expression, a KNN test case cannot be independently
        # reconstructed from a stored worked solution because the original
        # training rows are not part of that prompt.  If duplicate dataset
        # rows disagree, asking the user to correct the records is the only
        # truthful alternative to returning an arbitrary Pass/Fail result.
        exact_conflicts = []
        for exact_query in exact_queries:
            exact_conflicts = self._exact_dataset_conflicts(
                exact_query,
                exact_subjects,
                subject_hint=subject_hint,
            )
            if exact_conflicts:
                break
        knn_conflicts = [
            item for item in exact_conflicts
            if item.get("subject") == KNN_SUBJECT
        ]
        if knn_conflicts:
            conflict = knn_conflicts[0]
            answers = ", ".join(
                str(answer) for answer in conflict.get("answers", [])
            )
            return {
                "type": "needs_input",
                "subject": KNN_SUBJECT,
                "message": (
                    "This exact KNN question has conflicting labels in the "
                    f"dataset ({answers}). Please correct the {conflict.get('record_count', 0)} "
                    "duplicate records or provide the intended training data."
                ),
                "dataset_conflict": True,
                "conflicting_answers": conflict.get("answers", []),
                "conflicting_records": conflict.get("record_count", 0),
                "dataset_assisted": True,
                "similarity_answer_copy": False,
                "ocr_text": ocr_text,
                "input_text": combined,
                "routing": "dataset_conflict_guard",
                "processing_time_ms": round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                ),
            }

        # ════════════════════════════════════════════════════════════════
        # STEP 3 — MATRIX ROUTING
        # ════════════════════════════════════════════════════════════════
        #
        # Matrix image / matrix subject input has priority over Find-X. The
        # matrix solver receives only the user's text/OCR and computes the
        # result independently. Dataset examples are reference data only.
        # ════════════════════════════════════════════════════════════════

        matrix_text_hint = self._text_explicitly_looks_like_matrix(combined)
        # Two bracketed numeric lists can be X/Y statistical data. Do not
        # treat them as a matrix when the question explicitly asks statistics.
        matrix_route = bool(
            matrix_route_locked
            or (
                matrix_text_hint
                and not self._is_statistics_question(combined)
            )
        )

        if matrix_route:
            entries = dataset_loader.get_subject("matrix")
            matrix_result = self._solve_with_dataset_solver(
                "matrix",
                combined,
                entries,
            )
            if matrix_result is not None:
                matrix_result["image_matrix_detected"] = matrix_image_detected
                matrix_result["matrix_route_locked"] = matrix_route_locked
                matrix_result["ocr_text"] = ocr_text
                matrix_result["input_text"] = combined
                if image_fingerprint:
                    matrix_result["image_fingerprint"] = image_fingerprint
                matrix_result["processing_time_ms"] = round(
                    (time.perf_counter() - started_at) * 1000, 2
                )
                return matrix_result

            # An uploaded image that was routed as matrix must stop here. It is
            # never allowed to continue into Find-X, arithmetic, algorithms, or
            # dataset similarity.
            if matrix_route_locked:
                return self._matrix_image_error(
                    ocr_text=ocr_text,
                    input_text=combined,
                    routing="matrix_image_ocr_uncertain",
                    started_at=started_at,
                    fingerprint=image_fingerprint,
                )

        # FINAL MATRIX IMAGE HARD STOP. This deliberately remains immediately
        # before Find-X so future routing changes cannot accidentally bypass it.
        if matrix_route_locked:
            return self._matrix_image_error(
                ocr_text=ocr_text,
                input_text=combined,
                routing="matrix_image_hard_stop",
                started_at=started_at,
                fingerprint=image_fingerprint,
            )

        # ════════════════════════════════════════════════════════════════
        # STEP 4 — FIND-X ROUTING
        # ════════════════════════════════════════════════════════════════
        #
        # THIS MUST COME BEFORE:
        #
        #     try_arithmetic()
        #
        # and:
        #
        #     solve_with_algorithms()
        #
        # Otherwise a Find-X equation can be incorrectly handled by
        # another mathematical module.
        # ════════════════════════════════════════════════════════════════

        find_x_detected = (
            self._is_find_x_question(
                combined
            )
        )

        if find_x_detected:

            logger.info(
                "Find-X question detected. "
                "Routing to dataset-assisted Find-X solver."
            )

            return self._solve_find_x(
                combined,
                ocr_text,
                started_at
            )

        # ════════════════════════════════════════════════════════════════
        # STEP 4 — INPUT COMPLETENESS GUARD
        # Never borrow missing values/data from a dataset example.
        incomplete_message = self._underspecified_math_message(combined)
        if incomplete_message:
            return {
                "type": "needs_input",
                "message": incomplete_message,
                "subject": self._normalize_subject(subject_hint) or "unknown",
                "input_text": combined,
                "ocr_text": ocr_text,
                "dataset_assisted": False,
                "similarity_answer_copy": False,
                "routing": "input_completeness_guard",
                "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }

        # ════════════════════════════════════════════════════════════════
        # STEP 4 — STATISTICS ROUTING
        # ════════════════════════════════════════════════════════════════
        # Explicit statistics intent must beat generic list/matrix detection.
        # Example: covariance of X=[1,2,3,4] and Y=[2,4,6,8] is two vectors,
        # not a matrix.
        # ════════════════════════════════════════════════════════════════

        if self._is_statistics_question(combined):
            statistics_subject = None
            available_subjects = set(dataset_loader.get_subjects())

            if "statistics" in available_subjects:
                statistics_subject = "statistics"
            elif "mch" in available_subjects:
                statistics_subject = "mch"

            if statistics_subject:
                statistics_entries = dataset_loader.get_subject(statistics_subject)
                statistics_learning = self._get_dataset_learning_context(
                    combined,
                    subject=statistics_subject,
                )
                statistics_result = self._solve_with_dataset_solver(
                    statistics_subject,
                    combined,
                    statistics_entries,
                    learning_context=statistics_learning,
                )

                if statistics_result is not None:
                    statistics_result["ocr_text"] = ocr_text
                    statistics_result["input_text"] = combined
                    statistics_result["routing"] = "statistics_dataset_assisted_solver"
                    statistics_result["statistics_detected"] = True
                    statistics_result["processing_time_ms"] = round(
                        (time.perf_counter() - started_at) * 1000, 2
                    )
                    return statistics_result

                return {
                    "type": "no_match",
                    "message": (
                        "Statistics was detected, but the statistics solver "
                        "could not produce an independent result from the supplied data."
                    ),
                    "subject": statistics_subject,
                    "input_text": combined,
                    "ocr_text": ocr_text,
                    "dataset_entries": len(statistics_entries),
                    "dataset_assisted": bool(statistics_entries),
                    "similarity_answer_copy": False,
                    "routing": "statistics_solver_no_match",
                    "processing_time_ms": round(
                        (time.perf_counter() - started_at) * 1000, 2
                    ),
                }

        # ════════════════════════════════════════════════════════════════
        # STEP 4 — KNN DATASET-LEARNED ROUTING
        # ════════════════════════════════════════════════════════════════
        #
        # KNN is the one module where the dataset examples are actual labeled
        # training/reference points. The solver computes distances from the
        # user's feature vector and predicts by majority vote; it never copies
        # the answer text of the nearest dataset record.
        # ════════════════════════════════════════════════════════════════

        knn_detected = self._is_knn_question(combined)

        if knn_detected:
            knn_entries = dataset_loader.get_subject(KNN_SUBJECT)
            knn_result = self._solve_with_dataset_solver(
                KNN_SUBJECT,
                combined,
                knn_entries,
            )

            if knn_result is not None:
                knn_result["ocr_text"] = ocr_text
                knn_result["input_text"] = combined
                knn_result["routing"] = "knn_dataset_learned_solver"
                knn_result["knn_detected"] = True
                knn_result["processing_time_ms"] = round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                )
                return knn_result

            return {
                "type": "no_match",
                "message": (
                    "KNN was detected, but the KNN solver could not produce "
                    "an independent prediction from the supplied features."
                ),
                "subject": KNN_SUBJECT,
                "input_text": combined,
                "ocr_text": ocr_text,
                "dataset_entries": len(knn_entries),
                "dataset_assisted": bool(knn_entries),
                "similarity_answer_copy": False,
                "routing": "knn_solver_no_match",
                "processing_time_ms": round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                ),
            }

        # ════════════════════════════════════════════════════════════════
        # STEP 4 — TRIGONOMETRY ROUTING
        # ════════════════════════════════════════════════════════════════
        #
        # Trigonometry gets an explicit route before direct arithmetic and
        # before dataset similarity. This prevents a query such as
        # "Find sin 37 degrees" from being treated as a generic expression
        # or answered by the nearest dataset example.
        #
        # solver.solve_trigonometry() uses the user's actual angle/value.
        # Dataset entries are reference/training context only.
        # ════════════════════════════════════════════════════════════════

        trig_detected = self._is_trigonometry_question(combined)

        if trig_detected:
            trig_entries = dataset_loader.get_subject("trigonometry")

            # IMPORTANT:
            # _normalize_natural_language() converts Unicode superscript ² to
            # "^2" for general mathematical parsing. The dedicated
            # trigonometric polynomial solver accepts the canonical "sin²θ"
            # form, so restore superscript notation only for the trig solver.
            # This prevents a quadratic trig equation such as
            # "3sin²θ - 4sinθ + 1 = 0" from being detected but left unsolved.
            trig_question = re.sub(
                r"(?i)(sin|cos|tan|sec|csc|cot)\s*\^\s*2",
                r"\1²",
                combined,
            )

            trig_learning = self._get_dataset_learning_context(
                trig_question,
                subject="trigonometry",
            )
            trig_result = self._solve_with_dataset_solver(
                "trigonometry",
                trig_question,
                trig_entries,
                learning_context=trig_learning,
            )

            if trig_result is not None:
                trig_result["ocr_text"] = ocr_text
                trig_result["input_text"] = combined
                trig_result["routing"] = "trigonometry_solver"
                trig_result["trigonometry_detected"] = True
                trig_result["processing_time_ms"] = round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                )
                return trig_result

            # If the current solver cannot handle this trig form, continue
            # through the normal pipeline rather than borrowing a dataset
            # answer.

        # ════════════════════════════════════════════════════════════════
        # STEP 4.4 — EXPLICIT SYMBOLIC INTENT
        # ════════════════════════════════════════════════════════════════
        # An explicit instruction such as "factor", "differentiate", or
        # "integrate" is stronger evidence than a generic dataset pattern.
        # Handle it before learned routing so a broad matrix pattern cannot
        # turn an algebra request into an unrelated needs-input response.
        symbolic_result = solver.try_symbolic_math(combined)

        if symbolic_result:
            symbolic_result["ocr_text"] = ocr_text
            symbolic_result["input_text"] = combined
            symbolic_result["routing"] = "safe_symbolic_solver"
            symbolic_result["dataset_assisted"] = False
            symbolic_result["similarity_answer_copy"] = False
            symbolic_result["processing_time_ms"] = round(
                (time.perf_counter() - started_at) * 1000,
                2,
            )
            return symbolic_result

        # ════════════════════════════════════════════════════════════════
        # STEP 4.5 — LEARNED METHOD ROUTING
        # ════════════════════════════════════════════════════════════════
        # Use the learned method/formula representation before generic
        # arithmetic/algorithm routing. This is the new learning bridge.
        # The learned representation selects a method; solver.py still
        # computes the final answer from the user's actual values.
        # ════════════════════════════════════════════════════════════════

        learned_context = self._get_dataset_learning_context(combined)
        learned_subject = self._learning_subject_from_context(learned_context)

        arithmetic_preview = solver.try_arithmetic(combined)

        if (
            learned_subject
            and learned_subject not in {FIND_X_SUBJECT, KNN_SUBJECT, "trigonometry"}
            and arithmetic_preview is None
        ):
            learned_entries = dataset_loader.get_subject(learned_subject)
            learned_result = self._solve_with_dataset_solver(
                learned_subject,
                combined,
                learned_entries,
                learning_context=learned_context,
            )
            if learned_result is not None:
                learned_result["learned_method_selection"] = learned_context
                learned_result["dataset_learning_used"] = bool(
                    learned_context.get("enabled")
                )
                learned_result["routing"] = "dataset_learned_method_solver"
                learned_result["ocr_text"] = ocr_text
                learned_result["input_text"] = combined
                learned_result["processing_time_ms"] = round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                )
                return learned_result

        # ════════════════════════════════════════════════════════════════
        # STEP 5 — DIRECT ARITHMETIC
        # ════════════════════════════════════════════════════════════════

        arithmetic_result = solver.try_arithmetic(
            combined
        )

        if arithmetic_result:

            arithmetic_result[
                "subject"
            ] = (
                self._normalize_subject(
                    subject_hint
                )
                or "arithmetic"
            )

            arithmetic_result[
                "ocr_text"
            ] = ocr_text

            arithmetic_result[
                "input_text"
            ] = combined

            arithmetic_result[
                "routing"
            ] = "direct_arithmetic"

            arithmetic_result[
                "processing_time_ms"
            ] = round(
                (
                    time.perf_counter()
                    - started_at
                ) * 1000,
                2
            )

            return arithmetic_result

        # ════════════════════════════════════════════════════════════════
        # STEP 6 — EXPLICIT NUMERICAL ALGORITHM
        # ════════════════════════════════════════════════════════════════

        algorithm_result = solver.solve_with_algorithms(
            combined,
            subject_hint
        )

        if algorithm_result:

            algorithm_result[
                "subject"
            ] = (
                self._normalize_subject(
                    subject_hint
                )
                or "numerical_methods"
            )

            algorithm_result[
                "ocr_text"
            ] = ocr_text

            algorithm_result[
                "input_text"
            ] = combined

            algorithm_result[
                "routing"
            ] = "numerical_algorithm"

            algorithm_result[
                "processing_time_ms"
            ] = round(
                (
                    time.perf_counter()
                    - started_at
                ) * 1000,
                2
            )

            return algorithm_result

        # ════════════════════════════════════════════════════════════════
        # STEP 7 — AUTOMATIC DATASET PATTERN ROUTING
        # ════════════════════════════════════════════════════════════════

        learned_pattern = None if arithmetic_preview is not None else self._automatic_dataset_route(combined)

        if learned_pattern:
            learned_subject = learned_pattern.get(
                "subject"
            )

            if learned_subject:
                learned_entries = dataset_loader.get_subject(
                    learned_subject
                )

                learned_result = self._solve_with_dataset_solver(
                    learned_subject,
                    combined,
                    learned_entries,
                )

                if learned_result is not None:
                    learned_result["learned_pattern"] = (
                        learned_pattern
                    )
                    learned_result["routing"] = (
                        "automatic_dataset_pattern_solver"
                    )
                    learned_result["similarity_answer_copy"] = False
                    learned_result["ocr_text"] = ocr_text
                    learned_result["input_text"] = combined
                    learned_result["processing_time_ms"] = round(
                        (
                            time.perf_counter()
                            - started_at
                        ) * 1000,
                        2,
                    )
                    return learned_result

        # ════════════════════════════════════════════════════════════════
        # STEP 8 — GET LOADED SUBJECTS
        # ════════════════════════════════════════════════════════════════

        loaded_subjects = [
            self._normalize_subject(
                subject
            )

            for subject
            in dataset_loader.get_subjects()
        ]

        loaded_subjects = [
            subject
            for subject
            in loaded_subjects
            if subject
        ]

        if not loaded_subjects:

            return {
                "type": "error",

                "message": (
                    "No mathematical datasets are currently loaded."
                ),

                "ocr_text": ocr_text,

                "input_text": combined,

                "processing_time_ms": round(
                    (
                        time.perf_counter()
                        - started_at
                    ) * 1000,
                    2
                ),
            }

        # ════════════════════════════════════════════════════════════════
        # STEP 9 — SUBJECT CLASSIFICATION
        # ════════════════════════════════════════════════════════════════

        normalized_hint = self._normalize_subject(
            subject_hint
        )

        if (
            normalized_hint
            and normalized_hint in loaded_subjects
        ):

            subject = normalized_hint

            confidence = 1.0

            classification_source = "user_hint"

        else:

            subject, confidence = (
                self._classifier.classify(
                    combined
                )
            )

            subject = (
                self._normalize_subject(
                    subject
                )
                or "unknown"
            )

            confidence = float(
                confidence or 0.0
            )

            classification_source = (
                "classifier"
            )

        logger.info(
            "Classification: subject=%s confidence=%.3f source=%s",
            subject,
            confidence,
            classification_source
        )

        # ════════════════════════════════════════════════════════════════
        # STEP 10 — SUBJECT-SPECIFIC DATASET-ASSISTED SOLVER
        # ════════════════════════════════════════════════════════════════
        #
        # At this point a subject-specific independent solver gets the first
        # chance. Dataset entries are context/training/reference data only.
        # If no independent solver can answer, DO NOT copy a stored dataset
        # answer merely because the question looks similar.
        # ════════════════════════════════════════════════════════════════

        if subject in loaded_subjects:
            entries = dataset_loader.get_subject(subject)
            subject_learning = self._get_dataset_learning_context(
                combined,
                subject=subject,
            )

            solved_result = self._solve_with_dataset_solver(
                subject,
                combined,
                entries,
                learning_context=subject_learning,
            )

            if solved_result is not None:
                solved_result["ocr_text"] = ocr_text
                solved_result["input_text"] = combined
                solved_result["processing_time_ms"] = round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                )
                return solved_result

        # ════════════════════════════════════════════════════════════════
        # STEP 11 — STRICT NO-COPY FALLBACK
        # ════════════════════════════════════════════════════════════════
        #
        # IMPORTANT:
        # Generic nearest-neighbour dataset matching is intentionally NOT
        # used as an answer generator. Returning a stored solution for a
        # different question is exactly the failure mode this engine is
        # designed to prevent.
        # ════════════════════════════════════════════════════════════════

        result = solver.format_no_match(
            combined,
            subject,
        )

        if not isinstance(result, dict):
            result = {
                "type": "no_match",
                "message": (
                    "No independent solver is available for this mathematical "
                    "question."
                ),
            }

        result["type"] = result.get("type") or "no_match"

        # Do not show an unhelpful "Subject: unknown" card for an image-only
        # request.  At this stage OCR did produce text, but no installed
        # independent solver can identify the mathematical operation.  Give
        # the person a clear, actionable image-reading result instead.
        if (
            image_bytes
            and not str(text or "").strip()
            and subject == "unknown"
        ):
            result["type"] = "needs_input"
            result["subject"] = "image_question"
            result["message"] = (
                "I read text from the image, but could not identify a "
                "supported mathematical operation reliably. Please upload a "
                "clearer, tightly cropped image or type the question text."
            )
            result["ocr_guidance"] = (
                "Keep the complete question, symbols and numbers in frame; "
                "avoid shadows, blur and handwriting over printed text."
            )

        result["classification"] = {
            "subject": subject,
            "confidence": round(confidence, 4),
            "source": classification_source,
        }
        result["routing"] = "no_independent_solver"
        result["similarity_answer_copy"] = False
        result["dataset_similarity_used"] = False

        # ════════════════════════════════════════════════════════════════
        # STEP 14 — COMMON RESPONSE METADATA
        # ════════════════════════════════════════════════════════════════
        # STEP 14 — COMMON RESPONSE METADATA
        # ════════════════════════════════════════════════════════════════

        result[
            "ocr_text"
        ] = ocr_text

        result[
            "input_text"
        ] = combined

        result[
            "processing_time_ms"
        ] = round(
            (
                time.perf_counter()
                - started_at
            ) * 1000,
            2
        )

        return result

    # ════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ════════════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        """
        Return current engine and dataset statistics.
        """

        subjects = dataset_loader.get_subjects()

        statistics = {
            "datasets": dataset_loader.dataset_stats(),

            "subjects": subjects,

            "subject_count": len(
                subjects
            ),

            "hot_reload": bool(
                getattr(
                    config,
                    "ENABLE_HOT_RELOAD",
                    False
                )
            ),

            "match_threshold": self._match_threshold(),

            "find_x_loaded": (
                FIND_X_SUBJECT
                in subjects
            ),

            "find_x_entries": len(
                dataset_loader.get_subject(
                    FIND_X_SUBJECT
                )
            ),

            "dataset_learning": getattr(
                dataset_loader,
                "get_dataset_learning_info",
                lambda: {},
            )(),
        }

        integrity_getter = getattr(
            solver,
            "get_exact_dataset_integrity",
            None,
        )
        if callable(integrity_getter):
            integrity = {}
            for subject in subjects:
                try:
                    integrity[subject] = integrity_getter(
                        dataset_loader.get_subject(subject)
                    )
                except Exception as exc:
                    logger.warning(
                        "Dataset integrity audit failed for %s: %s",
                        subject,
                        exc,
                    )
            statistics["dataset_integrity"] = integrity

        if hasattr(
            self._classifier,
            "_active_subjects"
        ):

            try:

                statistics[
                    "classifier_subjects"
                ] = self._classifier._active_subjects()

            except Exception:

                pass

        return statistics


# ════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ════════════════════════════════════════════════════════════════════════════

model = M3Model()
