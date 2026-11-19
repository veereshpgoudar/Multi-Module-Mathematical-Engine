"""
config.py
---------

Central configuration for the Multi-Module Mathematical Engine.
"""

from pathlib import Path
import os


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = Path(
    os.getenv("DATASETS_DIR", str(BASE_DIR / "datasets"))
).expanduser()

FRONTEND_DIR = Path(
    os.getenv("FRONTEND_DIR", str(BASE_DIR / "frontend"))
).expanduser()


# ============================================================
# DATASET SETTINGS
# ============================================================

DATASET_EXTENSIONS = (".json", ".jsonl")

ENABLE_HOT_RELOAD = (
    os.getenv("ENABLE_HOT_RELOAD", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Dataset files are used only for training/reference data,
# pattern learning and method discovery.
#
# Dataset answers must NEVER be copied as the final answer
# for a new/unseen question.
DATASET_LEARNING_ENABLED = (
    os.getenv("DATASET_LEARNING_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

DATASET_ANSWER_COPY_ALLOWED = False

DATASET_DEDUPLICATE_QUESTIONS = (
    # Keep every supplied training row by default.  Repeated examples can be
    # meaningful training evidence, and the loader reports the true dataset
    # size when it has not been explicitly asked to deduplicate.
    os.getenv("DATASET_DEDUPLICATE_QUESTIONS", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

MAX_DATASET_RECORDS_PER_SUBJECT = int(
    os.getenv("MAX_DATASET_RECORDS_PER_SUBJECT", "50000")
)


# ============================================================
# CLASSIFICATION / MATCHING
# ============================================================

MATCH_THRESHOLD = float(
    os.getenv("MATCH_THRESHOLD", "0.28")
)

CLASSIFICATION_MARGIN = float(
    os.getenv("CLASSIFICATION_MARGIN", "0.08")
)


# ============================================================
# KNN SETTINGS
# ============================================================

# KNN predicts the label from numeric features in the
# training dataset. It does not copy the answer field.
KNN_DEFAULT_K = int(
    os.getenv("KNN_DEFAULT_K", "5")
)

KNN_MAX_K = int(
    os.getenv("KNN_MAX_K", "25")
)

KNN_DISTANCE_METRIC = (
    os.getenv("KNN_DISTANCE_METRIC", "euclidean")
    .strip()
    .lower()
)

KNN_MIN_CONFIDENCE = float(
    os.getenv("KNN_MIN_CONFIDENCE", "0.50")
)


# ============================================================
# OCR SETTINGS
# ============================================================

# Can be overridden through the TESSERACT_CMD environment variable.
# The default path is the normal Windows installation location.
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
).strip()

OCR_RESIZE_FACTOR = float(
    os.getenv("OCR_RESIZE_FACTOR", "2.5")
)

# Tesseract accepts one or more trained-data codes joined with ``+``. Kannada
# is included by default; the OCR layer uses the installed subset when the
# optional ``kan`` trained-data file is not available.
OCR_LANG = os.getenv(
    "OCR_LANG", "eng+kan"
).strip()

# The generic reader uses a compact, confidence-ranked set of OCR attempts.
# This is intentionally much faster than trying every PSM over every rotation.
OCR_MIN_TOKEN_CONFIDENCE = float(
    os.getenv("OCR_MIN_TOKEN_CONFIDENCE", "35")
)

OCR_MIN_DOCUMENT_CONFIDENCE = float(
    os.getenv("OCR_MIN_DOCUMENT_CONFIDENCE", "45")
)

OCR_MAX_DIMENSION = int(
    os.getenv("OCR_MAX_DIMENSION", "2400")
)

OCR_ENABLE_DESKEW = (
    os.getenv("OCR_ENABLE_DESKEW", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# The document reader starts with the orientation reported by Tesseract OSD.
# If that pass is weak, it automatically checks the other right-angle
# rotations.  This gives rotated phone images a reliable recovery path while
# keeping normal images fast.
OCR_ENABLE_ORIENTATION_FALLBACK = (
    os.getenv("OCR_ENABLE_ORIENTATION_FALLBACK", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

OCR_ORIENTATION_RETRY_CONFIDENCE = float(
    os.getenv("OCR_ORIENTATION_RETRY_CONFIDENCE", "58")
)

OCR_MAX_ORIENTATION_ATTEMPTS = int(
    os.getenv("OCR_MAX_ORIENTATION_ATTEMPTS", "4")
)


# ============================================================
# MATRIX OCR
# ============================================================

# Matrix OCR is handled separately from generic OCR so that
# matrix images do not accidentally enter Find-X or other solvers.
MATRIX_OCR_ENABLED = (
    os.getenv("MATRIX_OCR_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# If matrix OCR is not sufficiently confident, return needs_input
# instead of guessing with another mathematical solver.
MATRIX_OCR_CONFIDENCE_THRESHOLD = float(
    os.getenv("MATRIX_OCR_CONFIDENCE_THRESHOLD", "0.60")
)


# ============================================================
# FLASK API SETTINGS
# ============================================================

API_HOST = os.getenv(
    "API_HOST", "127.0.0.1"
).strip()

API_PORT = int(
    os.getenv("API_PORT", "5000")
)

API_DEBUG = (
    os.getenv("API_DEBUG", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)


# ============================================================
# REQUEST / UPLOAD SETTINGS
# ============================================================

MAX_CONTENT_LENGTH = int(
    os.getenv(
        "MAX_CONTENT_LENGTH",
        str(10 * 1024 * 1024)
    )
)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


# ============================================================
# CORS
# ============================================================

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "*"
).strip()


# ============================================================
# SOLVER SETTINGS
# ============================================================

DEFAULT_TOLERANCE = float(
    os.getenv("DEFAULT_TOLERANCE", "1e-8")
)

DEFAULT_MAX_ITERATIONS = int(
    os.getenv("DEFAULT_MAX_ITERATIONS", "100")
)

DEFAULT_STEP_SIZE = float(
    os.getenv("DEFAULT_STEP_SIZE", "0.01")
)


# ============================================================
# INDEPENDENT-SOLVER / PROVENANCE POLICY
# ============================================================

# When enabled, dataset records may teach the system patterns,
# methods and structure, but the final result must be calculated
# by an independent mathematical solver.
REQUIRE_INDEPENDENT_SOLVER = (
    os.getenv("REQUIRE_INDEPENDENT_SOLVER", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

INCLUDE_SOLVER_PROVENANCE = (
    os.getenv("INCLUDE_SOLVER_PROVENANCE", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

MATCH_CACHE_SIZE = int(
    os.getenv("MATCH_CACHE_SIZE", "512")
)

MAX_TEXT_LENGTH = int(
    os.getenv("MAX_TEXT_LENGTH", "5000")
)


# ============================================================
# APPLICATION INFORMATION
# ============================================================

APP_NAME = "Math Mind"

APP_VERSION = "2.1.0"

APP_DESCRIPTION = (
    "Multi-Module Mathematical Engine with "
    "classification, dataset pattern learning, OCR, "
    "independent mathematical solving and numerical algorithms."
)


# ============================================================
# SUBJECTS
# ============================================================

# Keep each subject exactly once.
#
# IMPORTANT:
# "statistics" is explicitly separated from "matrix" so that
# questions such as:
#   Find mean of [1,2,3,4]
#   Find variance of [1,2,3,4]
#   Find covariance of X and Y
# are not treated as matrix questions.
SUPPORTED_SUBJECTS = (
    "algebra",
    "calculus",
    "statistics",
    "trigonometry",
    "find_x",
    "matrix",
    "knn",
    "mch",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_directories():
    """Create required project directories if they do not exist."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


def get_config():
    """Return the main configuration as a dictionary."""
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,

        "base_dir": str(BASE_DIR),
        "datasets_dir": str(DATASETS_DIR),
        "frontend_dir": str(FRONTEND_DIR),

        "hot_reload": ENABLE_HOT_RELOAD,

        # Dataset learning policy
        "dataset_learning_enabled": DATASET_LEARNING_ENABLED,
        "dataset_answer_copy_allowed": DATASET_ANSWER_COPY_ALLOWED,
        "dataset_deduplicate_questions": DATASET_DEDUPLICATE_QUESTIONS,
        "max_dataset_records_per_subject":
            MAX_DATASET_RECORDS_PER_SUBJECT,

        # Classification
        "match_threshold": MATCH_THRESHOLD,
        "classification_margin": CLASSIFICATION_MARGIN,

        # KNN
        "knn_default_k": KNN_DEFAULT_K,
        "knn_max_k": KNN_MAX_K,
        "knn_distance_metric": KNN_DISTANCE_METRIC,
        "knn_min_confidence": KNN_MIN_CONFIDENCE,

        # OCR
        "tesseract_cmd": TESSERACT_CMD,
        "ocr_language": OCR_LANG,
        "ocr_resize_factor": OCR_RESIZE_FACTOR,
        "ocr_min_token_confidence": OCR_MIN_TOKEN_CONFIDENCE,
        "ocr_min_document_confidence": OCR_MIN_DOCUMENT_CONFIDENCE,
        "ocr_max_dimension": OCR_MAX_DIMENSION,
        "ocr_enable_deskew": OCR_ENABLE_DESKEW,
        "ocr_enable_orientation_fallback": OCR_ENABLE_ORIENTATION_FALLBACK,
        "ocr_orientation_retry_confidence": OCR_ORIENTATION_RETRY_CONFIDENCE,
        "ocr_max_orientation_attempts": OCR_MAX_ORIENTATION_ATTEMPTS,

        # Matrix OCR
        "matrix_ocr_enabled": MATRIX_OCR_ENABLED,
        "matrix_ocr_confidence_threshold":
            MATRIX_OCR_CONFIDENCE_THRESHOLD,

        # API
        "api_host": API_HOST,
        "api_port": API_PORT,
        "debug": API_DEBUG,

        # Upload
        "max_content_length": MAX_CONTENT_LENGTH,
        "allowed_image_types": sorted(ALLOWED_IMAGE_TYPES),

        # Solver
        "default_tolerance": DEFAULT_TOLERANCE,
        "default_max_iterations": DEFAULT_MAX_ITERATIONS,
        "default_step_size": DEFAULT_STEP_SIZE,

        # Independent solver
        "require_independent_solver":
            REQUIRE_INDEPENDENT_SOLVER,
        "include_solver_provenance":
            INCLUDE_SOLVER_PROVENANCE,

        # Performance
        "match_cache_size": MATCH_CACHE_SIZE,
        "max_text_length": MAX_TEXT_LENGTH,

        # Subjects
        "supported_subjects": list(SUPPORTED_SUBJECTS),
    }


# ============================================================
# STARTUP VALIDATION
# ============================================================

def validate_config():
    """Validate important configuration values."""

    if not 0.0 <= MATCH_THRESHOLD <= 1.0:
        raise ValueError(
            "MATCH_THRESHOLD must be between 0 and 1."
        )

    if not 0.0 <= CLASSIFICATION_MARGIN <= 1.0:
        raise ValueError(
            "CLASSIFICATION_MARGIN must be between 0 and 1."
        )

    if OCR_RESIZE_FACTOR <= 0:
        raise ValueError(
            "OCR_RESIZE_FACTOR must be greater than 0."
        )

    if not 0.0 <= OCR_MIN_TOKEN_CONFIDENCE <= 100.0:
        raise ValueError(
            "OCR_MIN_TOKEN_CONFIDENCE must be between 0 and 100."
        )

    if not 0.0 <= OCR_MIN_DOCUMENT_CONFIDENCE <= 100.0:
        raise ValueError(
            "OCR_MIN_DOCUMENT_CONFIDENCE must be between 0 and 100."
        )

    if OCR_MAX_DIMENSION < 320:
        raise ValueError(
            "OCR_MAX_DIMENSION must be at least 320."
        )

    if not 0.0 <= OCR_ORIENTATION_RETRY_CONFIDENCE <= 100.0:
        raise ValueError(
            "OCR_ORIENTATION_RETRY_CONFIDENCE must be between 0 and 100."
        )

    if not 1 <= OCR_MAX_ORIENTATION_ATTEMPTS <= 4:
        raise ValueError(
            "OCR_MAX_ORIENTATION_ATTEMPTS must be between 1 and 4."
        )

    if not 0.0 <= MATRIX_OCR_CONFIDENCE_THRESHOLD <= 1.0:
        raise ValueError(
            "MATRIX_OCR_CONFIDENCE_THRESHOLD "
            "must be between 0 and 1."
        )

    if MAX_CONTENT_LENGTH <= 0:
        raise ValueError(
            "MAX_CONTENT_LENGTH must be greater than 0."
        )

    if API_PORT < 1 or API_PORT > 65535:
        raise ValueError(
            "API_PORT must be between 1 and 65535."
        )

    if DEFAULT_TOLERANCE <= 0:
        raise ValueError(
            "DEFAULT_TOLERANCE must be greater than 0."
        )

    if DEFAULT_MAX_ITERATIONS <= 0:
        raise ValueError(
            "DEFAULT_MAX_ITERATIONS must be greater than 0."
        )

    if DEFAULT_STEP_SIZE <= 0:
        raise ValueError(
            "DEFAULT_STEP_SIZE must be greater than 0."
        )

    if MATCH_CACHE_SIZE <= 0:
        raise ValueError(
            "MATCH_CACHE_SIZE must be greater than 0."
        )

    if MAX_TEXT_LENGTH <= 0:
        raise ValueError(
            "MAX_TEXT_LENGTH must be greater than 0."
        )

    if MAX_DATASET_RECORDS_PER_SUBJECT <= 0:
        raise ValueError(
            "MAX_DATASET_RECORDS_PER_SUBJECT "
            "must be greater than 0."
        )

    if KNN_DEFAULT_K <= 0:
        raise ValueError(
            "KNN_DEFAULT_K must be greater than 0."
        )

    if KNN_MAX_K <= 0:
        raise ValueError(
            "KNN_MAX_K must be greater than 0."
        )

    if KNN_MAX_K < KNN_DEFAULT_K:
        raise ValueError(
            "KNN_MAX_K must be greater than or equal "
            "to KNN_DEFAULT_K."
        )

    if KNN_DISTANCE_METRIC != "euclidean":
        raise ValueError(
            "KNN_DISTANCE_METRIC currently supports "
            "only 'euclidean'."
        )

    if not 0.0 <= KNN_MIN_CONFIDENCE <= 1.0:
        raise ValueError(
            "KNN_MIN_CONFIDENCE must be between 0 and 1."
        )

    if not OCR_LANG:
        raise ValueError(
            "OCR_LANG must not be empty."
        )

    if not API_HOST:
        raise ValueError(
            "API_HOST must not be empty."
        )

    if not SUPPORTED_SUBJECTS:
        raise ValueError(
            "SUPPORTED_SUBJECTS must not be empty."
        )

    if len(set(SUPPORTED_SUBJECTS)) != len(SUPPORTED_SUBJECTS):
        raise ValueError(
            "SUPPORTED_SUBJECTS contains duplicate subjects."
        )

    ensure_directories()


# ============================================================
# INITIALIZE
# ============================================================

validate_config()
