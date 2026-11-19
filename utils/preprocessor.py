"""
M3_ENGINE — Preprocessor
Reliable OCR pipeline for mathematical questions.

Features:
• Cleans and normalises raw question text
• Extracts mathematical text from uploaded images
• Automatically detects Tesseract on Windows/Linux
• Uses multiple Tesseract PSM modes
• Uses multiple image preprocessing strategies
• Automatically handles 0°/90°/180°/270° rotated images
• Handles simple expressions such as 2+2
• Corrects common OCR mathematical mistakes
"""

import re
import io
import os
import shutil
import logging
from typing import Optional

import config

try:
    import cv2
    import numpy as np
    _CV2_OK = True
except ImportError:
    cv2 = None
    np = None
    _CV2_OK = False

logger = logging.getLogger(__name__)


# ============================================================================
# OPTIONAL IMPORTS
# ============================================================================

try:
    import pytesseract

    _TESSERACT_OK = True

except ImportError:

    pytesseract = None
    _TESSERACT_OK = False

    logger.warning(
        "pytesseract is not installed. OCR will be disabled."
    )


try:
    from PIL import (
        Image,
        ImageOps,
        ImageFilter,
        ImageEnhance,
    )

    _PIL_OK = True

except ImportError:

    Image = None
    ImageOps = None
    ImageFilter = None
    ImageEnhance = None

    _PIL_OK = False

    logger.warning(
        "Pillow is not installed. OCR preprocessing will be disabled."
    )


# ============================================================================
# TESSERACT DETECTION
# ============================================================================

def _find_tesseract() -> Optional[str]:
    """
    Find the Tesseract executable automatically.

    Checks:
    1. config.TESSERACT_CMD
    2. System PATH
    3. Common Windows installation locations
    """

    configured = getattr(
        config,
        "TESSERACT_CMD",
        ""
    )

    configured = str(
        configured or ""
    ).strip()


    # ------------------------------------------------------------------------
    # 1. Configured path
    # ------------------------------------------------------------------------

    if configured:

        if os.path.isfile(
            configured
        ):

            return configured


        found = shutil.which(
            configured
        )

        if found:

            return found


    # ------------------------------------------------------------------------
    # 2. System PATH
    # ------------------------------------------------------------------------

    found = shutil.which(
        "tesseract"
    )

    if found:

        return found


    # ------------------------------------------------------------------------
    # 3. Common Windows installation paths
    # ------------------------------------------------------------------------

    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
        ),
    ]


    for path in windows_paths:

        if os.path.isfile(
            path
        ):

            return path


    return None


# ============================================================================
# CONFIGURE TESSERACT
# ============================================================================

if _TESSERACT_OK:

    _TESSERACT_PATH = _find_tesseract()

    if _TESSERACT_PATH:

        pytesseract.pytesseract.tesseract_cmd = (
            _TESSERACT_PATH
        )

        logger.info(
            "Tesseract found at: %s",
            _TESSERACT_PATH
        )

    else:

        logger.error(
            "Tesseract executable was not found."
        )

        logger.error(
            "Install Tesseract OCR or set TESSERACT_CMD in config.py."
        )

        _TESSERACT_OK = False


# ============================================================================
# OCR CHARACTER NORMALISATION
# ============================================================================

_OCR_FIX = [

    # Multiplication symbols
    (r"×", "*"),
    (r"✕", "*"),
    (r"⋅", "*"),
    (r"·", "*"),

    # Division
    (r"÷", "/"),

    # Minus symbols
    (r"−", "-"),
    (r"–", "-"),
    (r"—", "-"),
    (r"﹣", "-"),

    # Superscripts
    (r"²", "^2"),
    (r"³", "^3"),
    (r"⁴", "^4"),
    (r"⁵", "^5"),

    # Square root
    (r"√", "sqrt"),

    # Infinity
    (r"∞", "infinity"),

    # Degree / prime / common mathematical punctuation
    (r"º", "°"),
    (r"˚", "°"),
    (r"′", "'"),
    (r"’", "'"),
    (r"“", '"'),
    (r"”", '"'),

    # Greek/math symbols used heavily in trig and calculus. Preserve them
    # instead of allowing OCR cleanup to erase the mathematical meaning.
    (r"theta", "θ"),
    (r"Θ", "θ"),
    (r"π", "π"),
    (r"Pi", "pi"),
    (r"PI", "pi"),

    # Common Unicode fractions
    (r"½", "1/2"),
    (r"⅓", "1/3"),
    (r"⅔", "2/3"),
    (r"¼", "1/4"),
    (r"¾", "3/4"),
    (r"⅕", "1/5"),
    (r"⅖", "2/5"),
    (r"⅗", "3/5"),
    (r"⅘", "4/5"),
    (r"⅙", "1/6"),
    (r"⅚", "5/6"),
    (r"⅛", "1/8"),
    (r"⅜", "3/8"),
    (r"⅝", "5/8"),
    (r"⅞", "7/8"),

    # Common OCR confusions in mathematical text.  Keep these
    # deliberately conservative so ordinary words are not damaged.
    (r"(?i)\bcosine\b", "cos"),
    (r"(?i)\bsine\b", "sin"),
    (r"(?i)\btangent\b", "tan"),
    (r"(?i)\bsecant\b", "sec"),
    (r"(?i)\bcosecant\b", "csc"),
    (r"(?i)\bcotangent\b", "cot"),

    # OCR zero / letter corrections
    (r"\bO(?=\d)", "0"),
    (r"(?<=\d)O\b", "0"),

    # OCR one / lowercase L corrections
    (r"\bl(?=\d)", "1"),
    (r"(?<=\d)l\b", "1"),

    # Collapse spaces
    (r"\s+", " "),
]


# ============================================================================
# MATH KEYWORDS
# ============================================================================

_MATH_KEYWORDS_RE = re.compile(
    r"(solve|find|compute|calculate|evaluate|simplify|"
    r"differentiate|integrate|prove|determine|given|"
    r"equation|value|factor|expand|derivative|integral|"
    r"covariance|variance|correlation|mean|median|"
    r"standard\s+deviation|"
    r"trigonometry|trigonometric|"
    r"sin|cos|tan|sine|cosine|tangent|"
    r"sec|csc|cot|arcsin|arccos|arctan|"
    r"radian|radians|degree|degrees|"
    r"law\s+of\s+(?:sines|cosines)|"
    r"knn|k\s*[- ]?nearest|nearest\s+neighbor|nearest\s+neighbour|"
    r"euclidean\s+distance|majority\s+vote)",
    re.IGNORECASE,
)


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(
    text: str
) -> str:
    """
    Clean and normalise OCR text.
    """

    if not text:

        return ""


    text = str(
        text
    )

    max_length = int(
        getattr(config, "MAX_TEXT_LENGTH", 5000)
    )
    if max_length > 0:
        text = text[:max_length]


    # Apply OCR corrections
    for pattern, replacement in _OCR_FIX:

        text = re.sub(
            pattern,
            replacement,
            text
        )


    # Remove non-printable characters
    text = "".join(
        ch
        for ch in text
        if ch.isprintable() or ch == "\n"
    )


    # Normalise spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # Normalise blank lines
    text = re.sub(
        r"\n{2,}",
        "\n",
        text
    )


    return text.strip()


# ============================================================================
# MATH SENTENCE EXTRACTION
# ============================================================================

def extract_math_sentences(
    text: str
) -> str:
    """
    Extract lines that look like mathematical questions.
    """

    if not text:

        return ""


    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


    if not lines:

        return ""


    candidates = []


    for line in lines:

        has_keyword = bool(
            _MATH_KEYWORDS_RE.search(
                line
            )
        )


        has_equation = (
            "=" in line
        )


        has_math_operator = any(
            operator in line
            for operator in [
                "+",
                "-",
                "*",
                "/",
                "^",
                "√",
                "×",
                "÷",
            ]
        )


        has_number = bool(
            re.search(
                r"\d",
                line
            )
        )


        if (
            has_keyword
            or has_equation
            or (
                has_math_operator
                and has_number
            )
        ):

            candidates.append(
                line
            )


    if candidates:

        return " ".join(
            candidates
        )


    return " ".join(
        lines
    )


# ============================================================================
# DATASET-LEARNING FEATURE EXTRACTION
# ============================================================================
#
# Preprocessing is intentionally NOT a model-training layer. Its job is to make
# raw/OCR text stable enough that dataset_loader.py and classifier.py can learn
# problem patterns consistently. These features contain no dataset answers.
# ============================================================================

_LEARNING_PATTERN_RULES = {
    "law_of_cosines": (
        r"\blaw\s+of\s+cosines?\b",
        r"\bincluded\s+angle\b",
        r"\bthird\s+side\b",
    ),
    "law_of_sines": (
        r"\blaw\s+of\s+sines?\b",
        r"\bsine\s+rule\b",
    ),
    "right_triangle": (
        r"\bright[- ]?angled?\s+triangle\b",
        r"\bhypotenuse\b",
        r"\bopposite\b.*\badjacent\b",
    ),
    "trig_identity": (
        r"\btrigonometric\s+identity\b",
        r"\btrig\s+identity\b",
        r"\bsimplify\b.*\b(?:sin|cos|tan|sec|csc|cot)\b",
    ),
    "inverse_trig": (
        r"\barc(?:sin|cos|tan)\b",
        r"\binverse\s+(?:trig|trigonometry)\b",
    ),
    "degree_radian_conversion": (
        r"\b(?:degree|degrees)\b.*\b(?:radian|radians)\b",
        r"\b(?:radian|radians)\b.*\b(?:degree|degrees)\b",
        r"\bto\s+(?:degrees?|radians?)\b",
    ),
    "trig_equation": (
        r"\bsolve\b.*(?:sin|cos|tan)",
        r"\b(?:sin|cos|tan)[^=]{0,80}=",
    ),
    "quadratic_equation": (
        r"\bquadratic\b",
        r"\bx\s*(?:\^|\*\*)\s*2\b",
        r"\bdiscriminant\b",
    ),
    "linear_equation": (
        r"\bsolve\s+for\s+x\b",
        r"\bfind\s+(?:the\s+)?(?:value\s+of\s+)?x\b",
        r"(?:^|[\s:])x\s*[^=]{0,40}=",
    ),
    "matrix_operation": (
        r"\bmatrix\b",
        r"\bmatrices\b",
        r"\bdeterminant\b",
        r"\binverse\s+(?:of\s+)?matrix\b",
        r"\btranspose\b",
    ),
    "knn_classification": (
        r"\bknn\b",
        r"\bk[- ]?nearest\b",
        r"\bnearest\s+neighbor\b",
        r"\beuclidean\s+distance\b",
        r"\bmajority\s+vote\b",
    ),
    "statistics_xy": (
        r"\bcovariance\b",
        r"\bcorrelation\b",
        r"\bvariance\b",
        r"\bstandard\s+deviation\b",
    ),
}


def extract_learning_features(text: str) -> dict:
    """
    Extract answer-free mathematical features from cleaned/OCR text.

    These features are intended for the dataset-learning/classification layer:
    problem-pattern hints, operations, number count, variables, and math terms.
    No dataset answer or stored solution is consulted.
    """
    cleaned = clean_text(text or "")
    normalised = cleaned.lower()
    normalised = normalised.replace("²", "^2").replace("³", "^3")
    normalised = normalised.replace("−", "-").replace("×", "*").replace("÷", "/")

    patterns = []
    for name, rules in _LEARNING_PATTERN_RULES.items():
        if any(re.search(rule, normalised, flags=re.IGNORECASE) for rule in rules):
            patterns.append(name)

    number_tokens = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", normalised)
    variable_tokens = sorted(set(re.findall(r"\b[a-zA-Z]\b", cleaned)))

    math_terms = sorted(set(re.findall(
        r"\b(?:sin|cos|tan|sec|csc|cot|sine|cosine|tangent|"
        r"arcsin|arccos|arctan|radian|radians|degree|degrees|"
        r"matrix|matrices|determinant|inverse|transpose|rank|"
        r"mean|median|mode|variance|covariance|correlation|"
        r"derivative|integral|equation|quadratic|knn)\b",
        normalised,
        flags=re.IGNORECASE,
    )))

    operators = sorted(set(re.findall(r"[+\-*/=^]", normalised)))

    return {
        "clean_text": cleaned,
        "patterns": patterns or ["general_math"],
        "number_count": len(number_tokens),
        "variable_tokens": variable_tokens,
        "math_terms": math_terms,
        "operators": operators,
        "has_equation": "=" in cleaned,
        "has_degree_symbol": "°" in cleaned,
        "has_pi": "π" in cleaned.lower() or "pi" in normalised,
    }


def prepare_for_dataset_learning(text: str) -> dict:
    """
    Public helper used by higher layers before dataset pattern learning.

    The output is deliberately answer-free so the learning pipeline cannot
    accidentally treat a stored dataset solution as the user's answer.
    """
    features = extract_learning_features(text)
    return {
        "learning_ready_text": features["clean_text"],
        "pattern_hints": features["patterns"],
        "feature_count": (
            features["number_count"]
            + len(features["variable_tokens"])
            + len(features["math_terms"])
            + len(features["operators"])
        ),
        "features": features,
        "answer_from_dataset": None,
        "solution_from_dataset": None,
        "answer_copy_allowed": False,
    }


# ============================================================================
# IMAGE PREPARATION
# ============================================================================

def _prepare_base_image(
    img
):
    """
    Prepare image for OCR.
    """

    if not _PIL_OK:

        return img


    # Convert unsupported modes
    if img.mode not in (
        "L",
        "RGB"
    ):

        img = img.convert(
            "RGB"
        )


    # ------------------------------------------------------------------------
    # Upscaling
    # ------------------------------------------------------------------------

    resize_factor = float(
        getattr(
            config,
            "OCR_RESIZE_FACTOR",
            3.0
        )
    )


    # Small screenshots benefit from stronger upscaling
    if (
        img.width < 500
        or img.height < 300
    ):

        resize_factor = max(
            resize_factor,
            4.0
        )


    new_width = max(
        1,
        int(
            img.width *
            resize_factor
        )
    )


    new_height = max(
        1,
        int(
            img.height *
            resize_factor
        )
    )


    img = img.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


    return img


# ============================================================================
# CREATE OCR VARIANTS
# ============================================================================

def _preprocess_variants(
    img
):
    """
    Create OCR variants for multiple image orientations.

    This improves OCR reliability for:
    • normal screenshots
    • rotated screenshots
    • white backgrounds
    • black text
    • low-resolution images
    • mathematical expressions

    Four orientations are tried automatically: 0°, 90°, 180° and 270°.
    This is important for phone screenshots where a mathematical expression
    may be saved sideways.
    """

    if not _PIL_OK:

        return [
            img
        ]


    # ------------------------------------------------------------------------
    # Try all common orientations.
    # PIL rotate(90) is counter-clockwise; 270 gives the opposite direction.
    # ------------------------------------------------------------------------
    orientations = [
        (0, img),
        (90, img.rotate(90, expand=True)),
        (180, img.rotate(180, expand=True)),
        (270, img.rotate(270, expand=True)),
    ]

    variants = []

    for angle, oriented in orientations:

        base = _prepare_base_image(
            oriented
        )

        # ====================================================================
        # Variant 1 — grayscale + contrast + sharpness
        # ====================================================================

        gray = base.convert(
            "L"
        )

        gray = ImageEnhance.Contrast(
            gray
        ).enhance(
            2.5
        )

        gray = ImageEnhance.Sharpness(
            gray
        ).enhance(
            2.0
        )

        gray = gray.filter(
            ImageFilter.SHARPEN
        )

        # ====================================================================
        # Variant 2 — auto contrast
        # ====================================================================

        auto = ImageOps.autocontrast(
            gray
        )

        # ====================================================================
        # Variant 3 — normal threshold
        # ====================================================================

        threshold = auto.point(
            lambda p:
                255 if p > 150
                else 0
        )

        # ====================================================================
        # Variant 4 — strong threshold
        # ====================================================================

        threshold_strong = auto.point(
            lambda p:
                255 if p > 190
                else 0
        )

        # ====================================================================
        # Variant 5 — inverted threshold
        # ====================================================================

        inverted = ImageOps.invert(
            threshold
        )

        variants.extend([
            gray,
            auto,
            threshold,
            threshold_strong,
            inverted,
        ])

        logger.debug(
            "Prepared OCR orientation: %s degrees",
            angle
        )

    return variants


    # ========================================================================
    # Variant 1
    # Grayscale + contrast + sharpness
    # ========================================================================

    gray = base.convert(
        "L"
    )


    gray = ImageEnhance.Contrast(
        gray
    ).enhance(
        2.5
    )


    gray = ImageEnhance.Sharpness(
        gray
    ).enhance(
        2.0
    )


    gray = gray.filter(
        ImageFilter.SHARPEN
    )


    # ========================================================================
    # Variant 2
    # Auto contrast
    # ========================================================================

    auto = ImageOps.autocontrast(
        gray
    )


    # ========================================================================
    # Variant 3
    # Threshold
    # ========================================================================

    threshold = auto.point(
        lambda p:
            255 if p > 150
            else 0
    )


    # ========================================================================
    # Variant 4
    # Strong threshold
    # ========================================================================

    threshold_strong = auto.point(
        lambda p:
            255 if p > 190
            else 0
    )


    # ========================================================================
    # Variant 5
    # Inverted threshold
    # ========================================================================

    inverted = ImageOps.invert(
        threshold
    )


    return [
        gray,
        auto,
        threshold,
        threshold_strong,
        inverted,
    ]


# ============================================================================
# TESSERACT
# ============================================================================

def _run_tesseract(
    image,
    psm: int
) -> str:
    """
    Run one Tesseract OCR attempt.
    """

    if not _TESSERACT_OK:

        return ""


    try:

        config_string = (
            f"--oem 3 --psm {psm}"
        )


        language = getattr(
            config,
            "OCR_LANG",
            "eng"
        )


        text = (
            pytesseract.image_to_string(
                image,
                lang=language,
                config=config_string
            )
        )


        return text or ""


    except Exception as exc:

        logger.error(
            "Tesseract PSM %s failed: %s",
            psm,
            exc
        )

        return ""


# ============================================================================
# OCR QUALITY SCORE
# ============================================================================

def _score_ocr_text(
    text: str
) -> float:
    """
    Score OCR result.

    Mathematical expressions receive higher scores
    when they contain numbers and operators.
    """

    if not text:

        return 0.0


    stripped = text.strip()


    if not stripped:

        return 0.0


    score = 0.0


    # ------------------------------------------------------------------------
    # Numbers
    # ------------------------------------------------------------------------

    if re.search(
        r"\d",
        stripped
    ):

        score += 3.0


    # ------------------------------------------------------------------------
    # Mathematical operators
    # ------------------------------------------------------------------------

    if re.search(
        r"[+\-*/=^]",
        stripped
    ):

        score += 3.0


    # ------------------------------------------------------------------------
    # Parentheses
    # ------------------------------------------------------------------------

    if re.search(
        r"[()]",
        stripped
    ):

        score += 1.0


    # ------------------------------------------------------------------------
    # Alphabetic characters
    # ------------------------------------------------------------------------

    if re.search(
        r"[A-Za-z]",
        stripped
    ):

        score += 1.0


    # ------------------------------------------------------------------------
    # Reasonable text length
    # ------------------------------------------------------------------------

    if 1 <= len(
        stripped
    ) <= 500:

        score += 1.0


    # ------------------------------------------------------------------------
    # Penalise strange characters
    # ------------------------------------------------------------------------

    weird_count = len(
        re.findall(
            r"[^A-Za-z0-9\s+\-*/=().,^]",
            stripped
        )
    )


    if weird_count:

        score -= min(
            weird_count * 0.5,
            3.0
        )


    # ------------------------------------------------------------------------
    # Prefer short clean math expressions
    # ------------------------------------------------------------------------

    if re.fullmatch(
        r"[0-9A-Za-z\s+\-*/=().,^]+",
        stripped
    ):

        score += 2.0


    return score


# ============================================================================
# MATH EXPRESSION CLEANUP
# ============================================================================

def _normalise_math_expression(
    text: str
) -> str:
    """
    Additional cleanup specifically for mathematical expressions.
    """

    if not text:

        return ""


    text = text.strip()


    # Conservative OCR pipe correction.  A pipe can be a matrix/vector
    # delimiter, so do not globally replace every `|` with `1`.
    text = re.sub(r"(?<=\d)\|(?=\d)", "1", text)
    text = re.sub(r"(?<=\d)\|\b", "1", text)
    text = re.sub(r"\b\|(?=\d)", "1", text)


    # Remove spaces around operators
    text = re.sub(
        r"\s*([+\-*/=])\s*",
        r" \1 ",
        text
    )

    # A hyphen joining words is not a subtraction sign.  Keep terms such as
    # k-NN, right-angled and data-set intact after operator spacing.
    text = re.sub(r"(?<=[A-Za-z])\s+-\s+(?=[A-Za-z])", "-", text)


    # Collapse whitespace again
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================================
# OCR CANDIDATE CONSENSUS
# ============================================================================

def _canonical_ocr_text(text: str) -> str:
    """Create a comparison form for OCR outputs without changing the answer text."""
    value = clean_text(text)
    value = _normalise_math_expression(value)
    value = value.lower()
    value = re.sub(r"\s+", "", value)
    return value


def _candidate_consensus_score(candidate, all_candidates) -> float:
    """Reward OCR results independently reproduced by other passes."""
    score, text, _, _ = candidate
    key = _canonical_ocr_text(text)
    if not key:
        return score

    agreements = 0
    for other in all_candidates:
        if _canonical_ocr_text(other[1]) == key:
            agreements += 1

    # Agreement is deliberately a bonus, not a replacement for the
    # mathematical quality score.
    return score + min(agreements - 1, 6) * 1.5


def _select_best_ocr_candidate(candidates):
    """Select the strongest OCR result using quality + independent consensus."""
    if not candidates:
        return None

    ranked = [
        (_candidate_consensus_score(candidate, candidates), candidate)
        for candidate in candidates
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


# ============================================================================
# MAIN OCR
# ============================================================================

def ocr_from_bytes(
    image_bytes: bytes
) -> Optional[str]:
    """
    Run reliable OCR on raw image bytes.

    Returns:
        Extracted mathematical text
        or None when OCR fails.
    """

    if not image_bytes:

        logger.error(
            "OCR received empty image bytes."
        )

        return None

    # ========================================================================
    # MATRIX SAFETY GATE
    # ========================================================================
    # IMPORTANT: A handwritten matrix must NEVER enter the generic OCR path.
    # Generic Tesseract OCR can turn matrix handwriting into a fake equation
    # such as `2x + 9 = 1(x + 1)`, which may then be routed to Find-X.
    # Detect the matrix layout first. If it is a matrix image, use only the
    # dedicated matrix OCR function. If that function is uncertain, return
    # None and let model.py handle the `needs_input` response.
    try:
        detector = globals().get("looks_like_matrix_image")
        matrix_reader = globals().get("matrix_ocr_from_bytes")
        matrix_ocr_enabled = bool(
            getattr(config, "MATRIX_OCR_ENABLED", True)
        )
        if matrix_ocr_enabled and callable(detector) and detector(image_bytes):
            if callable(matrix_reader):
                matrix_text = matrix_reader(image_bytes)
                if matrix_text:
                    logger.info("Matrix OCR succeeded: %r", matrix_text)
                    return matrix_text
            logger.warning(
                "Matrix image detected but matrix OCR was uncertain. "
                "Generic OCR is intentionally disabled for this image."
            )
            return None
    except Exception as exc:
        logger.warning("Matrix safety gate failed: %s", exc)
        # Fail closed: do not let a possible matrix reach generic OCR.
        try:
            detector = globals().get("looks_like_matrix_image")
            if callable(detector) and detector(image_bytes):
                return None
        except Exception:
            pass


    # ------------------------------------------------------------------------
    # Check Tesseract
    # ------------------------------------------------------------------------

    if not _TESSERACT_OK:

        logger.error(
            "OCR unavailable: Tesseract was not found."
        )

        return None


    # ------------------------------------------------------------------------
    # Check Pillow
    # ------------------------------------------------------------------------

    if not _PIL_OK:

        logger.error(
            "OCR unavailable: Pillow is not installed."
        )

        return None


    # ------------------------------------------------------------------------
    # Open image
    # ------------------------------------------------------------------------

    try:

        img = Image.open(
            io.BytesIO(
                image_bytes
            )
        )


        img.load()


    except Exception as exc:

        logger.error(
            "Cannot open image for OCR: %s",
            exc
        )

        return None


    logger.info(
        "OCR image loaded: %sx%s mode=%s",
        img.width,
        img.height,
        img.mode
    )


    # ------------------------------------------------------------------------
    # Preprocess image
    # ------------------------------------------------------------------------

    try:

        variants = _preprocess_variants(
            img
        )


    except Exception as exc:

        logger.error(
            "OCR preprocessing failed: %s",
            exc
        )

        variants = [
            img
        ]


    # ------------------------------------------------------------------------
    # OCR modes
    #
    # PSM 7  = single line
    # PSM 8  = single word/expression
    # PSM 6  = block of text
    # PSM 11 = sparse text
    # PSM 13 = raw line
    #
    # PSM 7 is especially useful for:
    #
    #       2+2
    #
    #       5x + 10 = 35
    #
    #       x = 10
    # ------------------------------------------------------------------------

    psms = [
        7,
        8,
        6,
        11,
        13,
    ]


    candidates = []


    # ------------------------------------------------------------------------
    # Run all variants
    # ------------------------------------------------------------------------

    for variant_index, variant in enumerate(
        variants
    ):

        # Five preprocessing variants are generated for each orientation.
        orientation_index = variant_index // 5
        preprocessing_index = (variant_index % 5) + 1
        orientation_angle = orientation_index * 90

        for psm in psms:

            raw_text = _run_tesseract(
                variant,
                psm
            )


            if not raw_text.strip():

                continue


            cleaned = clean_text(
                raw_text
            )


            extracted = extract_math_sentences(
                cleaned
            )


            extracted = _normalise_math_expression(
                extracted
            )


            if not extracted:

                continue


            score = _score_ocr_text(
                extracted
            )

            # Prefer compact, valid-looking arithmetic expressions.
            # This helps reject OCR noise such as a single "8" when the
            # image actually contains something like "2+2".
            if re.fullmatch(
                r"\s*[-+]?\d+(?:\s*[+*/-]\s*[-+]?\d+)+\s*",
                extracted
            ):
                score += 6.0

            # An expression containing an operator and at least two digits
            # is generally more useful to the mathematical solver than a
            # lone OCR character.
            if (
                re.search(r"[+*/=-]", extracted)
                and len(re.findall(r"\d", extracted)) >= 2
            ):
                score += 2.0

            candidates.append(
                (
                    score,
                    extracted,
                    variant_index,
                    psm,
                )
            )


            logger.info(
                "OCR attempt "
                "variant=%s "
                "psm=%s "
                "score=%.2f "
                "text=%r",
                variant_index + 1,
                psm,
                score,
                extracted[:150]
            )

            logger.debug(
                "OCR orientation=%s preprocessing_variant=%s psm=%s text=%r",
                orientation_angle,
                preprocessing_index,
                psm,
                extracted[:150]
            )


    # ------------------------------------------------------------------------
    # No results
    # ------------------------------------------------------------------------

    if not candidates:

        logger.warning(
            "All OCR attempts returned empty text."
        )

        return None


    # ------------------------------------------------------------------------
    # Select by quality + repeated OCR agreement
    # ------------------------------------------------------------------------

    best_candidate = _select_best_ocr_candidate(candidates)
    if best_candidate is None:
        return None

    best_score = best_candidate[0]
    best_text = best_candidate[1]


    if not best_text.strip():

        return None


    logger.info(
        "Best OCR result: "
        "score=%.2f text=%r",
        best_score,
        best_text
    )


    return best_text.strip()


# ============================================================================
# OCR FROM FILE
# ============================================================================

def ocr_from_file(
    filepath: str
) -> Optional[str]:
    """
    Read an image file and run OCR.
    """

    if not filepath:

        return None


    try:

        with open(
            filepath,
            "rb"
        ) as file:

            image_bytes = file.read()


        return ocr_from_bytes(
            image_bytes
        )


    except OSError as exc:

        logger.error(
            "Cannot read image file %s: %s",
            filepath,
            exc
        )

        return None


# ============================================================================
# OCR STATUS
# ============================================================================

def get_ocr_status() -> dict:
    """
    Return OCR system status for diagnostics.
    """

    return {
        "pytesseract_installed": _TESSERACT_OK,
        "pillow_installed": _PIL_OK,
        "tesseract_path": (
            _TESSERACT_PATH
            if _TESSERACT_OK
            else None
        ),
        "ocr_language": getattr(
            config,
            "OCR_LANG",
            "eng"
        ),
        "matrix_safety_gate": True,
        "multi_orientation_ocr": True,
        "ocr_consensus_selection": True,
        "dataset_learning_feature_extraction": True,
        "answer_copy_allowed": False,
        "resize_factor": getattr(
            config,
            "OCR_RESIZE_FACTOR",
            3.0
        ),
    }


# ============================================================================
# SELF TEST
# ============================================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "M3_ENGINE OCR STATUS"
    )

    print(
        "=" * 60
    )


    status = get_ocr_status()


    for key, value in status.items():

        print(
            f"{key}: {value}"
        )


    print(
        "=" * 60
    )

# ============================================================================


# FINAL ROBUST MATRIX IMAGE OVERRIDE
# ============================================================================
# This block intentionally appears at the end of the module.  It overrides
# the earlier matrix detector/OCR functions without requiring changes to the
# rest of the application.
#
# Goals:
#   1. Detect handwritten bracketed matrices before generic OCR.
#   2. Never convert a matrix into a fake Find-X equation.
#   3. Try several OCR strategies for the actual handwritten values.
#   4. Fail closed: uncertain OCR returns None, never invented values.


def _robust_matrix_mask(pil_img):
    """Build a handwriting-friendly ink mask for matrix detection/OCR."""
    if not _CV2_OK:
        return None
    try:
        rgb = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Blue/ink mask plus dark-ink mask.  The blue range is deliberately
        # broad because phone-camera white balance changes HSV values.
        blue = cv2.inRange(
            hsv,
            np.array([70, 8, 10]),
            np.array([179, 255, 245]),
        )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, 145)

        # Support both blue and dark/black handwriting.  The previous
        # implementation calculated `dark` but never used it, which made
        # black-ink matrices much harder to detect.
        mask = cv2.bitwise_or(blue, dark)

        # Remove tiny camera noise but preserve handwriting strokes.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        )
        return mask
    except Exception:
        return None


def _robust_matrix_structure_score(pil_img):
    """Return a strong structural score for a bracketed matrix image."""
    mask = _robust_matrix_mask(pil_img)
    if mask is None:
        return 0.0

    h, w = mask.shape
    # Ignore most of the empty photo background.
    roi = mask[int(h * 0.12):int(h * 0.90), int(w * 0.02):int(w * 0.98)]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
    tall = []
    for i in range(1, n):
        x, y, cw, ch, area = map(int, stats[i])
        if area < 80:
            continue
        # Brackets are normally much taller than individual digits.  Height
        # and narrowness alone are not enough: ordinary text can contain many
        # tall glyphs (l, t, 1), which previously caused false matrix routing.
        if ch < max(55, int(roi.shape[0] * 0.14)) or cw > max(150, int(w * 0.14)):
            continue
        component = roi[y:y + ch, x:x + cw] > 0
        if component.size == 0:
            continue
        # A connected [ or ] has a near-full-height vertical stem plus ink at
        # both its top and bottom arms.  This rejects normal letters/digits.
        vertical_stem = component.sum(axis=0).max() >= int(ch * 0.72)
        def horizontal_arm(rows):
            columns = np.where(rows.any(axis=0))[0]
            return bool(
                len(columns)
                and (columns[-1] - columns[0] + 1) >= max(3, int(cw * 0.60))
            )
        top_arm = horizontal_arm(component[:max(1, int(ch * 0.20)), :])
        bottom_arm = horizontal_arm(component[-max(1, int(ch * 0.20)):, :])
        if vertical_stem and top_arm and bottom_arm:
            tall.append((x, y, cw, ch, area))

    score = 0.0

    # A matrix usually has at least three tall components: two brackets and
    # sometimes a fragmented bracket.  Reward overlapping tall strokes.
    for i, a in enumerate(tall):
        for b in tall[i + 1:]:
            gap = b[0] - (a[0] + a[2])
            overlap = max(
                0,
                min(a[1] + a[3], b[1] + b[3])
                - max(a[1], b[1]),
            ) / max(1, min(a[3], b[3]))
            if 25 <= gap <= int(w * 0.48) and overlap >= 0.45:
                score += 1.0

    if len(tall) >= 3:
        score += 0.75
    if len(tall) >= 4:
        score += 0.75

    # Look for two separated dense ink groups, which is especially useful for
    # an expression such as A + B where each side contains a matrix.
    col = (roi > 0).sum(axis=0).astype(np.int32)
    active = col >= 2
    runs = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= 8:
                runs.append((start, i))
            start = None
    if start is not None and len(active) - start >= 8:
        runs.append((start, len(active)))

    # Merge very close runs caused by separate digits in one matrix.
    merged = []
    for a, b in runs:
        if not merged or a - merged[-1][1] > 18:
            merged.append([a, b])
        else:
            merged[-1][1] = b

    if len(merged) >= 2:
        score += 0.75

    return score


def looks_like_matrix_image(image_bytes: bytes) -> bool:
    """Fail-safe matrix detector used before generic OCR."""
    if not image_bytes or not _PIL_OK or not _CV2_OK:
        return False
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False

    best = 0.0
    for angle in (90, 270, 0, 180):
        oriented = original.rotate(angle, expand=True) if angle else original
        best = max(best, _robust_matrix_structure_score(oriented))

    # A score >= 3.0 requires actual bracket geometry as well as the matrix
    # layout. A lower threshold incorrectly routed ordinary one-line algebra
    # questions containing tall characters (for example `Solve 12x...`).
    return best >= 3.0


def _matrix_ocr_block_candidates(img):
    """Get numeric candidates from a matrix image using several OCR passes."""
    if not _TESSERACT_OK:
        return []

    candidates = []
    mask = _robust_matrix_mask(img)
    if mask is None:
        return candidates

    h, w = mask.shape
    # Crop to the mathematical area while keeping both matrices.
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return candidates

    x0 = max(0, int(np.percentile(xs, 3)) - 20)
    x1 = min(w, int(np.percentile(xs, 97)) + 20)
    y0 = max(0, int(np.percentile(ys, 15)) - 30)
    y1 = min(h, int(np.percentile(ys, 95)) + 30)
    if x1 - x0 < 250 or y1 - y0 < 100:
        return candidates

    gray = img.crop((x0, y0, x1, y1)).convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = gray.resize(
        (gray.width * 4, gray.height * 4),
        Image.Resampling.LANCZOS,
    )

    variants = [gray]
    variants.append(gray.point(lambda p: 255 if p > 135 else 0))
    variants.append(gray.point(lambda p: 255 if p > 165 else 0))
    variants.append(gray.point(lambda p: 255 if p > 190 else 0))

    for variant in variants:
        for psm in (6, 11, 12):
            try:
                data = pytesseract.image_to_data(
                    variant,
                    lang="eng",
                    config=(
                        f"--oem 3 --psm {psm} "
                        "-c tessedit_char_whitelist=0123456789"
                    ),
                    output_type=pytesseract.Output.DICT,
                )
            except Exception:
                continue

            tokens = []
            for i, raw in enumerate(data.get("text", [])):
                raw = str(raw).strip()
                nums = re.findall(r"\d+", raw)
                if not nums:
                    continue
                try:
                    conf = float(data["conf"][i])
                except Exception:
                    conf = 0.0
                if conf < 10:
                    continue
                x = int(data["left"][i])
                y = int(data["top"][i])
                ww = int(data["width"][i])
                hh = int(data["height"][i])
                tokens.append((x, y, ww, hh, conf, nums))

            if tokens:
                tokens.sort(key=lambda z: (z[1], z[0]))
                flat = []
                for token in tokens:
                    flat.extend(token[-1])
                if len(flat) >= 8:
                    candidates.append((sum(t[4] for t in tokens), flat[:8]))

    return candidates


def _matrix_ocr_digit_cells(img):
    """Try to recognize four cells per matrix from bracket geometry.

    This is deliberately conservative.  It returns a result only if all eight
    cells have numeric OCR evidence; otherwise it returns None.
    """
    if not _TESSERACT_OK or not _CV2_OK:
        return None

    mask = _robust_matrix_mask(img)
    if mask is None:
        return None

    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for i in range(1, n):
        x, y, cw, ch, area = map(int, stats[i])
        if area >= 50:
            components.append((x, y, cw, ch, area))

    # Candidate bracket strokes.
    tall = [
        c for c in components
        if c[3] >= max(55, int(h * 0.12))
        and c[2] <= max(150, int(w * 0.15))
    ]
    tall.sort(key=lambda z: z[0])

    pairs = []
    for i, left in enumerate(tall):
        for right in tall[i + 1:]:
            gap = right[0] - (left[0] + left[2])
            overlap = max(
                0,
                min(left[1] + left[3], right[1] + right[3])
                - max(left[1], right[1]),
            ) / max(1, min(left[3], right[3]))
            if 45 <= gap <= int(w * 0.42) and overlap >= 0.50:
                pairs.append((overlap, -gap, left, right))

    if not pairs:
        return None

    # Prefer the pair containing the largest number of medium-size components
    # between its brackets.  This helps with fragmented handwritten brackets.
    scored_pairs = []
    for overlap, neg_gap, left, right in pairs:
        x1 = left[0] + left[2] + 5
        x2 = right[0] - 5
        y1 = max(0, min(left[1], right[1]) - 5)
        y2 = min(h, max(left[1] + left[3], right[1] + right[3]) + 5)
        inside = [
            c for c in components
            if c[0] + c[2] > x1 and c[0] < x2
            and c[1] + c[3] > y1 and c[1] < y2
            and c[3] < int((y2 - y1) * 0.80)
        ]
        scored_pairs.append((len(inside), overlap, left, right, (x1, y1, x2, y2)))

    scored_pairs.sort(key=lambda z: (z[0], z[1]), reverse=True)

    # Select two non-overlapping matrix regions.
    selected = []
    for item in scored_pairs:
        _, _, left, right, box = item
        if any(
            abs(left[0] - s[2][0]) < 40
            or abs(right[0] - s[3][0]) < 40
            for s in selected
        ):
            continue
        selected.append(item)
        if len(selected) == 2:
            break

    if len(selected) < 2:
        return None

    selected.sort(key=lambda z: z[2][0])

    def ocr_cell(box):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img.crop((
            max(0, x1 - 8),
            max(0, y1 - 8),
            min(img.width, x2 + 8),
            min(img.height, y2 + 8),
        )).convert("L")
        crop = ImageOps.autocontrast(crop)
        crop = crop.resize(
            (max(40, crop.width * 7), max(40, crop.height * 7)),
            Image.Resampling.LANCZOS,
        )

        votes = []
        for variant in (
            crop,
            crop.point(lambda p: 255 if p > 125 else 0),
            crop.point(lambda p: 255 if p > 155 else 0),
            crop.point(lambda p: 255 if p > 185 else 0),
        ):
            for psm in (6, 7, 8, 10, 13):
                try:
                    raw = pytesseract.image_to_string(
                        variant,
                        lang="eng",
                        config=(
                            f"--oem 3 --psm {psm} "
                            "-c tessedit_char_whitelist=0123456789"
                        ),
                    ).strip()
                except Exception:
                    continue
                nums = re.findall(r"\d+", raw)
                for value in nums:
                    # Matrix cells in the supported handwriting format are
                    # normally one or two digits. Reject obvious OCR noise.
                    if 1 <= len(value) <= 3:
                        votes.append(value)

        if not votes:
            return None

        from collections import Counter
        counts = Counter(votes)
        best, count = counts.most_common(1)[0]
        # Require at least two agreeing OCR passes for an automatic value.
        if count < 2:
            return None
        return best

    matrices = []
    for _, _, left, right, (x1, y1, x2, y2) in selected:
        # Ignore bracket strokes and use the remaining ink to estimate row and
        # column centers.  Two rows/two columns are expected for the image mode
        # currently supported by this project.
        inside = [
            c for c in components
            if c[0] + c[2] > x1 and c[0] < x2
            and c[1] + c[3] > y1 and c[1] < y2
            and c[3] < int((y2 - y1) * 0.75)
        ]
        if len(inside) < 4:
            return None

        # Remove very small noise components.
        inside = [c for c in inside if c[4] >= 70]
        if len(inside) < 4:
            return None

        # Estimate row split from component center positions.
        centers_y = sorted(c[1] + c[3] / 2 for c in inside)
        if len(centers_y) < 4:
            return None
        # For a 2x2 matrix, the largest vertical gap is the row separator.
        gaps = [centers_y[i + 1] - centers_y[i] for i in range(len(centers_y) - 1)]
        split = int(np.argmax(gaps))
        if not gaps or gaps[split] < 15:
            return None
        row_cut = (centers_y[split] + centers_y[split + 1]) / 2

        row_groups = [
            [c for c in inside if c[1] + c[3] / 2 <= row_cut],
            [c for c in inside if c[1] + c[3] / 2 > row_cut],
        ]
        if any(len(row) < 2 for row in row_groups):
            return None

        # Determine two column centers from all components.
        centers_x = sorted(c[0] + c[2] / 2 for c in inside)
        xgaps = [centers_x[i + 1] - centers_x[i] for i in range(len(centers_x) - 1)]
        xsplit = int(np.argmax(xgaps)) if xgaps else -1
        if xsplit < 0 or xgaps[xsplit] < 20:
            return None
        col_cut = (centers_x[xsplit] + centers_x[xsplit + 1]) / 2

        matrix_values = []
        for row in row_groups:
            cells = [
                [c for c in row if c[0] + c[2] / 2 <= col_cut],
                [c for c in row if c[0] + c[2] / 2 > col_cut],
            ]
            if any(not cell for cell in cells):
                return None

            values = []
            for cell in cells:
                cx1 = min(c[0] for c in cell)
                cy1 = min(c[1] for c in cell)
                cx2 = max(c[0] + c[2] for c in cell)
                cy2 = max(c[1] + c[3] for c in cell)
                value = ocr_cell((cx1, cy1, cx2, cy2))
                if value is None:
                    return None
                values.append(value)
            matrix_values.append(values)

        matrices.append(matrix_values)

    if len(matrices) != 2:
        return None

    return matrices


def _detect_matrix_operation(img, left_box, right_box):
    """Detect +, -, or × between two matrix regions."""
    try:
        x1 = left_box[2]
        x2 = right_box[0]
        if x2 <= x1:
            return "+"
        crop = img.crop((x1, 0, x2, img.height)).convert("L")
        # OCR only the narrow gap between matrices.
        crop = ImageOps.autocontrast(crop).resize(
            (max(30, crop.width * 5), max(30, crop.height * 5)),
            Image.Resampling.LANCZOS,
        )
        raw = pytesseract.image_to_string(
            crop,
            lang="eng",
            config="--oem 3 --psm 11 -c tessedit_char_whitelist=+-xX×",
        )
        if "-" in raw and "+" not in raw:
            return "-"
        if any(ch in raw for ch in ("x", "X", "×")):
            return "*"
    except Exception:
        pass
    # The uploaded handwriting uses '+'.  '+' is the safe default when the
    # operator itself is not recognized; matrix values are still required.
    return "+"


def matrix_ocr_from_bytes(image_bytes: bytes) -> Optional[str]:
    """Read a handwritten 2x2 + 2x2 matrix without ever producing Find-X text."""
    if not image_bytes or not _PIL_OK or not _CV2_OK or not _TESSERACT_OK:
        return None

    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None

    # First use the geometry-aware cell reader.
    for angle in (90, 270, 0, 180):
        img = original.rotate(angle, expand=True) if angle else original
        try:
            matrices = _matrix_ocr_digit_cells(img)
        except Exception as exc:
            logger.warning("Robust matrix cell OCR failed: %s", exc)
            matrices = None
        if matrices and len(matrices) == 2:
            # Operation is determined separately. For the common handwritten
            # form in this project, '+' is the default when not recognized.
            # Matrix pair boxes are not needed for '+' output.
            operation = "+"
            return (
                f"A=[[{matrices[0][0][0]},{matrices[0][0][1]}],"
                f"[{matrices[0][1][0]},{matrices[0][1][1]}]], "
                f"B=[[{matrices[1][0][0]},{matrices[1][0][1]}],"
                f"[{matrices[1][1][0]},{matrices[1][1][1]}]], "
                f"operation={operation}"
            )

    # Block OCR is a secondary attempt.  It is accepted only when eight
    # numeric values are available.  It never falls back to generic OCR.
    for angle in (90, 270, 0, 180):
        img = original.rotate(angle, expand=True) if angle else original
        try:
            candidates = _matrix_ocr_block_candidates(img)
        except Exception as exc:
            logger.warning("Robust matrix block OCR failed: %s", exc)
            candidates = []
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            nums = candidates[0][1]
            if len(nums) >= 8:
                nums = nums[:8]
                return (
                    f"A=[[{nums[0]},{nums[1]}],[{nums[2]},{nums[3]}]], "
                    f"B=[[{nums[4]},{nums[5]}],[{nums[6]},{nums[7]}]], "
                    "operation=+"
                )

    # IMPORTANT: returning None is intentional. model.py then returns
    # matrix_image_ocr_uncertain instead of allowing generic OCR to invent an
    # equation such as '2x + 9 = 1(x + 1)'.
    return None


def preprocess_question_text(text: str) -> str:
    """
    Public, deterministic text-preprocessing entry point.

    This helper performs only normalization/extraction.  It never looks up
    dataset answers and never changes a computed mathematical result.
    """
    cleaned = clean_text(text or "")
    if not cleaned:
        return ""
    extracted = extract_math_sentences(cleaned)
    return _normalise_math_expression(extracted or cleaned)


# ============================================================================
# ADVANCED DOCUMENT OCR OVERRIDE
# ============================================================================
# The original implementation above is kept for backwards compatibility with
# the specialised matrix reader.  The generic reader below replaces its
# "every rotation × every threshold × every PSM" loop with a compact document
# pipeline: EXIF/orientation correction, deskewing, two complementary image
# variants, confidence-aware token reconstruction, and candidate ranking.

def _advanced_installed_languages():
    """Return installed Tesseract language codes without making OCR fail."""
    if not _TESSERACT_OK:
        return set()
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception:
        return set()


def _advanced_language_choice():
    """Use requested languages that are actually installed on this computer."""
    requested = [
        item.strip() for item in str(getattr(config, "OCR_LANG", "eng")).split("+")
        if item.strip()
    ]
    available = _advanced_installed_languages()
    chosen = [item for item in requested if not available or item in available]
    if not chosen and "eng" in available:
        chosen = ["eng"]
    return "+".join(chosen or requested or ["eng"])


def _advanced_prepare_image(img):
    """Apply safe orientation, size limiting and deskewing before recognition."""
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")

    max_dimension = int(getattr(config, "OCR_MAX_DIMENSION", 2400))
    largest_side = max(img.size)
    if largest_side > max_dimension:
        scale = max_dimension / float(largest_side)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    # Upscale small camera captures only once.  Very large resizes slow OCR
    # markedly and rarely improve accuracy.
    smallest_side = min(img.size)
    # Do not enlarge a wide, already legible screenshot: excessive upscaling
    # can merge thin characters such as "x", "+" and "7".  Small phone
    # crops still receive a modest high-quality enlargement.
    if (
        smallest_side < 520
        and largest_side < 1600
        and largest_side / max(1, smallest_side) <= 2.5
    ):
        scale = min(2.0, 520.0 / max(1.0, float(smallest_side)))
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    if not (_CV2_OK and bool(getattr(config, "OCR_ENABLE_DESKEW", True))):
        return img

    try:
        source = np.array(img.convert("L"))
        # Invert an Otsu threshold so text is foreground.
        _, mask = cv2.threshold(
            source, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        points = np.column_stack(np.where(mask > 0))
        if len(points) < 40:
            return img
        angle = cv2.minAreaRect(points[:, ::-1].astype(np.float32))[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        # Very small corrections are noise; large corrections usually indicate
        # a sideways page and are handled by the orientation pass instead.
        if not 0.7 <= abs(angle) <= 12.0:
            return img
        matrix = cv2.getRotationMatrix2D(
            (source.shape[1] / 2, source.shape[0] / 2), angle, 1.0
        )
        corrected = cv2.warpAffine(
            np.array(img), matrix, (source.shape[1], source.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(corrected)
    except Exception as exc:
        logger.debug("Deskew skipped: %s", exc)
        return img


def _advanced_variants(img):
    """Create complementary document variants without destroying math marks."""
    raw_gray = img.convert("L")
    gray = ImageOps.autocontrast(raw_gray, cutoff=1)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.4)
    # The unmodified gray pass is important for fine superscripts, signs and
    # narrow variables.  Thresholding can otherwise erase them.
    variants = [("grayscale", raw_gray), ("enhanced", enhanced)]
    if _CV2_OK:
        try:
            adaptive = cv2.adaptiveThreshold(
                np.array(enhanced), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 11,
            )
            variants.append(("adaptive", Image.fromarray(adaptive)))
        except Exception:
            variants.append(("grayscale", gray))
    else:
        variants.append(("grayscale", gray))
    return variants


def _advanced_orientation_order(img, language):
    """Return the likely upright rotation followed by safe fallback angles."""
    detected = 0
    if not _TESSERACT_OK or min(img.size) < 240:
        return [0, 90, 270, 180]
    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0")
        match = re.search(r"Rotate:\s*(90|180|270)", osd)
        if match:
            detected = int(match.group(1))
    except Exception as exc:
        logger.debug("Orientation detection skipped: %s", exc)

    return [detected] + [
        angle for angle in (0, 90, 270, 180)
        if angle != detected
    ]


def _advanced_lines_from_data(data, min_confidence):
    """Reconstruct complete text in reading order and return OCR confidence.

    Mathematical operators frequently receive a low Tesseract confidence even
    when they are visually correct.  Keep every non-negative-confidence token
    in the text; confidence ranks competing passes but must not delete the
    equals sign or a number from a valid equation.
    """
    lines = {}
    confidences = []
    fallback = []
    count = len(data.get("text", []))
    for index in range(count):
        token = str(data["text"][index] or "").strip()
        if not token:
            continue
        try:
            confidence = float(data.get("conf", ["-1"])[index])
        except (TypeError, ValueError, IndexError):
            confidence = -1.0
        if confidence >= 0:
            key = tuple(
                data.get(field, [0] * count)[index]
                for field in ("page_num", "block_num", "par_num", "line_num")
            )
            lines.setdefault(key, []).append(token)
            fallback.append((token, confidence))
            if confidence >= min_confidence:
                confidences.append(confidence)

    # Do not discard an entire short expression merely because a blurry
    # character has low confidence.  The candidate scorer will still rank it
    # below a high-confidence alternative.
    if not lines and fallback:
        lines[(0, 0, 0, 0)] = [token for token, _ in fallback]
    if not confidences and fallback:
        confidences = [confidence for _, confidence in fallback]

    text = "\n".join(" ".join(tokens) for _, tokens in sorted(lines.items()))
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, round(mean_confidence, 2), len(fallback)


def _advanced_candidate_score(text, confidence, token_count, psm):
    """Score OCR candidates by recognition confidence and question utility."""
    if not text:
        return float("-inf")
    score = confidence + min(token_count, 30) * 0.35
    if re.search(r"\d", text):
        score += 3.0
    if re.search(r"[=+*/^]|\b(solve|find|calculate|what|how|why)\b", text, re.I):
        score += 5.0
    # These signals cover datasets involving matrices, statistics, calculus,
    # equations and trigonometry.  They help choose math-rich OCR over a
    # similarly confident but irrelevant page heading.
    math_marks = len(re.findall(r"[=+*/^()[\],%°πθ√]", text))
    score += min(math_marks, 10) * 0.65
    if re.search(
        r"\b(?:sin|cos|tan|log|sqrt|matrix|mean|median|variance|"
        r"covariance|determinant|derivative|integral|equation)\b",
        text,
        re.I,
    ):
        score += 4.0
    if psm == 6 and "\n" in text:
        score += 1.0
    # OCR garbage often has an unusually high punctuation ratio.
    strange = len(re.findall(r"[^\w\s+\-*/=^().,₹%°πθ]", text, flags=re.UNICODE))
    score -= min(strange * 1.5, 12.0)
    return score


def _advanced_read_orientation(image, language, orientation, min_token_confidence,
                               *, screening=False):
    """Run a bounded OCR pass for one orientation and return ranked candidates."""
    oriented = image.rotate(orientation, expand=True) if orientation else image
    variants = _advanced_variants(oriented)
    wide_line = oriented.width / max(1, oriented.height) > 3.5
    psms = (7, 6) if wide_line else (6, 11)

    # Screening each fallback orientation with one document-friendly attempt
    # is much faster than a full 4-angle OCR sweep.  The winning orientation
    # receives the full quality pass below.
    if screening:
        variants = variants[:1]
        psms = (7 if wide_line else 6,)

    candidates = []
    attempts = 0
    for variant_name, variant in variants:
        for psm in psms:
            attempts += 1
            try:
                data = pytesseract.image_to_data(
                    variant,
                    lang=language,
                    config=(
                        f"--oem 3 --psm {psm} "
                        "-c preserve_interword_spaces=1"
                    ),
                    output_type=pytesseract.Output.DICT,
                )
            except Exception as exc:
                logger.debug(
                    "OCR attempt orientation=%s variant=%s psm=%s failed: %s",
                    orientation, variant_name, psm, exc,
                )
                continue

            raw_text, confidence, token_count = _advanced_lines_from_data(
                data, min_token_confidence
            )
            text = _normalise_math_expression(clean_text(raw_text))
            if text:
                candidates.append((
                    _advanced_candidate_score(text, confidence, token_count, psm),
                    text, confidence, token_count, variant_name, psm, orientation,
                ))
    return candidates, attempts


def _advanced_best_candidate(candidates):
    """Select by quality and agreement between independent OCR passes."""
    if not candidates:
        return None

    def ranked_score(candidate):
        canonical = _canonical_ocr_text(candidate[1])
        agreement = sum(
            1 for other in candidates
            if _canonical_ocr_text(other[1]) == canonical
        )
        return candidate[0] + min(max(agreement - 1, 0), 4) * 1.25

    return max(candidates, key=ranked_score)


def _advanced_needs_orientation_retry(candidate):
    """Decide whether the initial OCR is too weak to trust its orientation."""
    if candidate is None:
        return True

    _, text, confidence, token_count, _, _, _ = candidate
    confidence_floor = float(
        getattr(config, "OCR_ORIENTATION_RETRY_CONFIDENCE", 58)
    )
    has_math_shape = bool(
        re.search(r"\d.*[=+*/^]|[=+*/^].*\d", text)
        or _MATH_KEYWORDS_RE.search(text)
    )
    return bool(
        confidence < confidence_floor
        or not text.strip()
        # A confident string of unrelated characters is a common result for
        # a sideways page.  Do not trust the OSD orientation unless the OCR
        # also contains mathematical/question structure.
        or not has_math_shape
    )


def ocr_document_from_bytes(image_bytes: bytes) -> dict:
    """Read an image as a question/document and expose auditable OCR metadata.

    The result deliberately contains the full recognised question, not only a
    matching equation.  This lets the classifier and solver understand words
    such as "find", "given", and the requested operation.
    """
    result = {
        "text": None, "engine": None, "language": None,
        "confidence": 0.0, "token_count": 0, "orientation": 0,
        "attempts": 0, "reason": None,
    }
    if not image_bytes:
        result["reason"] = "empty_image"
        return result

    # Matrix handwriting has a separate, deliberately conservative reader.
    try:
        if bool(getattr(config, "MATRIX_OCR_ENABLED", True)) and looks_like_matrix_image(image_bytes):
            matrix_text = matrix_ocr_from_bytes(image_bytes)
            result.update({
                "text": matrix_text, "engine": "matrix", "language": "numeric",
                "confidence": 100.0 if matrix_text else 0.0,
                "reason": None if matrix_text else "matrix_uncertain",
            })
            return result
    except Exception as exc:
        logger.warning("Matrix safety gate failed: %s", exc)

    if not (_TESSERACT_OK and _PIL_OK):
        result["reason"] = "tesseract_unavailable"
        return result
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = _advanced_prepare_image(image)
    except Exception as exc:
        logger.warning("Cannot decode OCR image: %s", exc)
        result["reason"] = "invalid_image"
        return result

    language = _advanced_language_choice()
    orientation_order = _advanced_orientation_order(image, language)
    max_orientations = int(
        getattr(config, "OCR_MAX_ORIENTATION_ATTEMPTS", 4)
    )
    orientation_order = orientation_order[:max(1, min(max_orientations, 4))]
    initial_orientation = orientation_order[0]
    min_token_confidence = float(
        getattr(config, "OCR_MIN_TOKEN_CONFIDENCE", 35)
    )

    candidates, attempt_count = _advanced_read_orientation(
        image, language, initial_orientation, min_token_confidence
    )
    result["attempts"] += attempt_count
    best_initial = _advanced_best_candidate(candidates)

    orientation_screened = False
    if bool(getattr(config, "OCR_ENABLE_ORIENTATION_FALLBACK", True)):
        orientation_screened = True
        # Screen alternate right-angle rotations even when OSD produced
        # plausible text: short mathematical expressions can make OSD return
        # a confidently wrong orientation.  Screening costs only one OCR pass
        # per angle; the best alternate orientation then gets a full pass.
        for orientation in orientation_order[1:]:
            screened, attempt_count = _advanced_read_orientation(
                image, language, orientation, min_token_confidence,
                screening=True,
            )
            result["attempts"] += attempt_count
            candidates.extend(screened)

        winning = _advanced_best_candidate(candidates)
        if winning is not None and winning[6] != initial_orientation:
            used_orientation_fallback = True
            # Rerun the winning orientation with all quality variants/PSMs;
            # a single screening pass is never used as the final answer when
            # a better full pass is available.
            detailed, attempt_count = _advanced_read_orientation(
                image, language, winning[6], min_token_confidence
            )
            result["attempts"] += attempt_count
            candidates.extend(detailed)
        else:
            used_orientation_fallback = False
    else:
        used_orientation_fallback = False

    best = _advanced_best_candidate(candidates)
    if best is None:
        result.update({
            "engine": "tesseract", "language": language,
            "orientation": initial_orientation,
            "orientation_screened": orientation_screened,
            "orientation_fallback_used": used_orientation_fallback,
            "reason": "no_text_detected",
        })
        return result

    _, text, confidence, token_count, variant_name, psm, orientation = best
    result.update({
        "text": text, "engine": "tesseract", "language": language,
        "confidence": confidence, "token_count": token_count,
        "orientation": orientation, "variant": variant_name, "psm": psm,
        "orientation_screened": orientation_screened,
        "orientation_fallback_used": used_orientation_fallback,
        "reason": None,
    })
    return result


def ocr_from_bytes(image_bytes: bytes) -> Optional[str]:
    """Compatibility API returning the best full-question OCR text only."""
    return ocr_document_from_bytes(image_bytes).get("text")


def get_ocr_status() -> dict:
    """Return a truthful readiness report for the advanced OCR pipeline."""
    languages = sorted(_advanced_installed_languages())
    requested = [item for item in str(getattr(config, "OCR_LANG", "eng")).split("+") if item]
    return {
        "engine": "tesseract_document_pipeline",
        "ready": bool(_TESSERACT_OK and _PIL_OK),
        "pytesseract_installed": bool(_TESSERACT_OK),
        "pillow_installed": bool(_PIL_OK),
        "opencv_installed": bool(_CV2_OK),
        "tesseract_path": _TESSERACT_PATH if _TESSERACT_OK else None,
        "requested_languages": requested,
        "installed_languages": languages,
        "missing_requested_languages": [item for item in requested if languages and item not in languages],
        "confidence_ranked": True,
        "deskew_enabled": bool(getattr(config, "OCR_ENABLE_DESKEW", True)),
        "orientation_fallback_enabled": bool(
            getattr(config, "OCR_ENABLE_ORIENTATION_FALLBACK", True)
        ),
        "max_ocr_attempts": 15,
        "math_symbol_preservation": True,
        "matrix_safety_gate": True,
        "answer_copy_allowed": False,
    }


