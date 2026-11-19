"""
M3_ENGINE — Flask API

Endpoints
---------
POST /solve
    Accepts JSON, form data, multipart form data, or an image upload.
    Returns a structured mathematical solution.

GET /health
    Returns engine status and dataset statistics.

POST /reload
    Reloads datasets and synchronizes the model.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import sys
import uuid
from typing import Optional


# ============================================================
# PROJECT ROOT
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(os.path.join(_THIS_DIR, "model")):
    PROJECT_ROOT = _THIS_DIR
else:
    PROJECT_ROOT = os.path.dirname(_THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# FLASK IMPORTS
# ============================================================

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from model.model import model
from utils import preprocessor


# ============================================================
# STARTUP CONFIGURATION VALIDATION
# ============================================================

try:
    config.validate_config()
except Exception as exc:
    raise RuntimeError(
        f"Invalid Math Mind configuration: {exc}"
    ) from exc


# ============================================================
# API / SOLVER PROVENANCE
# ============================================================

def _json_safe(value):
    """Convert common mathematical scalar/container objects to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    # NumPy-like scalar support without requiring NumPy as an API dependency.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    # SymPy-like numbers/matrices and other objects with useful string forms.
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    return str(value)


def _apply_api_metadata(result, request_id):
    """Attach non-invasive API metadata to solver responses.

    The API exposes provenance only. Dataset similarity is never converted
    into a final answer by this layer.
    """
    if not isinstance(result, dict):
        return result

    result.setdefault("api_request_id", request_id)

    # Explicitly preserve the project's no-answer-copy contract.
    result["similarity_answer_copy"] = bool(
        result.get("similarity_answer_copy", False)
    )

    return _json_safe(result)




# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=(
        logging.DEBUG
        if config.API_DEBUG
        else logging.INFO
    ),
    format=(
        "%(asctime)s  %(levelname)-8s  "
        "%(name)s — %(message)s"
    ),
)

logger = logging.getLogger("M3_ENGINE.api")


# ============================================================
# RUNTIME DIAGNOSTICS
# ============================================================

def _runtime_diagnostics():
    """
    Show exactly which Python files are being imported.

    This helps detect situations where Python is accidentally
    using an old/stale copy of the project.
    """

    try:
        import model.model as model_module
        import model.solver as solver_module
        import model.classifier as classifier_module
        import utils.preprocessor as preprocessor_module
        import utils.dataset_loader as dataset_loader_module

        return {
            "app_file": os.path.abspath(__file__),
            "project_root": PROJECT_ROOT,

            "model_file": os.path.abspath(
                model_module.__file__
            ),

            "solver_file": os.path.abspath(
                solver_module.__file__
            ),

            "classifier_file": os.path.abspath(
                classifier_module.__file__
            ),

            "preprocessor_file": os.path.abspath(
                preprocessor_module.__file__
            ),

            "dataset_loader_file": os.path.abspath(
                dataset_loader_module.__file__
            ),
        }

    except Exception as exc:
        logger.exception(
            "Runtime diagnostics failed"
        )

        return {
            "diagnostics_error": str(exc)
        }


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    static_folder=str(config.FRONTEND_DIR),
    static_url_path="",
)

app.config["MAX_CONTENT_LENGTH"] = (
    config.MAX_CONTENT_LENGTH
)


# ============================================================
# CORS
# ============================================================

cors_origins = getattr(
    config,
    "CORS_ORIGINS",
    "*"
)

CORS(
    app,
    origins=cors_origins
)


# ============================================================
# COMMON ERROR RESPONSE
# ============================================================

def _bad(
    message: str,
    code: int = 400
):
    """
    Return a consistent JSON error response.
    """

    return jsonify({
        "type": "error",
        "message": str(message),
    }), code


# ============================================================
# SAFE TEXT
# ============================================================

def _safe_text(
    value
) -> Optional[str]:
    """
    Convert incoming text into safe bounded text.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    max_length = int(
        getattr(
            config,
            "MAX_TEXT_LENGTH",
            5000
        )
    )

    if len(value) > max_length:
        raise ValueError(
            f"Text is too long. "
            f"Maximum length is "
            f"{max_length} characters."
        )

    return value


# ============================================================
# BASE64 IMAGE DECODER
# ============================================================

def _decode_base64_image(
    value: str
) -> bytes:
    """
    Decode base64 image data.

    Supports:

        raw base64

    and:

        data:image/png;base64,...
    """

    value = str(value).strip()

    if not value:
        raise ValueError(
            "Image data is empty."
        )

    # Browser data URI support
    if value.startswith("data:"):

        if "," not in value:
            raise ValueError(
                "Invalid image data URI."
            )

        header, value = value.split(
            ",",
            1
        )

        if ";base64" not in header.lower():
            raise ValueError(
                "Image data URI must contain "
                "base64 data."
            )

    try:

        return base64.b64decode(
            value,
            validate=True
        )

    except (
        binascii.Error,
        ValueError
    ) as exc:

        raise ValueError(
            "Invalid base64 image data."
        ) from exc


# ============================================================
# IMAGE READER
# ============================================================

def _read_image() -> Optional[bytes]:
    """
    Read an uploaded image.

    Supported multipart fields:

        image
        file
        photo
        upload
    """

    allowed_types = set(
        getattr(
            config,
            "ALLOWED_IMAGE_TYPES",
            set()
        )
    )

    # --------------------------------------------------------
    # MULTIPART IMAGE
    # --------------------------------------------------------

    for field_name in (
        "image",
        "file",
        "photo",
        "upload"
    ):

        if field_name not in request.files:
            continue

        file = request.files.get(
            field_name
        )

        if not file or not file.filename:
            continue

        mimetype = (
            file.mimetype or ""
        ).split(
            ";",
            1
        )[0].lower()

        # ----------------------------------------------------
        # MIME TYPE CHECK
        # ----------------------------------------------------

        if mimetype not in allowed_types:

            filename = (
                file.filename or ""
            ).lower()

            extension_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tif": "image/tiff",
                ".tiff": "image/tiff",
            }

            detected_type = next(
                (
                    mime
                    for ext, mime
                    in extension_types.items()
                    if filename.endswith(ext)
                ),
                "",
            )

            if detected_type not in allowed_types:

                raise ValueError(
                    "Unsupported image type. "
                    "Allowed types: "
                    + ", ".join(
                        sorted(
                            allowed_types
                        )
                    )
                )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        data = file.read()

        if not data:
            raise ValueError(
                "Uploaded image is empty."
            )

        max_image_bytes = int(
            getattr(
                config,
                "MAX_IMAGE_BYTES",
                getattr(config, "MAX_CONTENT_LENGTH", 0),
            ) or 0
        )
        if max_image_bytes > 0 and len(data) > max_image_bytes:
            raise ValueError(
                f"Uploaded image is too large. Maximum size is "
                f"{max_image_bytes} bytes."
            )

        logger.info(
            "IMAGE RECEIVED: "
            "field=%s filename=%s bytes=%d mimetype=%s",
            field_name,
            file.filename,
            len(data),
            mimetype or "unknown",
        )

        return data

    # ========================================================
    # RAW IMAGE REQUEST
    # ========================================================

    content_type = (
        request.content_type or ""
    ).split(
        ";",
        1
    )[0].lower()

    if content_type in allowed_types:

        data = request.get_data()

        if not data:
            raise ValueError(
                "Uploaded image is empty."
            )

        max_image_bytes = int(
            getattr(
                config,
                "MAX_IMAGE_BYTES",
                getattr(config, "MAX_CONTENT_LENGTH", 0),
            ) or 0
        )
        if max_image_bytes > 0 and len(data) > max_image_bytes:
            raise ValueError(
                f"Uploaded image is too large. Maximum size is "
                f"{max_image_bytes} bytes."
            )

        logger.info(
            "RAW IMAGE RECEIVED: "
            "bytes=%d content_type=%s",
            len(data),
            content_type,
        )

        return data

    return None


# ============================================================
# REQUEST PARSER
# ============================================================

def _parse_request():
    """
    Parse supported request formats.

    Returns:

        question
        image_bytes
        subject_hint
    """

    question = None
    image_bytes = None
    subject_hint = None

    content_type = (
        request.content_type or ""
    ).lower()

    # ========================================================
    # MULTIPART / FORM
    # ========================================================

    if (
        "multipart/form-data"
        in content_type
        or
        "application/x-www-form-urlencoded"
        in content_type
    ):

        question = _safe_text(
            request.form.get(
                "question"
            )
        )

        subject_hint = _safe_text(
            request.form.get(
                "subject"
            )
        )

        image_bytes = _read_image()

        return (
            question,
            image_bytes,
            subject_hint
        )

    # ========================================================
    # JSON
    # ========================================================

    if "application/json" in content_type:

        data = request.get_json(
            silent=True
        )

        if data is None:
            raise ValueError(
                "Invalid JSON request body."
            )

        if not isinstance(
            data,
            dict
        ):

            raise ValueError(
                "Request body must contain "
                "a JSON object."
            )

        question = _safe_text(
            data.get(
                "question"
            )
        )

        subject_hint = _safe_text(
            data.get(
                "subject"
            )
        )

        # Support multiple image field names
        image_b64 = (
            data.get("image_b64")
            or data.get("image")
            or data.get("image_data")
            or data.get("photo")
        )

        if isinstance(
            image_b64,
            dict
        ):

            image_b64 = (
                image_b64.get(
                    "base64"
                )
                or
                image_b64.get(
                    "data"
                )
            )

        if image_b64:

            image_bytes = (
                _decode_base64_image(
                    str(image_b64)
                )
            )

            logger.info(
                "BASE64 IMAGE RECEIVED: "
                "bytes=%d",
                len(image_bytes)
            )

        return (
            question,
            image_bytes,
            subject_hint
        )

    # ========================================================
    # RAW IMAGE / QUERY FALLBACK
    # ========================================================

    image_bytes = _read_image()

    question = _safe_text(
        request.values.get(
            "question"
        )
    )

    subject_hint = _safe_text(
        request.values.get(
            "subject"
        )
    )

    return (
        question,
        image_bytes,
        subject_hint
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    frontend = (
        config.FRONTEND_DIR
        / "index.html"
    )

    if not frontend.exists():

        return _bad(
            "Frontend index.html "
            "was not found.",
            404
        )

    return send_from_directory(
        str(config.FRONTEND_DIR),
        "index.html"
    )


# ============================================================
# SOLVE API
# ============================================================

@app.route(
    "/solve",
    methods=["POST"]
)
def solve():
    """
    Main mathematical solving endpoint.

    Supported inputs:

        question
        image
        subject
        image_b64
    """

    # ========================================================
    # PARSE REQUEST
    # ========================================================

    try:

        (
            question,
            image_bytes,
            subject_hint
        ) = _parse_request()

    except ValueError as exc:

        return _bad(
            str(exc),
            400
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    if (
        not question
        and not image_bytes
    ):

        return _bad(
            "Provide 'question' text "
            "and/or an image file using "
            "field name image, file, "
            "photo, or upload.",
            400
        )

    # ========================================================
    # SOLVE
    # ========================================================

    request_id = uuid.uuid4().hex

    try:

        logger.info(
            "SOLVE REQUEST: "
            "request_id=%s question=%r subject=%r image_bytes=%d",
            request_id,
            question,
            subject_hint,
            len(image_bytes)
            if image_bytes
            else 0,
        )

        # ----------------------------------------------------
        # Send original image directly to model
        # ----------------------------------------------------

        result = model.solve(
            text=question,
            image_bytes=image_bytes,
            subject_hint=subject_hint,
        )

        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        if (
            config.API_DEBUG
            and isinstance(
                result,
                dict
            )
        ):

            result.setdefault(
                "api_image_received",
                bool(image_bytes)
            )

            result.setdefault(
                "api_image_bytes",
                (
                    len(image_bytes)
                    if image_bytes
                    else 0
                )
            )

            result.setdefault(
                "api_image_sha256",
                (
                    hashlib.sha256(
                        image_bytes
                    ).hexdigest()
                    if image_bytes
                    else None
                )
            )

            result.setdefault(
                "runtime_modules",
                _runtime_diagnostics()
            )

        # ====================================================
        # API RESPONSE METADATA
        # ====================================================

        result = _apply_api_metadata(
            result,
            request_id
        )

        # ====================================================
        # VALIDATE SOLVER RESPONSE
        # ====================================================

        if not isinstance(
            result,
            dict
        ):

            logger.error(
                "model.solve returned "
                "unexpected type: %s",
                type(result).__name__
            )

            return _bad(
                "The solver returned "
                "an invalid response.",
                500
            )

        # Final JSON-safety pass. This prevents SymPy/NumPy scalar objects
        # from leaking through the Flask JSON encoder.
        result = _json_safe(result)

        return jsonify(
            result
        ), 200

    # ========================================================
    # INTERNAL ERROR
    # ========================================================

    except Exception as exc:

        logger.exception(
            "Unexpected error in model.solve request_id=%s",
            request_id
        )

        return jsonify({
            "type": "error",
            "message": (
                "An internal solver "
                "error occurred. "
                "Check the server log "
                "for details."
            ),
            "api_request_id": request_id,
        }), 500


# ============================================================
# DEBUG
# ============================================================

@app.route(
    "/debug",
    methods=["GET"]
)
def debug_info():

    if not config.API_DEBUG:

        return _bad(
            "Debug endpoint is disabled.",
            404
        )

    return jsonify({

        "status": "ok",

        "runtime":
            _runtime_diagnostics(),

        "config": {

            "api_debug":
                config.API_DEBUG,

            "datasets_dir":
                str(
                    config.DATASETS_DIR
                ),

            "frontend_dir":
                str(
                    config.FRONTEND_DIR
                ),

            "matrix_ocr_enabled":
                getattr(
                    config,
                    "MATRIX_OCR_ENABLED",
                    True
                ),

            "matrix_ocr_confidence_threshold":
                getattr(
                    config,
                    "MATRIX_OCR_CONFIDENCE_THRESHOLD",
                    0.60
                ),
            "dataset_learning_enabled":
                getattr(
                    config,
                    "DATASET_LEARNING_ENABLED",
                    True
                ),
            "dataset_answer_copy_allowed":
                getattr(
                    config,
                    "DATASET_ANSWER_COPY_ALLOWED",
                    False
                ),
            "knn_enabled":
                getattr(
                    config,
                    "KNN_ENABLED",
                    True
                ),
            "knn_default_k":
                getattr(
                    config,
                    "KNN_DEFAULT_K",
                    5
                ),
        }

    }), 200


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    try:

        stats = model.stats()

        return jsonify({

            "status": "ok",

            "engine":
                getattr(
                    config,
                    "APP_NAME",
                    "M3_ENGINE"
                ),

            "version":
                getattr(
                    config,
                    "APP_VERSION",
                    "unknown"
                ),

            "capabilities": {
                "ocr": (
                    preprocessor.get_ocr_status()
                    if hasattr(preprocessor, "get_ocr_status")
                    else {"ready": False}
                ),
                "dataset_learning": bool(
                    getattr(
                        config,
                        "DATASET_LEARNING_ENABLED",
                        True
                    )
                ),
                "dataset_answer_copy_allowed": bool(
                    getattr(
                        config,
                        "DATASET_ANSWER_COPY_ALLOWED",
                        False
                    )
                ),
                "matrix_ocr_enabled": bool(
                    getattr(
                        config,
                        "MATRIX_OCR_ENABLED",
                        True
                    )
                ),
                "knn_enabled": bool(
                    getattr(
                        config,
                        "KNN_ENABLED",
                        True
                    )
                ),
            },

            **(
                stats
                if isinstance(
                    stats,
                    dict
                )
                else {}
            )

        }), 200

    except Exception:

        logger.exception(
            "Health check failed"
        )

        return jsonify({
            "status": "error",
            "engine": "M3_ENGINE"
        }), 500


# ============================================================
# RELOAD DATASETS
# ============================================================

@app.route(
    "/reload",
    methods=["POST"]
)
def reload_datasets():

    try:

        stats = model.reload()

        return jsonify({

            "status": "reloaded",

            "datasets":
                stats

        }), 200

    except Exception:

        logger.exception(
            "Dataset reload failed"
        )

        return _bad(
            "Dataset reload failed. "
            "Check the server log "
            "for details.",
            500
        )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_too_large(_error):

    return _bad(
        "Request is too large.",
        413
    )


@app.errorhandler(404)
def not_found(_error):

    return _bad(
        "Endpoint not found.",
        404
    )


@app.errorhandler(405)
def method_not_allowed(_error):

    return _bad(
        "HTTP method is not allowed "
        "for this endpoint.",
        405
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    logger.info(
        "M3 API FILE: %s",
        os.path.abspath(__file__)
    )

    logger.info(
        "M3 PROJECT ROOT: %s",
        PROJECT_ROOT
    )

    logger.info(
        "Starting M3_ENGINE on %s:%s",
        config.API_HOST,
        config.API_PORT
    )

    logger.info(
        "RUNTIME MODULES: %s",
        _runtime_diagnostics()
    )

    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_DEBUG
    )
