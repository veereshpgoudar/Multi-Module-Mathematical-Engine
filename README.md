# M3_ENGINE 🧮

> A dataset-driven AI math solver with a confidence-ranked, multilingual OCR image-question pipeline.  
> **No API keys required.**

---

## Project Structure

```
M3_ENGINE/
├── frontend/
│   └── index.html          ← glassmorphism dark UI (served by Flask)
├── datasets/
│   ├── algebra.json
│   ├── calculus.json
│   ├── statistics.json
│   └── trigonometry.json
├── model/
│   ├── model.py            ← orchestrator (OCR → classify → solve)
│   ├── solver.py           ← Tier-1 AST arithmetic + Tier-2 dataset match
│   └── classifier.py       ← keyword + TF-IDF subject classifier
├── utils/
│   ├── dataset_loader.py   ← checksum hot-reload JSON loader
│   └── preprocessor.py     ← Tesseract OCR + text normalisation
├── api/
│   └── app.py              ← Flask REST API
├── config.py               ← central configuration
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Install Tesseract (system binary)

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install tesseract-ocr -y
```

**macOS (Homebrew)**
```bash
brew install tesseract
```

**Windows**  
Download the installer from https://github.com/UB-Mannheim/tesseract/wiki  
Then add the install directory to your `PATH`, or set the env var:
```
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

For Kannada image questions, install the Kannada `kan.traineddata` file in
Tesseract's `tessdata` directory. The application requests `eng+kan` by
default and safely uses the installed subset if Kannada has not been added.

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add your datasets

Place your JSON dataset files in the `datasets/` folder:
- `datasets/algebra.json`
- `datasets/calculus.json`
- `datasets/statistics.json`
- `datasets/trigonometry.json`

Each file must be a JSON **array** of objects with this shape:
```json
[
  {
    "question": "Solve 2x + 3 = 7",
    "solution": [
      { "step": 1, "description": "Subtract 3 from both sides", "calculation": "2x = 4" },
      { "step": 2, "description": "Divide both sides by 2",     "result": "x = 2" }
    ]
  }
]
```

### 4 — Run the server

```bash
cd M3_ENGINE
python api/app.py
```

Open your browser at **http://localhost:5000**

---

## How It Works

```
User types / uploads image
         │
         ▼
[Advanced Tesseract OCR]  ← EXIF rotation, deskew, adaptive thresholding,
         │                    language selection and token-confidence ranking
         │                    extracts the complete question text
         ▼
[Subject Classifier]
   Pass 1: keyword heuristics  (algebra / calculus / statistics / trigonometry)
   Pass 2: TF-IDF cosine similarity against dataset question corpora
         │
         ▼
[Solver — Tier 1]   AST-safe arithmetic evaluator
   "evaluate 7.56 / (7 * 6)" → instant numeric result
         │
         ▼  (if not pure arithmetic)
[Solver — Tier 2]   TF-IDF cosine similarity search
   Finds the closest matching question in the classified dataset
   Returns its step-by-step solution
         │
         ▼
[Flask API]  JSON response → rendered in the glassmorphism frontend
```

---

## API Reference

### `POST /solve`

**Form data (multipart) or JSON:**

| Field      | Type   | Required | Description                                      |
|------------|--------|----------|--------------------------------------------------|
| `question` | string | optional | Typed question text                              |
| `image`    | file   | optional | Image file (.jpg/.png/.webp) for OCR             |
| `subject`  | string | optional | Force subject: `algebra` `calculus` `statistics` `trigonometry` |

At least one of `question` or `image` must be provided.

**Response (JSON):**
```json
{
  "type": "dataset_match",
  "subject": "statistics",
  "matched_question": "Given Cov(X,Y) = 7.56 ...",
  "confidence": 87.4,
  "solution_steps": [ ... ],
  "ocr_text": "...",
  "input_text": "..."
}
```

`type` is one of: `dataset_match` | `arithmetic` | `no_match` | `error`

### OCR behavior

The reader preserves the complete recognised question instead of extracting
only a short equation. This gives the classifier context such as `find`,
`given`, or `calculate`, and lets the solver pick the intended operation.

- At most four generic OCR attempts are used per image (two preprocessing
  variants × two layout modes), rather than repeatedly OCRing every rotation.
- EXIF rotation and light camera skew are corrected before recognition.
- Each candidate is ranked using word-level Tesseract confidence, token count,
  mathematical/question context, and OCR-noise penalties.
- Matrix handwriting remains on its conservative dedicated route; ordinary
  algebra text is no longer blocked by a false matrix detection.
- `GET /health` includes `capabilities.ocr`, which reports the installed
  languages, engine readiness, and any requested language packs that are
  missing.

### `GET /health`
Returns server status and per-subject dataset entry counts.

### `POST /reload`
Force hot-reload of all dataset files without restarting the server.

---

## Configuration (`config.py`)

| Variable            | Default        | Description                              |
|---------------------|----------------|------------------------------------------|
| `TESSERACT_CMD`     | `"tesseract"`  | Path to Tesseract binary                 |
| `OCR_RESIZE_FACTOR` | `2.5`          | Upscale factor before OCR                |
| `OCR_LANG`          | `eng+kan`      | Tesseract language packs to use          |
| `OCR_MIN_TOKEN_CONFIDENCE` | `35`    | Word confidence floor for candidate text |
| `OCR_MAX_DIMENSION` | `2400`         | Largest image side processed by OCR      |
| `OCR_ENABLE_DESKEW` | `true`         | Correct small camera-skew angles         |
| `MATCH_THRESHOLD`   | `0.15`         | Minimum cosine similarity for a match    |
| `ENABLE_HOT_RELOAD` | `True`         | Re-read datasets on checksum change      |
| `API_PORT`          | `5000`         | Flask listen port                        |
| `API_DEBUG`         | env `M3_DEBUG` | Enable Flask debug mode                  |

---

## Dataset Format Tips

- The more questions in each dataset, the better the matching accuracy.
- Question text should be natural English (not LaTeX).  
  Good: `"Compute the Pearson Correlation Coefficient given Cov(X,Y) = 5 ..."`
- Solution steps support keys: `description`, `formula`, `calculation`, `result`, `interpretation`.
- All keys are optional — include only what applies to each step.

---

## License

MIT — free to use and modify.
