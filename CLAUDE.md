# Video Lecture Analyzer

Graduation project (GTU CSE496) by Ahmet Abdulgaffar Fettahoğlu, advisor Dr. Gökhan Kaya. Transforms recorded lecture videos into interactive study guides via slide detection, transcription, and AI-powered topic segmentation.

## Architecture

Four-stage fully-local pipeline:

1. **Visual** (`src/visual/`) — OpenCV slide detection + OCR
2. **Audio** (`src/audio/`) — Faster-Whisper transcription
3. **NLP** (`src/nlp/`) — Ollama (LLaMA 3.1 / qwen2.5) for topic segmentation
4. **Integration** (`src/integration/`) — Merges everything into segmented JSON
5. **Web** (`web/`) — Flask app with interactive video player + analysis panel
6. **Evaluation** (`src/evaluation/`) — Automated quality metrics

Entry points:
- CLI: `scripts/run_analysis.py`
- Web: `python web/app.py` → http://localhost:5000

## Tech Stack

- Python 3.11
- OpenCV, NumPy, FFmpeg (audio extraction)
- Faster-Whisper on CUDA (float16, medium model)
- Ollama HTTP API (localhost:11434) — primary LLM
- Gemini API — optional fallback (quota-limited, rarely used)
- Tesseract OCR via pytesseract (optional, gracefully skipped if missing)
- Flask + Jinja2, vanilla JS frontend (no framework)
- pyannote.audio — DISABLED, not used

Hardware target: NVIDIA RTX 3070 Laptop, 8GB VRAM.

## Key Design Decisions

### Slide Detection (`src/visual/slide_detector.py`)
- **Sequential read with `cap.grab()`** instead of `cv2.CAP_PROP_POS_FRAMES` seek — H.264 seek is extremely slow, sequential is 10x+ faster
- **Sample every 1 second** (configurable via `sample_interval`)
- **Dual-method detection**:
  - Pixel diff (`mean absolute difference` on grayscale) catches major visual changes
  - **Canny edge diff** catches text-only changes (same background, different text) — critical for lecture slides where pHash fails
  - Triggers if EITHER exceeds its adaptive threshold
- **Auto thresholds**: `pixel = mean + 1.8*std (min 1.5)`, `edge = mean + 1.5*std (min 0.50)`
- **Left 75% crop** before any comparison — ignores Zoom/Teams right-side webcam panel
- **Duplicate detection via edge fingerprints** (NOT pHash). pHash is background-dominated and misses text changes. Edge-based 256x144 fingerprint compares text/structure directly. Threshold: 0.90 similarity.

### Transcription (`src/audio/transcriber.py`)
- Faster-Whisper medium model on CUDA float16
- VAD filter enabled, word-level timestamps
- `batch_size` parameter NOT supported in user's faster-whisper version — do NOT add it

### NLP (`src/nlp/topic_segmenter.py`)
- Transcript split into 8K-char chunks with 3-line overlap
- Each chunk gets EXPLICIT time range in prompt (`time_range_start_sec` to `time_range_end_sec`)
- Detailed prompt with BAD/GOOD examples enforcing specific titles
- Summaries required 5-8 sentences with concrete examples
- Post-processing `_fill_coverage_gaps()` extends segments to eliminate gaps
- Model selectable from web UI (default `llama3.1:8b`, optional `qwen2.5:14b`)

### Integration (`src/integration/`)
- `pipeline.py` orchestrates 10 progress steps with callback for web UI
- `merger.py merge_by_llm_topics()` uses LLM-determined topic boundaries
- Each segment includes ALL slides in its time range (list, not just one)
- `slide_text` field contains concatenated OCR text from those slides

### Web Interface (`web/app.py`)
- Background threading via `threading.Thread`
- Job state stored in module-level `jobs` dict (not thread-safe but fine for single-user)
- **`use_reloader=False` is REQUIRED** — Flask reload kills the Whisper subprocess
- XHR polling every 2-3s on `/status/<name>`
- Routes: `/`, `/upload`, `/analyze/<name>`, `/reanalyze/<name>`, `/delete/<name>`, `/status/<name>`, `/lecture/<name>`, `/slides/<name>/<file>`, `/video/<name>`
- Delete only removes results, KEEPS video file
- All UI text is English

### Evaluation (`src/evaluation/evaluator.py`)
Four criteria with mixed measurement approaches:
- **Latency** — Directly measured, normalized to 60-min projection
- **WER** — LLM-as-judge estimate from transcript samples
- **Slide Duplicates** — Programmatic pHash comparison across all saved slides (key: `slide_duplicates`)
- **Topic F1** — LLM-as-judge on segmentation quality

Disclaimer always shown: WER and F1 are LLM estimates, not ground-truth.

## Success Criteria (Project Requirements)

1. No duplicate slides in output (programmatic check)
2. Processing < 15 min for 60-min video (measured)
3. Topic segmentation F1 ≥ 0.8 (LLM-judged estimate)
4. Transcription WER ≤ 15% (LLM-judged estimate)

## Common Issues & Fixes

- **OCR skipped warning**: Tesseract OCR program not installed on system (pytesseract package alone is not enough). Install from https://github.com/UB-Mannheim/tesseract/wiki and ensure "Add to PATH" is checked.
- **Slide detection misses transitions with same background + different text**: Make sure `slide_detector.py` has `_compute_edge_fingerprint` function and dual-method detection. Old pHash-only code misses these.
- **Topic segmentation low coverage**: Check `_fill_coverage_gaps` is being called and chunk time ranges are passed to LLM.
- **Whisper `batch_size` TypeError**: User's faster-whisper version doesn't support it. Remove the parameter.
- **Pipeline killed mid-run**: Ensure `app.run(use_reloader=False)`.

## File Locations

- Config: `config/default.yaml`
- Environment: `.env` (Ollama URL, Gemini key, model overrides)
- Input videos: `data/input/<name>.mp4`
- Outputs: `data/output/<name>/` (slides/, audio.wav, <name>.srt, <name>_result.json)

## Development Guidelines

- Keep responses in code/UI in **English only**
- Don't reintroduce pyannote.audio or Gemini as primary
- When debugging slide detection, ask user to run the diagnostic script that prints top 30 edge diff spikes — this gives real data instead of guessing thresholds
- When `_compute_edge_fingerprint` modifications are needed, verify by `grep` after writing
- Always clear `__pycache__` after changes to avoid stale imports
- Test changes on real lecture videos in `data/input/` (CSE312, CSE341 are reference videos)

## Out of Scope

Don't add:
- Cloud LLM as default (privacy + quota issues)
- Speaker diarization (overkill for single-speaker lectures)
- Real-time/streaming analysis (batch-only project scope)
- Authentication or multi-user features (single-user local app)
