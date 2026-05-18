"""Flask web application for Video Lecture Analyzer.

Run: python web/app.py
Open: http://localhost:5000
"""

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, redirect, url_for,
)
from werkzeug.utils import secure_filename

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.video_utils import get_video_info, format_timestamp
from src.integration.pipeline import Pipeline

logger = setup_logger("web")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# Processing tracker: { "ders1": { running, step, progress, done, error } }
jobs = {}


# ── Helpers ──────────────────────────────────────────────

def get_lectures():
    """Scan data dirs and build lecture list with status."""
    lectures = []
    for v in sorted(INPUT_DIR.glob("*")):
        if v.suffix.lower() not in ALLOWED_EXT:
            continue
        nm = v.stem
        out = OUTPUT_DIR / nm
        result_file = out / f"{nm}_result.json"

        lec = {
            "name": nm,
            "filename": v.name,
            "size_mb": round(v.stat().st_size / 1024 / 1024, 1),
            "processed": result_file.exists(),
            "processing": nm in jobs and jobs[nm].get("running", False),
            "error": jobs.get(nm, {}).get("error"),
        }

        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            lec["metadata"] = data.get("metadata", {})
            lec["segments"] = data.get("segments", [])

        lectures.append(lec)
    return lectures


def start_analysis(lecture_name, model=None):
    """Launch pipeline in background thread.

    Args:
        lecture_name: Name of the lecture (without extension).
        model: Optional Ollama model to use (overrides config).
    """
    video_path = None
    for ext in ALLOWED_EXT:
        p = INPUT_DIR / f"{lecture_name}{ext}"
        if p.exists():
            video_path = p
            break

    if not video_path:
        jobs[lecture_name] = {"running": False, "error": "Video not found"}
        return

    jobs[lecture_name] = {
        "running": True,
        "step": "Starting...",
        "progress": 0,
        "done": False,
        "model": model,
    }

    def worker():
        try:
            config = load_config()
            # Override Ollama model if specified
            if model:
                config.setdefault("nlp", {})["ollama_model"] = model
                logger.info(f"Using model: {model}")

            out = OUTPUT_DIR / lecture_name
            pipeline = Pipeline(config=config)

            # Real-time progress callback
            def on_progress(step, pct):
                jobs[lecture_name]["step"] = step
                jobs[lecture_name]["progress"] = pct

            result = pipeline.run(str(video_path), str(out), on_progress=on_progress)

            jobs[lecture_name] = {
                "running": False,
                "done": True,
                "step": "Complete!",
                "progress": 100,
                "segments": result["metadata"]["total_segments"],
            }
            logger.info(f"Analysis done: {lecture_name}")

        except Exception as e:
            logger.error(f"Pipeline error [{lecture_name}]: {e}")
            jobs[lecture_name] = {
                "running": False,
                "done": False,
                "error": str(e),
                "step": f"Error: {e}",
                "progress": 0,
            }

    threading.Thread(target=worker, daemon=True).start()


# ── Routes ───────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", lectures=get_lectures())


@app.route("/upload", methods=["POST"])
def upload():
    """Upload video and auto-start analysis."""
    if "video" not in request.files:
        return jsonify({"error": "No video file selected"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Save file
    filepath = INPUT_DIR / filename
    file.save(str(filepath))
    logger.info(f"Uploaded: {filename} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")

    # Auto-start analysis
    lecture_name = Path(filename).stem
    auto = request.form.get("auto_analyze", "true")
    model = request.form.get("model") or None
    if auto == "true":
        start_analysis(lecture_name, model=model)

    return jsonify({
        "success": True,
        "filename": filename,
        "name": lecture_name,
        "auto_started": auto == "true",
    })


@app.route("/analyze/<lecture_name>", methods=["POST"])
def analyze(lecture_name):
    """Manually trigger analysis for an uploaded video."""
    if lecture_name in jobs and jobs[lecture_name].get("running"):
        return jsonify({"error": "Already processing"}), 409
    model = request.form.get("model") or request.json.get("model") if request.is_json else request.form.get("model")
    start_analysis(lecture_name, model=model)
    return jsonify({"success": True})


@app.route("/reanalyze/<lecture_name>", methods=["POST"])
def reanalyze(lecture_name):
    """Re-run analysis (delete old results first)."""
    import shutil
    out = OUTPUT_DIR / lecture_name
    if out.exists():
        shutil.rmtree(out)
    model = None
    if request.is_json:
        model = request.json.get("model")
    else:
        model = request.form.get("model")
    start_analysis(lecture_name, model=model)
    return jsonify({"success": True})


@app.route("/delete/<lecture_name>", methods=["POST"])
def delete(lecture_name):
    """Delete analysis results only, keep the video file."""
    import shutil
    # Only delete output (results, slides, audio)
    out = OUTPUT_DIR / lecture_name
    if out.exists():
        shutil.rmtree(out)
    # Clear job status
    jobs.pop(lecture_name, None)
    return jsonify({"success": True})


@app.route("/status/<lecture_name>")
def status(lecture_name):
    """Poll processing status."""
    return jsonify(jobs.get(lecture_name, {"running": False, "done": False}))


@app.route("/lecture/<lecture_name>")
def lecture_detail(lecture_name):
    """Lecture detail page."""
    result_file = OUTPUT_DIR / lecture_name / f"{lecture_name}_result.json"
    if not result_file.exists():
        return render_template("not_found.html", name=lecture_name), 404
    with open(result_file, "r", encoding="utf-8") as f:
        result = json.load(f)
    return render_template("lecture.html", name=lecture_name, result=result)


@app.route("/slides/<lecture_name>/<filename>")
def serve_slide(lecture_name, filename):
    """Serve slide images."""
    return send_from_directory(str(OUTPUT_DIR / lecture_name / "slides"), filename)


@app.route("/video/<lecture_name>")
def serve_video(lecture_name):
    """Serve video file for browser playback."""
    # Find video file
    for ext in ALLOWED_EXT:
        video_path = INPUT_DIR / f"{lecture_name}{ext}"
        if video_path.exists():
            return send_from_directory(str(INPUT_DIR), video_path.name)
    return jsonify({"error": "Video not found"}), 404


@app.route("/api/lectures")
def api_lectures():
    return jsonify(get_lectures())


@app.route("/api/lecture/<lecture_name>")
def api_lecture(lecture_name):
    result_file = OUTPUT_DIR / lecture_name / f"{lecture_name}_result.json"
    if not result_file.exists():
        return jsonify({"error": "Not found"}), 404
    with open(result_file, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Video Lecture Analyzer")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
