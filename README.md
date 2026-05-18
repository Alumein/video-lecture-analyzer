# Automated Video Lecture Analyzer

An automated tool that "watches" and "listens" to lecture videos to extract meaningful structure. It transforms standard, flat video files into interactive and highly readable study guides.

## Overview

Lecture recordings are highly valuable but often long and unstructured, making it difficult for students to find specific information. This system automatically detects when slides change, creates topic-based segments, and generates smart summaries for each part of the lesson.

## Architecture

The project uses a **Hybrid Processing Architecture** to balance computational efficiency and cognitive performance.

```
┌─────────────────┐     ┌──────────────────────────────────┐     ┌──────────────────┐
│                 │     │       LOCAL PROCESSING           │     │ CLOUD PROCESSING │
│  Lecture Video  │────▶│                                  │────▶│                  │
│  (.mp4 / .mkv)  │     │  ┌────────────┐ ┌─────────────┐ │     │  Google Gemini   │
│                 │     │  │  OpenCV     │ │Faster-Whisper│ │     │  API (LLM)       │
└─────────────────┘     │  │  Slide      │ │Speech-to-Text│ │     │                  │
                        │  │  Detection  │ │             │ │     │  - Topic Segment. │
                        │  └────────────┘ └─────────────┘ │     │  - Summarization  │
                        │                 ┌─────────────┐ │     │  - Key Extraction │
                        │                 │  pyannote   │ │     └────────┬─────────┘
                        │                 │  Speaker    │ │              │
                        │                 │  Diarization│ │              │
                        │                 └─────────────┘ │              │
                        └──────────────────────────────────┘              │
                                                                         ▼
                                                              ┌──────────────────┐
                                                              │   Unified JSON   │
                                                              │   Study Guide    │
                                                              └──────────────────┘
```

## Project Structure

```
video-lecture-analyzer/
├── src/
│   ├── visual/              # Slide detection & frame analysis
│   │   ├── __init__.py
│   │   ├── slide_detector.py    # OpenCV frame differencing
│   │   └── frame_extractor.py   # Key frame extraction
│   ├── audio/               # Speech processing pipeline
│   │   ├── __init__.py
│   │   ├── transcriber.py       # Faster-Whisper STT
│   │   └── diarizer.py          # pyannote speaker diarization
│   ├── nlp/                 # Cloud NLP processing
│   │   ├── __init__.py
│   │   ├── gemini_client.py     # Gemini API integration
│   │   ├── topic_segmenter.py   # Topic segmentation
│   │   └── summarizer.py        # Key sentence extraction
│   ├── integration/         # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Main processing pipeline
│   │   └── merger.py            # Timestamp merging & sync
│   └── utils/               # Shared utilities
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── logger.py            # Logging setup
│       └── video_utils.py       # Video file helpers
├── tests/                   # Unit & integration tests
│   ├── __init__.py
│   ├── test_slide_detector.py
│   ├── test_transcriber.py
│   └── test_pipeline.py
├── config/
│   └── default.yaml         # Default configuration
├── scripts/
│   └── run_analysis.py      # CLI entry point
├── data/
│   ├── input/               # Place lecture videos here
│   └── output/              # Generated study guides
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── requirements-dev.txt     # Development dependencies
├── setup.py                 # Package setup
├── .gitignore
├── .env.example             # Environment variables template
├── LICENSE
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- FFmpeg (for audio extraction)
- CUDA-compatible GPU (recommended for Faster-Whisper)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/video-lecture-analyzer.git
cd video-lecture-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your Gemini API key
```

### Configuration

Edit `config/default.yaml` or set environment variables:

```yaml
# Slide detection
visual:
  threshold: 30.0          # Frame difference threshold
  min_interval: 2.0        # Minimum seconds between slides

# Audio processing
audio:
  whisper_model: "large-v3" # Whisper model size
  language: "auto"          # Auto-detect language
  device: "cuda"            # cuda or cpu

# NLP
nlp:
  gemini_model: "gemini-2.0-flash"
  max_topics: 20
```

## Usage

```bash
# Basic usage
python scripts/run_analysis.py --input data/input/lecture.mp4

# With options
python scripts/run_analysis.py \
  --input data/input/lecture.mp4 \
  --output data/output/lecture_guide.json \
  --language tr \
  --whisper-model large-v3
```

## Output Format

The system generates a structured JSON study guide:

```json
{
  "metadata": {
    "source_file": "lecture.mp4",
    "duration": "01:15:23",
    "processing_time": "00:12:45",
    "language": "tr"
  },
  "segments": [
    {
      "id": 1,
      "topic": "Introduction to Neural Networks",
      "start_time": "00:00:00",
      "end_time": "00:08:32",
      "slide_image": "slides/slide_001.png",
      "transcript": "...",
      "speaker": "Dr. Kaya",
      "summary": "Overview of neural network fundamentals...",
      "key_points": [
        "Neural networks mimic biological neurons",
        "Three main types: feedforward, recurrent, convolutional"
      ]
    }
  ]
}
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Slide Detection Precision/Recall | >= 90% |
| Processing Latency (60 min video) | < 15 minutes |
| Topic Segmentation F1-score | >= 0.8 |
| Word Error Rate (WER) | <= 15% |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Slide Detection | OpenCV (frame differencing) |
| Speech-to-Text | Faster-Whisper |
| Speaker Diarization | pyannote.audio |
| NLP / Summarization | Google Gemini API |
| Configuration | PyYAML |
| CLI | argparse |

## License

This project is developed as part of the CSE496 course at Gebze Technical University.

## Author

**Ahmet Abdulgaffar Fettahoglu**
Advisor: Dr. Gokhan KAYA
Gebze Technical University, 2026
