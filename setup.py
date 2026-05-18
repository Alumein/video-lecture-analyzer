from setuptools import setup, find_packages

setup(
    name="video-lecture-analyzer",
    version="0.1.0",
    description="Automated Video Lecture Analyzer - Transforms lecture videos into structured study guides",
    author="Ahmet Abdulgaffar Fettahoglu",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "faster-whisper>=1.0.0",
        "pyannote.audio>=3.1.0",
        "google-generativeai>=0.8.0",
        "ffmpeg-python>=0.2.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "lecture-analyzer=scripts.run_analysis:main",
        ],
    },
)
