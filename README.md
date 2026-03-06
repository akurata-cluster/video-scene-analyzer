# Video Scene Analyzer

Video Scene Analyzer is a Python application designed to process videos into text transcripts and event logs. It automates the extraction of dialogue and visual events using modern AI models.

## Overview

The pipeline consists of the following components:

1. **Scene Detection**: Uses `PySceneDetect` to split the video into meaningful, scene-based chunks. This avoids arbitrary slicing, providing natural context boundaries.
2. **Audio Processing**: For each scene, audio is extracted using `ffmpeg` and transcribed using `faster-whisper`. This generates a dialogue transcript.
3. **Vision Processing**: For each scene, a set of evenly spaced frames is extracted using OpenCV and passed to a vision LLM. Specifically, the pipeline is configured to target the `huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated` model via an OpenAI-compatible API endpoint (like vLLM).
   - **Contextual Continuity**: The vision processor passes the descriptions of the previous $N$ scenes to the model to ensure it maintains continuity and tracks subjects across scene boundaries.
4. **Output Artifacts**: Finally, the analyzer produces two main text files:
   - `transcript.txt`: A timeline of transcribed dialogue.
   - `event_log.txt`: A timeline describing the visual events and actions that occurred in the video.

## Project Structure

- `src/video_scene_analyzer/`
  - `__init__.py`: Package initialization.
  - `cli.py`: Command-line interface entry point.
  - `core.py`: The main orchestrator (`VideoAnalyzer`) that ties the components together.
  - `scene_processor.py`: Wrapper for `PySceneDetect`.
  - `audio_processor.py`: Audio extraction and `faster-whisper` transcription.
  - `vision_processor.py`: Frame extraction and OpenAI-compatible Vision API integration.

## Installation

This project uses `pyproject.toml` for modern dependency management.

```bash
# Clone the repository
git clone https://github.com/akurata-cluster/video-scene-analyzer.git
cd video-scene-analyzer

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install the application and its dependencies
pip install -e .
```

### System Requirements

You must have `ffmpeg` installed on your system to extract audio from the video.

- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

## Usage

Once installed, you can use the `video-scene-analyzer` command-line tool.

```bash
video-scene-analyzer path/to/video.mp4 \
  --vision-url http://localhost:8000/v1 \
  --vision-model huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated \
  --whisper-model base \
  --context-window 3 \
  --output-transcript transcript.txt \
  --output-event-log event_log.txt
```

### Options

- `video_path`: (Required) Path to the input video file.
- `--whisper-model`: The Faster Whisper model size to use (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`). Default: `base`.
- `--vision-url`: The OpenAI-compatible vision API base URL (e.g., your local vLLM instance). Default: `http://localhost:8000/v1`.
- `--vision-model`: The name of the vision model to target. Default: `huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated`.
- `--vision-api-key`: API key for the vision endpoint (if required by your provider). Default: `dummy`.
- `--context-window`: Number of previous scene descriptions to include as context for the vision model. Default: `3`.
- `--scene-threshold`: Content detection threshold for PySceneDetect. Default: `27.0`.
- `--output-transcript`: Path to save the dialogue transcript. Default: `transcript.txt`.
- `--output-event-log`: Path to save the event log. Default: `event_log.txt`.
