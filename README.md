# Video Scene Analyzer

Video Scene Analyzer is a Python application designed to process videos into text transcripts and event logs. It automates the extraction of dialogue and visual events using a true natively multimodal (Omni) model.

## Overview

The pipeline consists of the following components:

1. **Scene Detection**: Uses `PySceneDetect` to split the video into meaningful, physical, scene-based chunks. This avoids arbitrary slicing, providing natural context boundaries.
2. **Local Omni Processing on NVIDIA GPU**: Each physical chunk is passed directly to the local vision/Omni LLM (`huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated`).
   - **GPU Acceleration**: The pipeline natively supports running large models efficiently using `device_map="auto"` and `torch.bfloat16`, capable of fitting a 30B parameter model natively onto high-VRAM NVIDIA GPUs (like A100 or H100 80GB).
   - **Unified Extraction**: A single prompt asks the model to output BOTH the dialogue transcription and the visual event log for the chunk in a structured JSON response.
   - **Contextual Continuity**: The processor passes the descriptions of previous scenes as context to maintain continuity.
3. **Output Artifacts**: Finally, the analyzer produces two main text files:
   - `transcript.txt`: A timeline of transcribed dialogue.
   - `event_log.txt`: A timeline describing the visual events and actions that occurred in the video.

## Project Structure

- `src/video_scene_analyzer/`
  - `__init__.py`: Package initialization.
  - `cli.py`: Command-line interface entry point.
  - `core.py`: The main orchestrator (`VideoAnalyzer`) that ties the components together.
  - `scene_processor.py`: Wrapper for `PySceneDetect` and video chunk cutting.
  - `omni_processor.py`: Direct CUDA-accelerated local inference client utilizing `transformers` and `device_map="auto"`.

## Installation

This project uses `pyproject.toml` for modern dependency management. You must have a CUDA-compatible environment.

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

- **FFmpeg:** You must have `ffmpeg` installed on your system to extract and cut the video chunks.
  - **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **NVIDIA GPU:** A high-VRAM GPU (e.g., A100 or H100) is recommended to comfortably fit the 30B parameter model natively in `bfloat16`. Make sure CUDA drivers and toolkits are correctly installed.

## Usage

Once installed and the environment is configured, you can run the tool:

```bash
video-scene-analyzer path/to/video.mp4 \
  --vision-model huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated \
  --context-window 3 \
  --output-transcript transcript.txt \
  --output-event-log event_log.txt
```

### Options

- `video_path`: (Required) Path to the input video file.
- `--vision-model`: The name of the Omni model to target. Default: `huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated`.
- `--context-window`: Number of previous scene descriptions to include as context. Default: `3`.
- `--scene-threshold`: Content detection threshold for PySceneDetect. Default: `27.0`.
- `--output-transcript`: Path to save the dialogue transcript. Default: `transcript.txt`.
- `--output-event-log`: Path to save the event log. Default: `event_log.txt`.
