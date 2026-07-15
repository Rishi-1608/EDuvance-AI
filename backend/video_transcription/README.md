# Standalone Video Transcription Tool

A highly optimized, completely standalone command-line tool to extract and transcribe audio from video files using `faster-whisper` and ffmpeg.

## Prerequisites
1. Ensure you have `ffmpeg` installed on your system.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script passing the path to the video file you want to transcribe.

```bash
python transcribe.py /path/to/your/video.mp4
```

### Options

| Argument      | Default | Description |
| ----------- | ----------- | ----------- |
| `video_path`  | **Required** | The path to the video you want to process. |
| `--model`     | `tiny`| Size of the Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `--device`    | `cuda`| Device to run computation on (`cuda` for GPU, `cpu`). |
| `--outdir`    | `outputs`| Directory where text and JSON outputs will be saved. |

### Example 
```bash
python transcribe.py my_lecture.mp4 --model base --device cuda --outdir my_transcripts
```

### Outputs
For a file named `lecture.mp4`, it will automatically create:
- `lecture_transcript.txt`: The raw text of your video.
- `lecture_transcript.json`: A structured JSON file containing chunks, timestamps, and confidence scores.
