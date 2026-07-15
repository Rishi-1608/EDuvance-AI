import os
import argparse
import subprocess
import json
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ Error: faster-whisper not installed.\nPlease run: pip install -r requirements.txt")
    exit(1)

def extract_audio(video_path: str, audio_path: str):
    """
    Extracts audio from video using ffmpeg.
    Converts it to 16 kHz Mono WAV format as expected by Whisper.
    """
    command = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path
    ]
    print(f"🎬 Extracting audio from '{os.path.basename(video_path)}'...")
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ Audio extraction complete.")
    except subprocess.CalledProcessError:
        print("❌ Error extracting audio. Make sure ffmpeg is installed and the video is valid.")
        exit(1)

def transcribe_audio(audio_path: str, model_size="tiny", device="cuda"):
    """
    Transcribes audio using the highly optimized faster-whisper engine.
    Uses VAD (Voice Activity Detection) to skip silence and prevent hallucinations.
    """
    print(f"🧠 Loading faster-whisper model '{model_size}' on '{device}'...")
    try:
        # We attempt Int8 precision for VRAM efficiency if on a supported GPU
        compute_type = "int8_float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"⚠️ Failed to load model on {device}: {e}")
        print("🔄 Falling back to CPU mode...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"📝 Transcribing audio...\n")
    # Stream segments using generator logic
    segments, info = model.transcribe(
        audio_path, 
        beam_size=1, # Greedy decoding for speed
        vad_filter=True, 
        vad_parameters={"min_silence_duration_ms": 500, "threshold": 0.5}
    )

    print(f"🗣️ Detected language '{info.language}' with {info.language_probability*100:.1f}% confidence")
    print("-" * 50)
    
    full_text = []
    output_segments = []
    
    for segment in segments:
        print(f"[{segment.start:05.1f}s -> {segment.end:05.1f}s] {segment.text.strip()}")
        full_text.append(segment.text.strip())
        output_segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
        
    print("-" * 50)
    return " ".join(full_text), output_segments

def main():
    parser = argparse.ArgumentParser(description="Standalone Video Transcription Tool")
    parser.add_argument("video_path", help="Path to the video file to transcribe")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large-v3"], 
                        help="Whisper model size. Larger = better accuracy, slower speed (default: tiny).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], 
                        help="Hardware to run on: cuda for GPU, cpu for processor.")
    parser.add_argument("--outdir", default="outputs", help="Directory to save the transcripts.")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.video_path):
        print(f"❌ Error: Video file not found at '{args.video_path}'")
        return

    os.makedirs(args.outdir, exist_ok=True)
    stem = Path(args.video_path).stem
    temp_audio = os.path.join(args.outdir, f"{stem}_temp.wav")
    
    try:
        extract_audio(args.video_path, temp_audio)
        text, segments = transcribe_audio(temp_audio, model_size=args.model, device=args.device)
        
        txt_out = os.path.join(args.outdir, f"{stem}_transcript.txt")
        json_out = os.path.join(args.outdir, f"{stem}_transcript.json")
        
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(text)
            
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"language": "auto", "text": text, "segments": segments}, f, indent=2, ensure_ascii=False)
            
        print(f"\n🎉 Transcription successfully saved to:")
        print(f" 📄 Text: {os.path.abspath(txt_out)}")
        print(f" 📊 JSON: {os.path.abspath(json_out)}")
    
    finally:
        # Cleanup temp audio
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    main()
