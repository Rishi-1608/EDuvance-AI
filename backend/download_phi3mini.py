"""
download_phi3mini.py
====================
Downloads only the MISSING model files for Phi-3-mini-4k-instruct
into your existing models/phi3mini directory.

Your tokenizer files (chat_template.jinja, tokenizer_config.json,
tokenizer.json) are already present and will NOT be overwritten.

Files this script fetches:
  - config.json                 ← required by AutoModelForCausalLM
  - generation_config.json      ← generation defaults
  - special_tokens_map.json     ← special token definitions
  - model.safetensors           ← model weights (~7.6 GB for 4-bit,
                                   ~14 GB for float16 on CPU)

Run from your project root (the folder that contains models/):
    python download_phi3mini.py

Requirements:
    pip install huggingface_hub
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO   = "microsoft/Phi-3-mini-4k-instruct"
LOCAL_DIR = Path("models/phi3mini")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Files that MUST be present for the model to load.
# We skip tokenizer.json / tokenizer_config.json / chat_template.jinja
# because you already have those.
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
]

# ── Step 1: download config + generation files ────────────────────────────────
print("Downloading config files …")
for filename in REQUIRED_FILES:
    dest = LOCAL_DIR / filename
    if dest.exists():
        print(f"  [skip] {filename} already exists")
        continue
    print(f"  Downloading {filename} …")
    hf_hub_download(
        repo_id   = HF_REPO,
        filename  = filename,
        local_dir = str(LOCAL_DIR),
        local_dir_use_symlinks = False,
    )
    print(f"  [done] {filename}")

# ── Step 2: download model weights ───────────────────────────────────────────
# Phi-3-mini-4k-instruct is typically split across 2 safetensors shards.
# We try to detect shard filenames automatically first; fall back to
# snapshot_download for the whole model if listing fails.
print("\nDownloading model weights …")
print("(This may take a while — the weights are ~7–14 GB depending on dtype)")

try:
    from huggingface_hub import list_repo_files

    weight_files = [
        f for f in list_repo_files(HF_REPO)
        if f.endswith(".safetensors") or f.endswith(".bin")
    ]

    if not weight_files:
        raise ValueError("No weight files found in repo listing")

    for wf in weight_files:
        dest = LOCAL_DIR / wf
        if dest.exists():
            print(f"  [skip] {wf} already exists")
            continue
        print(f"  Downloading {wf} …")
        hf_hub_download(
            repo_id   = HF_REPO,
            filename  = wf,
            local_dir = str(LOCAL_DIR),
            local_dir_use_symlinks = False,
        )
        print(f"  [done] {wf}")

except Exception as exc:
    print(f"  Auto-detect failed ({exc}), falling back to snapshot_download …")
    snapshot_download(
        repo_id           = HF_REPO,
        local_dir         = str(LOCAL_DIR),
        local_dir_use_symlinks = False,
        ignore_patterns   = [
            # skip tokenizer files you already have
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            # skip pytorch bin if safetensors present (saves space)
            "*.msgpack",
            "flax_model*",
            "tf_model*",
            "rust_model*",
            "*.ot",
        ],
    )

# ── Step 3: verify ────────────────────────────────────────────────────────────
print("\nVerifying local directory …")
files = sorted(LOCAL_DIR.iterdir())
for f in files:
    size_mb = f.stat().st_size / (1024 ** 2)
    print(f"  {f.name:<45} {size_mb:>8.1f} MB")

missing = []
for req in REQUIRED_FILES:
    if not (LOCAL_DIR / req).exists():
        missing.append(req)
has_weights = any(
    f.suffix in (".safetensors", ".bin") for f in LOCAL_DIR.iterdir()
)
if not has_weights:
    missing.append("model weights (.safetensors or .bin)")

if missing:
    print(f"\n[WARNING] Still missing: {missing}")
    print("  Try running again, or manually download from:")
    print(f"  https://huggingface.co/{HF_REPO}/tree/main")
else:
    print("\n[OK] All required files present — model should load correctly.")
    print("Start your server with:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")