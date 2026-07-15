# """
# fix_phi3_config.py
# ==================
# Fixes the 'configuration_phi3.py not found' error by removing the
# auto_map block from config.json.

# Background
# ----------
# Phi-3 was originally released with custom Python files (configuration_phi3.py,
# modeling_phi3.py). If you downloaded ONLY config.json without those .py files,
# transformers tries to find them and crashes.

# Since transformers >= 4.40, Phi-3 is natively supported — no custom files
# needed. This script strips the auto_map entries from config.json so
# AutoModelForCausalLM uses the built-in Phi3ForCausalLM class instead.

# Run from your project root:
#     python fix_phi3_config.py
# """

# import json
# import shutil
# from pathlib import Path

# CONFIG_PATH = Path("models/phi3mini/config.json")

# if not CONFIG_PATH.exists():
#     print(f"ERROR: {CONFIG_PATH} not found.")
#     print("Make sure you are running this from your project root (D:\\multi_model\\)")
#     raise SystemExit(1)

# # ── Backup original ────────────────────────────────────────────────────────────
# backup = CONFIG_PATH.with_suffix(".json.bak")
# shutil.copy2(CONFIG_PATH, backup)
# print(f"Backed up original to: {backup}")

# # ── Load and patch ─────────────────────────────────────────────────────────────
# with open(CONFIG_PATH, encoding="utf-8") as f:
#     cfg = json.load(f)

# print(f"\nOriginal model_type: {cfg.get('model_type')}")
# print(f"Original auto_map:   {cfg.get('auto_map')}")

# # Remove the auto_map block — this is the only required change.
# # With auto_map gone, AutoConfig resolves "phi3" -> Phi3Config natively
# # (built into transformers >= 4.40).
# removed_keys = []
# for key in ("auto_map",):
#     if key in cfg:
#         del cfg[key]
#         removed_keys.append(key)

# # Ensure model_type is exactly "phi3" (lowercase, no variant suffix).
# # Some downloaded configs have "phi3" already; a few have "phi-3" with a dash.
# original_model_type = cfg.get("model_type", "")
# if original_model_type.lower() in ("phi-3", "phi_3", "phi3"):
#     cfg["model_type"] = "phi3"

# # ── Write patched config ───────────────────────────────────────────────────────
# with open(CONFIG_PATH, "w", encoding="utf-8") as f:
#     json.dump(cfg, f, indent=2, ensure_ascii=False)

# print(f"\nPatched config.json:")
# print(f"  model_type : {cfg.get('model_type')}")
# print(f"  auto_map   : {cfg.get('auto_map', '(removed)')}")
# if removed_keys:
#     print(f"  Removed keys: {removed_keys}")

# # ── Verify transformers can now resolve the model ──────────────────────────────
# print("\nVerifying with AutoConfig …")
# try:
#     from transformers import AutoConfig
#     config = AutoConfig.from_pretrained("models/phi3mini", trust_remote_code=False)
#     print(f"  AutoConfig loaded OK: {type(config).__name__}")
#     print(f"  architectures: {getattr(config, 'architectures', 'n/a')}")
#     print(f"  hidden_size:   {getattr(config, 'hidden_size', 'n/a')}")
#     print(f"  num_layers:    {getattr(config, 'num_hidden_layers', 'n/a')}")
#     print("\n[OK] config.json is now correct. Run your server again.")
# except Exception as exc:
#     print(f"  AutoConfig still failed: {exc}")
#     print("\n  Check your transformers version:")
#     import transformers
#     print(f"  transformers=={transformers.__version__}")
#     print("  Phi-3 native support requires transformers >= 4.40.0")
#     print("  Upgrade with:  pip install --upgrade transformers")
from video_pipeline.reasoning.phi3_engine import Phi3Reasoner
llm = Phi3Reasoner(model_id="models/phi3mini", device="cuda", load_in_4bit=True)
result = llm.reason('Return this exact JSON: {"test": "ok", "value": 42}')
print("Parsed:", result)
print("Raw:", llm._last_raw_output)             # e.g. 2.5.1+cu124  