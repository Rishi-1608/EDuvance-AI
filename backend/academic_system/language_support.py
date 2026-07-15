"""
academic_system/language_support.py
=====================================
Multi-language support for the Academic Intelligence System.

Reads the language detected by Whisper, then provides:
  • The correct EasyOCR language code list  (fed into OCRExtractor)
  • A prompt instruction string             (prepended to all LLM prompts)
  • An RTL flag                             (for Arabic / Hebrew / Urdu / Persian)
  • The human-readable language name        (for status / diagnostics)

37 languages supported out of the box.

Usage
-----
    detector = LanguageDetector()

    lang = detector.from_whisper(transcription_dict)
    # → {"code": "hi", "name": "Hindi", "ocr_langs": ["hi", "en"],
    #    "rtl": False, "prompt_instruction": "Respond in Hindi..."}

    # patch any prompt string before passing to LLM
    patched_prompt = detector.patch_prompt(original_prompt, lang)

    # pass OCR languages to a new OCRExtractor instance
    ocr = OCRExtractor(use_gpu=True, languages=lang["ocr_langs"])
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# (whisper_code, human_name, easyocr_codes, rtl)
_TABLE: List[Tuple[str, str, List[str], bool]] = [
    ("en",    "English",          ["en"],          False),
    ("hi",    "Hindi",            ["hi", "en"],    False),
    ("zh",    "Chinese (Simp.)",  ["ch_sim", "en"],False),
    ("zh-tw", "Chinese (Trad.)",  ["ch_tra", "en"],False),
    ("ar",    "Arabic",           ["ar", "en"],    True),
    ("fr",    "French",           ["fr", "en"],    False),
    ("de",    "German",           ["de", "en"],    False),
    ("es",    "Spanish",          ["es", "en"],    False),
    ("pt",    "Portuguese",       ["pt", "en"],    False),
    ("ru",    "Russian",          ["ru", "en"],    False),
    ("ja",    "Japanese",         ["ja", "en"],    False),
    ("ko",    "Korean",           ["ko", "en"],    False),
    ("ur",    "Urdu",             ["ur", "en"],    True),
    ("fa",    "Persian",          ["fa", "en"],    True),
    ("tr",    "Turkish",          ["tr", "en"],    False),
    ("it",    "Italian",          ["it", "en"],    False),
    ("nl",    "Dutch",            ["nl", "en"],    False),
    ("pl",    "Polish",           ["pl", "en"],    False),
    ("uk",    "Ukrainian",        ["uk", "en"],    False),
    ("vi",    "Vietnamese",       ["vi", "en"],    False),
    ("th",    "Thai",             ["th", "en"],    False),
    ("bn",    "Bengali",          ["bn", "en"],    False),
    ("ta",    "Tamil",            ["ta", "en"],    False),
    ("te",    "Telugu",           ["te", "en"],    False),
    ("mr",    "Marathi",          ["mr", "en"],    False),
    ("gu",    "Gujarati",         ["gu", "en"],    False),
    ("kn",    "Kannada",          ["kn", "en"],    False),
    ("ml",    "Malayalam",        ["en"],          False),
    ("pa",    "Punjabi",          ["en"],          False),
    ("id",    "Indonesian",       ["id", "en"],    False),
    ("ms",    "Malay",            ["ms", "en"],    False),
    ("he",    "Hebrew",           ["he", "en"],    True),
    ("sv",    "Swedish",          ["sv", "en"],    False),
    ("no",    "Norwegian",        ["no", "en"],    False),
    ("da",    "Danish",           ["da", "en"],    False),
    ("fi",    "Finnish",          ["fi", "en"],    False),
    ("el",    "Greek",            ["el", "en"],    False),
]

_BY_CODE: Dict[str, Dict] = {
    code: {"code": code, "name": name, "ocr_langs": ocr, "rtl": rtl}
    for code, name, ocr, rtl in _TABLE
}
_DEFAULT = _BY_CODE["en"]


class LanguageDetector:
    """
    Converts Whisper's detected language code into everything else the
    pipeline needs to run in that language.
    """

    def from_whisper(self, transcription: Dict) -> Dict:
        """
        Build a language_info dict from a Whisper transcription result.

        Whisper puts the detected language in transcription["language"]
        as a lowercase ISO 639-1 code ("en", "hi", "ar", …).
        """
        code = (transcription or {}).get("language", "en") or "en"
        return self._make(code.lower().strip())

    def from_code(self, code: str) -> Dict:
        """Look up by explicit ISO 639-1 code. Falls back to English."""
        return self._make(code.lower().strip())

    def patch_prompt(self, prompt: str, lang_info: Dict) -> str:
        """
        Prepend the language instruction to a prompt string when the
        lecture language is not English.

        Always call this before passing a prompt to the LLM.
        """
        instruction = lang_info.get("prompt_instruction", "")
        if not instruction:
            return prompt
        return f"{instruction}\n\n{prompt}"

    @staticmethod
    def supported_languages() -> List[Dict]:
        return [
            {"code": v["code"], "name": v["name"], "rtl": v["rtl"]}
            for v in _BY_CODE.values()
        ]

    @staticmethod
    def supported_codes() -> List[str]:
        return sorted(_BY_CODE.keys())

    # ── internal ──────────────────────────────────────────────────────────────

    def _make(self, code: str) -> Dict:
        lang = dict(_BY_CODE.get(code, _DEFAULT))   # shallow copy
        lang["prompt_instruction"] = self._instruction(lang["name"])
        return lang

    @staticmethod
    def _instruction(lang_name: str) -> str:
        if lang_name == "English":
            return ""
        return (
            f"IMPORTANT: This lecture is in {lang_name}. "
            f"Write ALL output values (study notes, flashcard questions and answers, "
            f"summaries, bullet points, definitions) in {lang_name}. "
            f"Keep JSON keys in English exactly as specified."
        )