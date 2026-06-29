"""
correction/medgemma.py
-----------------------
Single-pass medical ASR correction using MedGemma 4B.

Uses the BASE medgemma-4b-it model, not medgemma-1.5-4b-it — the 1.5
version has a "thinking trace" behavior that, combined with the
"think silently if needed" system instruction (which Google's own docs
say ACTIVATES extended reasoning, not suppresses it), produces long
internal reasoning text instead of a clean corrected line. The base
4b-it model does not have this behavior, and this code does not use
that instruction at all.

Requires:
    pip install transformers>=4.50.0 accelerate
    License already accepted for google/medgemma-4b-it (confirmed earlier).
"""

import os
import re
import difflib
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-4b-it"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32

ENABLE_CORRECTION = os.environ.get("ENABLE_MEDGEMMA_CORRECTION", "true").lower() == "true"

_model = None
_processor = None

if ENABLE_CORRECTION:
    print(f"[medgemma] Loading {MODEL_ID}...")
    try:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        _model = _model.to("cpu")
        _model.eval()
        print(f"[medgemma] Ready on {DEVICE}.")
    except Exception as e:
        print(f"[medgemma] FAILED — correction disabled: {e}")
        _model = None
        _processor = None
else:
    print("[medgemma] Correction disabled via ENABLE_MEDGEMMA_CORRECTION=false")


SYSTEM_PROMPT = (
    "You are a clinical transcription specialist. You correct ASR errors "
    "in doctor-patient conversations. You never invent clinical content. "
    "You respond directly with no reasoning, explanation, or commentary — "
    "only the requested output."
)

PROMPT_TEMPLATE = """\
Below is an ASR transcript of a doctor-patient consultation, numbered by turn.
The speaker may be an Indian-English speaker, and the speech recognizer may have
mis-transcribed or phonetically mangled medical terms, drug names, or numbers.

Examples of the kind of error to fix (style only, not domain):
"with form in" -> "metformin" | "solution mall" -> "salbutamol"
"seismic dysfunction" -> "systolic dysfunction" | "five hundred mg" -> "500 mg"

RULES:
- Fix a word only if context makes the intended term clearly inferable.
- If unsure, leave the original word UNCHANGED — never guess.
- Do not add, remove, or rephrase anything beyond fixing garbled terms.
- Output ONLY the corrected lines, same numbering, nothing else — no
  explanation, no reasoning, no preamble.

1. <text>
2. <text>

TRANSCRIPT:
{numbered_turns}
"""


def correct_medical_terms(conversation: list) -> list:
    if not ENABLE_CORRECTION or _model is None or not conversation:
        return conversation

    numbered = "\n".join(f"{i+1}. {t.get('text','')}" for i, t in enumerate(conversation))
    prompt = PROMPT_TEMPLATE.format(numbered_turns=numbered)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]

    try:
        output = _generate(messages, max_new_tokens=1024)
        corrected_lines = _parse_numbered(output, expected_count=len(conversation))
        if corrected_lines is None:
            print(f"[medgemma] Parse/count mismatch — keeping original transcript. Raw output: {output[:200]!r}")
            return conversation

        result = []
        for turn, corrected in zip(conversation, corrected_lines):
            original = turn.get("text", "")
            if corrected and _is_safe_correction(original, corrected):
                new_turn = dict(turn)
                new_turn["raw_text"] = original
                new_turn["text"] = corrected
                result.append(new_turn)
            else:
                result.append(turn)
        return result
    except Exception as e:
        print(f"[medgemma] Correction failed, returning original: {e}")
        return conversation


def _generate(messages, max_new_tokens):
    inputs = _processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(_model.device, dtype=DTYPE)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return _processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def _parse_numbered(text, expected_count):
    lines = re.findall(r"^\s*\d+\.\s*(.*)$", text, flags=re.MULTILINE)
    lines = [l.strip() for l in lines if l.strip()]
    return lines if len(lines) == expected_count else None


def _is_safe_correction(original, corrected):
    BLOCK = ["as an ai", "i cannot", "medical advice", "disclaimer", "here is the corrected"]
    lower = corrected.lower()
    if any(p in lower for p in BLOCK) or not corrected.strip():
        return False
    orig_words, corr_words = original.split(), corrected.split()
    if len(corr_words) > len(orig_words) * 1.5:
        return False
    return difflib.SequenceMatcher(None, orig_words, corr_words).ratio() >= 0.60