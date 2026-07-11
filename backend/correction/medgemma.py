"""
correction/medgemma.py
----------------------
Medical ASR correction using MedGemma 4B deployed on RunPod serverless.

Setup:
    1. Get your RunPod API key from https://www.runpod.io/console/user/settings
    2. Add to your backend/.env file:
         RUNPOD_API_KEY=your_key_here
         RUNPOD_ENDPOINT_ID=7o29s37f0frxed
"""

import os
import re
import json
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RUNPOD_API_KEY     = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "7o29s37f0frxed")

BASE_URL         = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUN_URL          = f"{BASE_URL}/run"
STATUS_URL_TPL   = f"{BASE_URL}/status/{{job_id}}"
CANCEL_URL_TPL   = f"{BASE_URL}/cancel/{{job_id}}"

POLL_INTERVAL    = 2      # seconds between status polls
MAX_WAIT_SECONDS = 200   # 5 minutes before giving up

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(tag: str, msg: str):
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)

def _log_separator():
    print("─" * 70, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# RunPod API
# ─────────────────────────────────────────────────────────────────────────────

def _headers() -> dict:
    if not RUNPOD_API_KEY:
        raise EnvironmentError(
            "RUNPOD_API_KEY is not set. Add it to your .env file."
        )
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    }


def _submit_job(conversation: list) -> str:
    """Submit job to RunPod, return job_id."""
    # Send only speaker/text — timing/confidence metadata is re-merged locally
    # after correction, so MedGemma never needs to see it.
    slim_conversation = [
        {"speaker": t["speaker"], "text": t["text"]} for t in conversation
    ]
    payload = {"input": {"conversation": slim_conversation}}

    _log_separator()
    _log("RUNPOD", f"Submitting job to endpoint: {RUNPOD_ENDPOINT_ID}")
    _log("RUNPOD", f"POST {RUN_URL}")
    _log("RUNPOD", f"Conversation turns: {len(conversation)}")

    resp = requests.post(RUN_URL, json=payload, headers=_headers(), timeout=30)
    _log("RUNPOD", f"HTTP {resp.status_code} received")

    if resp.status_code != 200:
        _log("RUNPOD", f"ERROR — response body: {resp.text}")
        resp.raise_for_status()

    data           = resp.json()
    job_id         = data.get("id")
    initial_status = data.get("status", "unknown")

    _log("RUNPOD", f"Job ID        : {job_id}")
    _log("RUNPOD", f"Initial status: {initial_status}")
    _log_separator()

    return job_id


def _cancel_job(job_id: str):
    """Best-effort cancel so an abandoned job doesn't keep running on RunPod."""
    try:
        resp = requests.post(CANCEL_URL_TPL.format(job_id=job_id), headers=_headers(), timeout=10)
        _log("RUNPOD", f"Cancel requested for job {job_id} — HTTP {resp.status_code}")
    except requests.RequestException as exc:
        _log("RUNPOD", f"Cancel request failed for job {job_id}: {exc}")


def _poll_until_complete(job_id: str) -> dict:
    """Poll status endpoint until terminal state. Returns full response dict."""
    status_url = STATUS_URL_TPL.format(job_id=job_id)
    elapsed    = 0
    poll_count = 0

    _log("RUNPOD", f"Polling: {status_url}")
    _log("RUNPOD", f"Interval: {POLL_INTERVAL}s | Timeout: {MAX_WAIT_SECONDS}s")
    _log_separator()

    STATUS_EMOJI = {
        "IN_QUEUE":    "⏳",
        "IN_PROGRESS": "🔄",
        "COMPLETED":   "✅",
        "FAILED":      "❌",
        "CANCELLED":   "🚫",
        "TIMED_OUT":   "⏰",
    }

    while elapsed < MAX_WAIT_SECONDS:
        try:
            resp = requests.get(status_url, headers=_headers(), timeout=30)
        except requests.RequestException as exc:
            _log("POLL", f"Network error ({exc.__class__.__name__}) — retrying...")
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue

        if resp.status_code != 200:
            _log("POLL", f"HTTP {resp.status_code} — retrying...")
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue

        data       = resp.json()
        status     = data.get("status", "unknown")
        poll_count += 1

        delay_info = f" | queue delay: {data['delayTime']}ms"    if "delayTime"     in data else ""
        exec_info  = f" | exec time: {data['executionTime']}ms"  if "executionTime" in data else ""
        emoji      = STATUS_EMOJI.get(status, "❓")

        _log("POLL", f"#{poll_count:03d} elapsed={elapsed:>4}s  {emoji} {status}{delay_info}{exec_info}")

        if status == "COMPLETED":
            _log_separator()
            _log("RUNPOD", "Job COMPLETED successfully")
            if "executionTime" in data:
                _log("RUNPOD", f"Total execution time: {data['executionTime']}ms")
            return data

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            _log_separator()
            error_msg = data.get("error", data.get("output", "No error details returned"))
            _log("RUNPOD", f"Job ended with status: {status} — {error_msg}")
            raise RuntimeError(f"RunPod job {job_id} ended with status '{status}': {error_msg}")

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    _log("RUNPOD", f"Timed out after {MAX_WAIT_SECONDS}s — cancelling job {job_id}")
    _cancel_job(job_id)
    raise TimeoutError(f"RunPod job {job_id} did not complete within {MAX_WAIT_SECONDS}s")


def _find_conversation_list(obj):
    """
    Recursively search the RunPod output for an already-structured conversation:
    a list of {"speaker", "text"} dicts. The deployed handler returns
    output.output.conversation, so no text parsing is needed at all.
    """
    if isinstance(obj, list) and obj and all(isinstance(t, dict) and "text" in t for t in obj):
        return obj
    if isinstance(obj, dict):
        for key in ("conversation", "corrected_conversation", "turns"):
            found = _find_conversation_list(obj.get(key))
            if found:
                return found
        for value in obj.values():
            found = _find_conversation_list(value)
            if found:
                return found
    return None


def _extract_corrected_text(result: dict) -> str:
    """Pull corrected transcript string from RunPod response."""
    output = result.get("output", {})
    _log("RUNPOD", f"Raw output type: {type(output).__name__} | keys: {list(output.keys()) if isinstance(output, dict) else 'n/a'}")

    if isinstance(output, str):
        return output.strip()

    if isinstance(output, dict):
        for key in ("corrected_text", "text", "result", "response", "output"):
            if key in output:
                _log("RUNPOD", f"Extracted from output['{key}']")
                return str(output[key]).strip()

    _log("RUNPOD", "WARNING: unexpected output shape — stringifying whole output")
    return str(output).strip()


def _parse_corrected_output(raw: str, original: list) -> list:
    """
    Parse MedGemma output back into per-speaker turns.
    Tries, in order:
      1. Markdown-fenced JSON (```json ... ``` or ``` ... ```)
      2. Plain JSON list of {"speaker","text"}
      3. JSON object wrapping the list under a common key
      4. First {...}/[...] JSON block found anywhere in the text (model added prose)
      5. "Speaker: text" lines
      6. Fallback: original turns unchanged
    """
    cleaned = raw.strip()

    # 1. Strip markdown code fences if present
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        _log("PARSE", "Stripped markdown code fence")

    def _try_json(text: str):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        

    # 2. Direct JSON list
    parsed = _try_json(cleaned)
    if isinstance(parsed, list) and all(isinstance(t, dict) and "text" in t for t in parsed):
        _log("PARSE", "Parsed as direct JSON list")
        return parsed

    # 3. JSON object wrapping the list
    if isinstance(parsed, dict):
        for key in ("conversation", "corrected_conversation", "turns", "output", "result"):
            candidate = parsed.get(key)
            if isinstance(candidate, list) and all(isinstance(t, dict) and "text" in t for t in candidate):
                _log("PARSE", f"Parsed as JSON object, unwrapped key '{key}'")
                return candidate

    # 4. Find first JSON array/object embedded anywhere in the raw text
    #    (handles model prefacing output with explanation text)
    for match in re.finditer(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL):
        candidate = _try_json(match.group(1))
        if isinstance(candidate, list) and all(isinstance(t, dict) and "text" in t for t in candidate):
            _log("PARSE", "Parsed as embedded JSON array found in text")
            return candidate
        if isinstance(candidate, dict):
            for key in ("conversation", "corrected_conversation", "turns"):
                inner = candidate.get(key)
                if isinstance(inner, list) and all(isinstance(t, dict) and "text" in t for t in inner):
                    _log("PARSE", f"Parsed as embedded JSON object, unwrapped key '{key}'")
                    return inner

    # 5. "Speaker: text" lines
    lines      = [l.strip() for l in cleaned.splitlines() if l.strip()]
    line_turns = []
    for line in lines:
        match = re.match(r"^(Doctor|Patient|Speaker[_\s]\d+)\s*:\s*(.+)$", line, re.IGNORECASE)
        if match:
            line_turns.append({"speaker": match.group(1), "text": match.group(2).strip()})

    if len(line_turns) == len(original):
        _log("PARSE", "Parsed as Speaker: text lines")
        return line_turns

    # 6. Give up — log the raw output so you can inspect it and extend this function
    _log("PARSE", f"WARNING: could not parse structured output — returning original turns unchanged")
    _log("PARSE", f"RAW OUTPUT (first 500 chars): {cleaned[:500]!r}")
    return [{"speaker": t["speaker"], "text": t["text"]} for t in original]

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def correct_transcript(conversation: list) -> dict:
    _log("MEDGEMMA", f"Starting correction | turns: {len(conversation)}")
    _log_separator()

    job_id  = _submit_job(conversation)
    result  = _poll_until_complete(job_id)

    # Preferred path: the handler already returns structured JSON
    # (output.output.conversation) — use it directly, no text parsing.
    corrected = _find_conversation_list(result.get("output"))
    if corrected:
        _log("MEDGEMMA", f"Found structured conversation in output ({len(corrected)} turns)")
    else:
        raw = _extract_corrected_text(result)
        _log("MEDGEMMA", f"Raw output length: {len(raw)} chars")
        corrected = _parse_corrected_output(raw, conversation)

    # Re-merge timing/confidence metadata that MedGemma doesn't return,
    # so the frontend still gets start/end/similarity/diarized_as.
    if len(corrected) == len(conversation):
        merged = []
        for original, fixed in zip(conversation, corrected):
            merged.append({
                **original,          # keeps start, end, similarity, diarized_as
                "speaker": fixed.get("speaker", original["speaker"]),
                "text":    fixed.get("text", original["text"]),
            })
        corrected = merged
    else:
        _log("MEDGEMMA", f"WARNING: turn count mismatch ({len(corrected)} vs {len(conversation)}) — metadata not merged")

    _log_separator()
    _log("MEDGEMMA", "Correction complete")
    _log_separator()

    return {
        "corrected_conversation": corrected,
        "job_id":                 job_id,
        "execution_time_ms":      result.get("executionTime"),
    }