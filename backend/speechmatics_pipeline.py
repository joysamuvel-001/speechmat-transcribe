import os
import sys
import json
import asyncio
import httpx
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

SPEECHMATICS_API_KEY = os.environ.get("SPEECHMATICS_API_KEY") or "6jKHkAF86LxtOCYlMGgWIwliHqVHQBtk"

def map_speaker(spk_raw: str) -> str:
    if not spk_raw:
        return "Unknown"
    # Speechmatics typically outputs speaker_0, speaker_1, etc.
    # We map them to Speaker 1, Speaker 2, etc.
    parts = spk_raw.split("_")
    if len(parts) == 2 and parts[0] == "speaker" and parts[1].isdigit():
        return f"Speaker {int(parts[1]) + 1}"
    
    # Also support alternative format S1, S2, etc.
    if len(spk_raw) >= 2 and spk_raw[0] == "S" and spk_raw[1:].isdigit():
        return f"Speaker {spk_raw[1:]}"
        
    return spk_raw

def _safe_print_text(text: str) -> str:
    # Encodes foreign/non-ASCII characters to ASCII using backslashreplace to prevent CP1252/terminal crashes on print
    return text.encode('ascii', errors='backslashreplace').decode('ascii')

def parse_speechmatics_results(results: list) -> list:
    print(f"[Speechmatics Parser] Beginning parsing of {len(results)} raw transcript elements...")
    turns = []
    current_turn = None
    
    for item in results:
        item_type = item.get("type")
        if item_type not in ("word", "punctuation"):
            continue
            
        alt = item.get("alternatives", [{}])[0]
        content = alt.get("content", "")
        if not content:
            continue
            
        # Determine speaker
        speaker_raw = alt.get("speaker")
        if item_type == "punctuation":
            if current_turn:
                speaker = current_turn["speaker"]
            else:
                speaker = map_speaker(speaker_raw) if speaker_raw else "Unknown"
        else:
            speaker = map_speaker(speaker_raw)
            
        start_time = item.get("start_time")
        end_time = item.get("end_time")
        
        if current_turn is not None and current_turn["speaker"] == speaker:
            # Append content to current turn text
            if item_type == "punctuation" and content not in ("(", "[", "{"):
                # Append punctuation without space
                current_turn["text"] += content
            else:
                # Add word or open bracket with space
                if current_turn["text"] and not current_turn["text"].endswith(("( ", "[ ", "{ ")):
                    current_turn["text"] += " " + content
                else:
                    current_turn["text"] += content
                    
            if end_time is not None:
                current_turn["end"] = end_time
        else:
            # Start a new turn
            if current_turn is not None:
                safe_text = _safe_print_text(current_turn['text'][:60])
                print(f"[Speechmatics Parser] Finished turn for {current_turn['speaker']}: \"{safe_text}...\" ({current_turn['start']}s - {current_turn['end']}s)")
            
            print(f"[Speechmatics Parser] New turn detected for speaker: {speaker} starting at {start_time}s")
            current_turn = {
                "speaker": speaker,
                "text": content,
                "start": start_time if start_time is not None else 0.0,
                "end": end_time if end_time is not None else 0.0
            }
            turns.append(current_turn)
            
    if current_turn is not None:
        safe_text = _safe_print_text(current_turn['text'][:60])
        print(f"[Speechmatics Parser] Finished turn for {current_turn['speaker']}: \"{safe_text}...\" ({current_turn['start']}s - {current_turn['end']}s)")
        
    # Post-process turns
    for turn in turns:
        turn["start"] = round(turn["start"], 2)
        turn["end"] = round(turn["end"], 2)
        turn["text"] = turn["text"].strip()
        
    print(f"[Speechmatics Parser] Successfully assembled {len(turns)} total turns.")
    return turns

async def run_speechmatics_pipeline(wav_path: str, num_speakers: int = None, language: str = "en") -> dict:
    url = "https://asr.api.speechmatics.com/v2/jobs"
    headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}
    
    config = {
        "type": "transcription",
        "transcription_config": {
            "language": language,
            "diarization": "speaker",
            "operating_point": "enhanced"
        }
    }
    
    print(f"\n[Speechmatics Pipeline] --- STARTING BATCH TRANSCRIPTION ---")
    print(f"[Speechmatics Pipeline] Audio file: {wav_path}")
    print(f"[Speechmatics Pipeline] Language: '{language}'")
    print(f"[Speechmatics Pipeline] Diarization: 'speaker'")
    
    # Read the audio file
    print(f"[Speechmatics Pipeline] Reading audio file binary data...")
    with open(wav_path, "rb") as f:
        audio_data = f.read()
        
    files = {
        "config": (None, json.dumps(config), "application/json"),
        "data_file": ("audio.wav", audio_data, "audio/wav")
    }
    
    print(f"[Speechmatics Pipeline] Submitting job request to Speechmatics endpoint: {url}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, files=files)
        if response.status_code != 201:
            print(f"[Speechmatics Pipeline] ERROR: Submission failed with status {response.status_code}")
            print(f"[Speechmatics Pipeline] Details: {response.text}")
            raise Exception(f"Failed to submit Speechmatics job: {response.status_code} - {response.text}")
            
        job_data = response.json()
        job_id = job_data["id"]
        print(f"[Speechmatics Pipeline] Job submitted successfully. Assigned ID: {job_id}")
        
        # Poll status
        status_url = f"https://asr.api.speechmatics.com/v2/jobs/{job_id}"
        poll_interval = 2.0
        max_attempts = 150  # 300 seconds total timeout
        
        print(f"[Speechmatics Pipeline] Commencing job polling (interval={poll_interval}s, max_attempts={max_attempts})...")
        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval)
            status_resp = await client.get(status_url, headers=headers)
            if status_resp.status_code != 200:
                print(f"[Speechmatics Pipeline] Warning: status check failed with status code {status_resp.status_code}")
                continue
                
            status_data = status_resp.json()
            job_status = status_data.get("job", {}).get("status")
            print(f"[Speechmatics Pipeline] Poll check #{attempt+1}: job status is '{job_status}'")
            
            if job_status == "done":
                print(f"[Speechmatics Pipeline] Job finished! Requesting v2 JSON transcript...")
                # Get the JSON v2 transcript
                transcript_url = f"{status_url}/transcript?format=json-v2"
                transcript_resp = await client.get(transcript_url, headers=headers)
                if transcript_resp.status_code != 200:
                    print(f"[Speechmatics Pipeline] ERROR: Failed to fetch transcript (status {transcript_resp.status_code})")
                    raise Exception(f"Failed to retrieve transcript: {transcript_resp.status_code} - {transcript_resp.text}")
                print(f"[Speechmatics Pipeline] Transcript retrieved successfully. Parsing results...")
                return transcript_resp.json()
            elif job_status in ("rejected", "expired"):
                print(f"[Speechmatics Pipeline] ERROR: Job failed with status '{job_status}'")
                errors = status_data.get("job", {}).get("errors", [])
                if errors:
                    err_msg = errors[0].get("message", "")
                    if "Language identification could not identify" in err_msg:
                        raise Exception("Auto Detect failed: The audio was too short to identify the language. Please select your specific language from the dropdown (e.g. English, Tamil) instead.")
                    raise Exception(f"Speechmatics job rejected: {err_msg}")
                raise Exception(f"Speechmatics job {job_status}")
                
        print(f"[Speechmatics Pipeline] ERROR: Job timed out after {max_attempts * poll_interval} seconds")
        raise Exception("Speechmatics job execution timed out")
