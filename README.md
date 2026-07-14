# MedTranscribe

Medical consultation transcription system with Speechmatics Batch transcription & diarization, TitaNet local speaker identification, and a React frontend.

## What it does

Records doctor-patient consultations and produces a structured transcript with:
- Per-speaker turns (identified by name if enrolled via TitaNet, otherwise "Speaker N" / "not enrolled")
- Speechmatics high-accuracy transcription (supporting 65 languages including Tamil, English, Arabic, Spanish, etc.)
- Auto-detection or explicit manual language selection
- Timestamps per turn
- Cosine similarity matching vs enrolled speaker voice signatures

## Architecture

```
Browser audio (WebM/Opus)
        ↓
audio_utils/converter.py       Convert to 16kHz mono WAV
        ↓
speechmatics_pipeline.py       Speechmatics Batch API (enhanced ASR & diarization)
        ↓
identification/titanet.py      TitaNet Large speaker embeddings on speaker turns
identification/registry.py     Cosine similarity vs enrolled speakers
        ↓
Frontend (React + Vite)        Chat-style UI with speaker bubbles
```

## Project structure

```
medtranscribe/
├── backend/
│   ├── server.py                      FastAPI app — main entry point
│   ├── audio_utils/
│   │   ├── __init__.py
│   │   └── converter.py               WebM → 16kHz mono WAV
│   ├── identification/
│   │   ├── __init__.py
│   │   ├── titanet.py                 TitaNet Large embedding extractor
│   │   └── registry.py                Enrollment store + cosine matching
│   ├── speechmatics_pipeline.py       Speechmatics batch request wrapper
│   ├── model_cache/
│   │   └── speakerverification_en_titanet_large.nemo (automatic download)
│   ├── enrolled_speakers/             Auto-created on first enrollment
│   ├── .env                           API keys and config (never commit)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── VoiceTranscriber.jsx
    │   ├── components/
    │   │   ├── Chat/
    │   │   │   ├── ChatWindow.jsx
    │   │   │   └── ChatMessage.module.css
    │   │   ├── Sidebar/
    │   │   │   ├── Sidebar.jsx
    │   │   │   └── Sidebar.module.css
    │   │   └── controls/
    │   │       ├── RecordButton.jsx
    │   │       └── RecordButton.module.css
    │   ├── hooks/
    │   │   └── useRecorder.js
    │   └── services/
    │       └── transcribeApi.js
    ├── package.json
    └── vite.config.js
```

## Requirements

### Hardware
- GPU recommended but CPU-only is fully supported for TitaNet speaker verification embeddings extraction.
- Minimum 8GB RAM.

### Software
- Python 3.10
- Node.js 18+
- conda (for environment management)
- ffmpeg (for audio conversion) — install from https://ffmpeg.org/download.html and add to PATH

## Setup

### 1. Create conda environment

```bash
conda create -n medtranscribe python=3.10 -y
conda activate medtranscribe
```

### 2. Install PyTorch (CUDA 12.4 — skip --index-url for CPU-only)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `backend/.env`:

```env
# Speechmatics Batch API Key
SPEECHMATICS_API_KEY=6jKHkAF86LxtOCYlMGgWIwliHqVHQBtk

# TitaNet speaker identification threshold (0.0-1.0)
# Lower = more lenient. Raise to 0.70+ after enrolling 3+ samples per person.
TITANET_THRESHOLD=0.55
```

### 5. Install frontend dependencies

```bash
cd frontend
npm install
```

## Running

### Backend (Terminal 1)

```bash
conda activate medtranscribe
cd backend
python server.py
```

Server starts at http://localhost:8000 (Detailed terminal logs will print for every audio conversion, job upload, status polling attempt, and segment parser step).

### Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

UI available at http://localhost:5173 (or http://localhost:5174 if the port is busy)

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/transcribe | Upload audio → full Speechmatics transcript + TitaNet speaker match |
| POST | /api/enroll | Register a speaker by name + voice sample |
| GET | /api/speakers | List all enrolled speakers |
| DELETE | /api/speakers/{name} | Remove an enrolled speaker |
| GET | /api/health | System health check |

## Speaker enrollment

For best identification accuracy:
1. Enter the person's name in the sidebar.
2. Click "Record Sample" and speak naturally for 8-10 seconds.
3. Click "Stop & Enroll".
4. Repeat 3-5 times with different sentences — each enrollment improves accuracy by averaging embeddings.

The TitaNet threshold in `.env` controls how strict matching is. Start at 0.55 and raise to 0.70 once you have multiple samples per person.

## Models used

| Model / Service | Purpose | Size | Source |
|-------|---------|------|--------|
| Speechmatics Batch API | Medical ASR & speaker diarization | Cloud API | Speechmatics SaaS |
| TitaNet Large | Speaker identity matching | ~90MB | nvidia/speakerverification_en_titanet_large |

## Common issues

**`Language identification could not identify any language`** — When selecting "Auto Detect" as the language, Speechmatics needs at least 60 seconds of audio to identify the language with high confidence. For shorter clips, please select the specific language directly from the dropdown (e.g. English, Tamil).

**`CUDA is not available`** — Running on CPU. TitaNet works on CPU but is slower. No action needed unless you need faster GPU embedding extraction.

**`Speaker always shows "Speaker N"`** or **`not enrolled`** — The TitaNet similarity score is below the threshold or the speaker profile has not been enrolled. Enroll the speaker with 3-5 recordings, or lower `TITANET_THRESHOLD` in `.env`.

## Notes

- Enrolled speaker embeddings are stored locally as `.npy` files in `backend/enrolled_speakers/`.