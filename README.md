# MedTranscribe

Medical consultation transcription system with speaker diarization, speaker identification, and AI-powered medical terminology correction.

## What it does

Records doctor-patient consultations and produces a structured transcript with:
- Per-speaker turns (identified by name if enrolled, otherwise "Unknown")
- Accurate medical terminology corrected by an LLM
- Timestamps per turn
- TitaNet identity confidence score per speaker

## Architecture

```
Browser audio (WebM/Opus)
        ↓
audio_utils/converter.py       Convert to 16kHz mono WAV
        ↓
diarization/model.py           pyannote/speaker-diarization-3.1
diarization/speaker.py         Merge close same-speaker segments
        ↓
identification/titanet.py      TitaNet Large speaker embeddings
identification/registry.py     Cosine similarity vs enrolled speakers
        ↓
transcription/asr.py           Omi Med STT v1 (medical ASR)
        ↓
correction/medgemma.py         LLM medical terminology correction
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
│   ├── correction/
│   │   ├── __init__.py
│   │   └── medgemma.py                LLM correction (Gemini API)
│   ├── diarization/
│   │   ├── __init__.py
│   │   ├── model.py                   pyannote diarization pipeline
│   │   └── speaker.py                 Segment parsing and merging
│   ├── identification/
│   │   ├── __init__.py
│   │   ├── titanet.py                 TitaNet Large embedding extractor
│   │   └── registry.py                Enrollment store + cosine matching
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── asr.py                     Omi Med STT inference
│   ├── model_cache/
│   │   └── omi-med-stt-v1.nemo        Downloaded manually (see Setup)
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
- CPU-only supported (slower — expect 10-30s per transcription)
- GPU strongly recommended for production use (NVIDIA CUDA 12.4+)
- Minimum 8GB RAM (16GB recommended when running all models together)

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

### 4. Download Omi Med STT model

The model must be downloaded manually due to a NeMo extraction bug with `from_pretrained()`.

```bash
# Login to HuggingFace (free account required)
huggingface-cli login

# Download via Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('omi-health/omi-med-stt-v1', local_dir='./omi_tmp')
"

# Copy the .nemo file to model_cache/
# On Windows:
copy omi_tmp\omimedstt-v1.nemo model_cache\omi-med-stt-v1.nemo

# On Linux/Mac:
cp omi_tmp/omimedstt-v1.nemo model_cache/omi-med-stt-v1.nemo
```

### 5. Accept gated model licenses on HuggingFace

Both of these require accepting a license on the HuggingFace website before they can be downloaded:

- pyannote diarization: https://huggingface.co/pyannote/speaker-diarization-3.1
- pyannote segmentation: https://huggingface.co/pyannote/segmentation-3.0

### 6. Configure environment variables

Create `backend/.env`:

```env
# HuggingFace token (required for pyannote gated models)
HF_TOKEN=hf_your_token_here

# LLM correction — get free key at https://aistudio.google.com
GEMINI_API_KEY=your_gemini_key_here

# Set to false to skip LLM correction (faster, less accurate)
ENABLE_MEDGEMMA_CORRECTION=true

# TitaNet speaker identification threshold (0.0-1.0)
# Lower = more lenient. Raise to 0.70+ after enrolling 3+ samples per person.
TITANET_THRESHOLD=0.55
```

### 7. Install frontend dependencies

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

Server starts at http://localhost:8000

### Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

UI available at http://localhost:5173

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/transcribe | Upload audio → full transcript |
| POST | /api/enroll | Register a speaker by name + voice sample |
| GET | /api/speakers | List all enrolled speakers |
| DELETE | /api/speakers/{name} | Remove an enrolled speaker |
| GET | /api/health | System health check |

## Speaker enrollment

For best identification accuracy:
1. Enter the person's name in the sidebar
2. Click "Record Sample" and speak naturally for 8-10 seconds
3. Click "Stop & Enroll"
4. Repeat 3-5 times with different sentences — each enrollment improves accuracy by averaging embeddings

The TitaNet threshold in `.env` controls how strict matching is. Start at 0.55 and raise to 0.70 once you have multiple samples per person.

## Models used

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| Omi Med STT v1 | Medical speech-to-text | 2.5GB | omi-health/omi-med-stt-v1 |
| pyannote/speaker-diarization-3.1 | Who spoke when | ~300MB | pyannote (gated) |
| TitaNet Large | Speaker identity matching | ~90MB | nvidia/speakerverification_en_titanet_large |
| Gemini 2.5 Flash | Medical terminology correction | API | Google AI Studio (free) |

## Common issues

**`CUDA is not available`** — Running on CPU. All models work on CPU but are slower. No action needed unless you need faster inference.

**`HF_TOKEN` error on startup** — Add your HuggingFace token to `.env`. Get one at https://huggingface.co/settings/tokens.

**`model_config.yaml not found`** — The Omi Med STT model was not extracted correctly. Follow the manual download steps in Setup section 4.

**`torchvision::nms does not exist`** — torch and torchvision version mismatch. Reinstall both together: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`

**Speaker always shows "Unknown"** — TitaNet score is below threshold. Enroll the speaker with more samples (3-5 recordings), or lower `TITANET_THRESHOLD` in `.env`.

**pyannote not separating speakers** — Recording too short (under 5 seconds) or only one real voice present. Record at least 8-10 seconds with clear speaker turns.

## Notes

- All audio processing is local — only the LLM correction step sends text to Google's API
- Enrolled speaker embeddings are stored as `.npy` files in `enrolled_speakers/` — back these up
- The correction step is best-effort — if the API is unavailable, the raw ASR transcript is returned unchanged
- "Loose motions", "gas problem", "acidity" are preserved as-is (valid Indian medical English)