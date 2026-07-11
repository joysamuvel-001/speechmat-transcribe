import os
import av
import numpy as np
from scipy.io import wavfile

TARGET_RMS_DBFS  = -20.0   # target loudness every clip is normalized to
SILENCE_RMS_DBFS = -40.0   # frames quieter than this are treated as silence
FRAME_MS         = 30      # analysis frame size for silence trimming


def convert_to_wav(audio_bytes: bytes, output_path: str) -> None:
    """
    Write audio_bytes (WebM/Opus from browser) to output_path as
    a 16kHz mono signed-16-bit WAV file, loudness-normalized with
    leading/trailing silence trimmed.

    This function is used for BOTH speaker enrollment and every
    transcription upload. Normalizing here — once, in the one place
    all audio passes through — means enrollment and live-recording
    audio always arrive at TitaNet at comparable loudness, regardless
    of mic distance/gain differences between sessions. That removes a
    real source of embedding drift that is otherwise indistinguishable
    from a genuine voice match/mismatch.

    Raises RuntimeError if the input has no audio stream or no samples.
    """
    tmp_input = output_path + "_input"
    with open(tmp_input, "wb") as f:
        f.write(audio_bytes)

    try:
        samples = []

        with av.open(tmp_input) as container:
            audio_stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if audio_stream is None:
                raise RuntimeError("No audio stream found in the uploaded file.")

            print(
                f"[converter] codec={audio_stream.codec_context.name}  "
                f"rate={audio_stream.sample_rate}Hz  "
                f"channels={audio_stream.channels}"
            )

            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

            for frame in container.decode(audio_stream):
                for resampled in resampler.resample(frame):
                    samples.append(resampled.to_ndarray()[0])

            # Flush the resampler
            for resampled in resampler.resample(None):
                samples.append(resampled.to_ndarray()[0])

        if not samples:
            raise RuntimeError("No audio samples decoded — file may be empty or corrupt.")

        audio_np = np.concatenate(samples).astype(np.int16)

        audio_np = _normalize_level(audio_np)
        audio_np = _trim_silence(audio_np)

        if audio_np.size == 0:
            raise RuntimeError("Audio was entirely silence after trimming.")

        wavfile.write(output_path, 16000, audio_np)

        duration = len(audio_np) / 16000
        print(f"[converter] wrote {len(audio_np)} samples ({duration:.2f}s) → {output_path}")

    finally:
        if os.path.exists(tmp_input):
            os.remove(tmp_input)


def _normalize_level(audio_np: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    """
    RMS-normalize to a fixed loudness so embedding quality doesn't depend
    on mic distance/gain. Applies the same formula to any input — nothing
    speaker- or session-specific.
    """
    audio_f = audio_np.astype(np.float64)
    audio_f -= audio_f.mean()  # remove DC offset

    rms = np.sqrt(np.mean(audio_f ** 2))
    if rms < 1e-6:
        return audio_np  # pure silence — nothing to normalize

    current_dbfs = 20 * np.log10(rms / 32768.0)
    gain_db = target_dbfs - current_dbfs
    gain = 10 ** (gain_db / 20)

    normalized = np.clip(audio_f * gain, -32768, 32767)
    return normalized.astype(np.int16)


def _trim_silence(
    audio_np: np.ndarray,
    frame_ms: int = FRAME_MS,
    silence_dbfs: float = SILENCE_RMS_DBFS,
    sr: int = 16000,
) -> np.ndarray:
    """
    Drop leading/trailing silence so a segment's embedding isn't diluted
    by dead air sitting at diarization segment boundaries. The loudness
    floor is absolute, not tuned per speaker or per recording.
    """
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(audio_np) < frame_len:
        return audio_np

    audio_f = audio_np.astype(np.float64)
    n_frames = len(audio_f) // frame_len
    frames = audio_f[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_dbfs = 20 * np.log10(np.maximum(frame_rms, 1) / 32768.0)

    loud = np.where(frame_dbfs > silence_dbfs)[0]
    if loud.size == 0:
        return audio_np  # everything reads as "quiet" — don't nuke potential real speech

    start = loud[0] * frame_len
    end = min((loud[-1] + 1) * frame_len, len(audio_np))
    return audio_np[start:end]