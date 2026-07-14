/**
 * Sidebar.jsx  v3
 * ----------------
 * Fixes:
 *   - NaN:NaN timer → clean elapsed display with pulsing indicator
 *   - Retry / New Session buttons properly styled and placed
 *   - Overall sidebar polish
 */

import { useState, useRef } from "react";
import styles               from "./Sidebar.module.css";

const LANGUAGES = [
  { code: "auto", name: "Auto Detect" },
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "zh", name: "Chinese (Mandarin)" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "hi", name: "Hindi" },
  { code: "ar", name: "Arabic" },
  { code: "hy", name: "Armenian" },
  { code: "eu", name: "Basque" },
  { code: "be", name: "Belarusian" },
  { code: "bn", name: "Bengali" },
  { code: "bg", name: "Bulgarian" },
  { code: "ca", name: "Catalan" },
  { code: "hr", name: "Croatian" },
  { code: "cs", name: "Czech" },
  { code: "da", name: "Danish" },
  { code: "nl", name: "Dutch" },
  { code: "eo", name: "Esperanto" },
  { code: "et", name: "Estonian" },
  { code: "fi", name: "Finnish" },
  { code: "gl", name: "Galician" },
  { code: "ka", name: "Georgian" },
  { code: "el", name: "Greek" },
  { code: "gu", name: "Gujarati" },
  { code: "he", name: "Hebrew" },
  { code: "hu", name: "Hungarian" },
  { code: "is", name: "Icelandic" },
  { code: "id", name: "Indonesian" },
  { code: "ga", name: "Irish" },
  { code: "kn", name: "Kannada" },
  { code: "kk", name: "Kazakh" },
  { code: "lv", name: "Latvian" },
  { code: "lt", name: "Lithuanian" },
  { code: "mk", name: "Macedonian" },
  { code: "ms", name: "Malay" },
  { code: "ml", name: "Malayalam" },
  { code: "mt", name: "Maltese" },
  { code: "mr", name: "Marathi" },
  { code: "mn", name: "Mongolian" },
  { code: "ne", name: "Nepali" },
  { code: "no", name: "Norwegian" },
  { code: "fa", name: "Persian" },
  { code: "pl", name: "Polish" },
  { code: "pa", name: "Punjabi" },
  { code: "ro", name: "Romanian" },
  { code: "ru", name: "Russian" },
  { code: "sr", name: "Serbian" },
  { code: "sk", name: "Slovak" },
  { code: "sl", name: "Slovenian" },
  { code: "so", name: "Somali" },
  { code: "sw", name: "Swahili" },
  { code: "sv", name: "Swedish" },
  { code: "tl", name: "Tagalog" },
  { code: "ta", name: "Tamil" },
  { code: "te", name: "Telugu" },
  { code: "th", name: "Thai" },
  { code: "tr", name: "Turkish" },
  { code: "uk", name: "Ukrainian" },
  { code: "ur", name: "Urdu" },
  { code: "vi", name: "Vietnamese" },
  { code: "cy", name: "Welsh" },
];

export function Sidebar({
  recording, elapsed, processing,
  onStart, onStop, onCancel, onRetryVoice, onNewSession,
  hasConversation, turnCount,
  speakerCounts,
  enrolledSpeakers,
  enrollStatus,
  onEnroll,
  numSpeakers,
  onNumSpeakersChange,
  language,
  onLanguageChange,
}) {
  const [speakerName,     setSpeakerName]     = useState("");
  const [enrollRecording, setEnrollRecording] = useState(false);
  const enrollChunksRef   = useRef([]);
  const enrollStreamRef   = useRef(null);
  const enrollRecorderRef = useRef(null);

  const startEnrollRecording = async () => {
    if (!speakerName.trim()) { alert("Enter the person's name before recording."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      enrollStreamRef.current  = stream;
      enrollChunksRef.current  = [];
      const recorder = new MediaRecorder(stream);
      enrollRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => { if (e.data.size > 0) enrollChunksRef.current.push(e.data); };
      recorder.start();
      setEnrollRecording(true);
    } catch { alert("Microphone access denied."); }
  };

  const stopEnrollRecording = () => {
    const recorder = enrollRecorderRef.current;
    if (!recorder) return;
    recorder.onstop = () => {
      const blob = new Blob(enrollChunksRef.current, { type: "audio/webm" });
      onEnroll(blob, speakerName.trim());
      enrollStreamRef.current?.getTracks().forEach((t) => t.stop());
      setEnrollRecording(false);
    };
    recorder.stop();
  };

  // Fix NaN:NaN — elapsed may be float or undefined
  const fmtElapsed = (s) => {
    const total = Math.floor(Number(s) || 0);
    const m = Math.floor(total / 60);
    const sec = String(total % 60).padStart(2, "0");
    return m > 0 ? `${m}:${sec}` : `0:${sec}`;
  };

  return (
    <aside className={styles.sidebar}>

      {/* ── Brand ── */}
      <div className={styles.brand}>
        <span className={styles.brandIcon}>🎙</span>
        <span className={styles.brandName}>MedTranscribe</span>
      </div>

      {/* ── Enrollment ── */}
      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>Speaker Enrollment</h3>
        <p className={styles.sectionHint}>
          Record a 5–10 sec sample per person so TitaNet can identify them by voice.
        </p>

        <input
          className={styles.nameInput}
          type="text"
          placeholder="Person's name (e.g. Dr. Priya)"
          value={speakerName}
          onChange={(e) => setSpeakerName(e.target.value)}
          disabled={enrollRecording}
        />

        {!enrollRecording ? (
          <button
            className={styles.enrollRecord}
            onClick={startEnrollRecording}
            disabled={!speakerName.trim() || enrollStatus?.state === "loading"}
          >
            <span className={styles.recDot} /> Record Sample
          </button>
        ) : (
          <button className={styles.enrollStop} onClick={stopEnrollRecording}>
            ⏹ Stop &amp; Enroll
          </button>
        )}

        {enrollStatus && (
          <p className={`${styles.enrollMsg} ${styles[enrollStatus.state]}`}>
            {enrollStatus.state === "loading" && "⏳ "}
            {enrollStatus.state === "success" && "✅ "}
            {enrollStatus.state === "error"   && "❌ "}
            {enrollStatus.message}
          </p>
        )}

        {enrolledSpeakers.length > 0 && (
          <div className={styles.enrolledList}>
            <span className={styles.enrolledLabel}>Enrolled:</span>
            {enrolledSpeakers.map((name) => (
              <span key={name} className={styles.enrolledChip}>{name}</span>
            ))}
          </div>
        )}
      </section>

      {/* ── Recording ── */}
      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>Recording</h3>

        <div className={styles.speakersSelectRow}>
          <label htmlFor="num-speakers" className={styles.selectLabel}>Number of Speakers</label>
          <select
            id="num-speakers"
            className={styles.selectInput}
            value={numSpeakers}
            onChange={(e) => onNumSpeakersChange(e.target.value)}
            disabled={recording || processing}
          >
            <option value="auto">Auto Detect</option>
            <option value="1">1 Speaker</option>
            <option value="2">2 Speakers</option>
            <option value="3">3 Speakers</option>
            <option value="4">4 Speakers</option>
            <option value="5">5 Speakers</option>
          </select>
        </div>

        <div className={styles.speakersSelectRow}>
          <label htmlFor="language" className={styles.selectLabel}>Language</label>
          <select
            id="language"
            className={styles.selectInput}
            value={language}
            onChange={(e) => onLanguageChange(e.target.value)}
            disabled={recording || processing}
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.micWrap}>
          {/* Timer shown above mic while recording */}
          {recording && (
            <div className={styles.timerRow}>
              <span className={styles.timerDot} />
              <span className={styles.timerText}>{fmtElapsed(elapsed)}</span>
            </div>
          )}

          <button
            className={`${styles.micBtn} ${recording ? styles.micBtnActive : ""}`}
            onClick={recording ? onStop : onStart}
            disabled={processing}
            aria-label={recording ? "Stop recording" : "Start recording"}
          >
            {recording
              ? <span className={styles.stopIcon} />
              : <svg viewBox="0 0 24 24" fill="currentColor" width="28" height="28"><path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm-1 16.93V20H9v2h6v-2h-2v-2.07A7.001 7.001 0 0 0 19 12h-2a5 5 0 0 1-10 0H5a7.001 7.001 0 0 0 6 6.93z"/></svg>
            }
          </button>

          <p className={styles.micLabel}>
            {processing ? "Processing…" : recording ? "Tap to stop" : "Tap to record"}
          </p>

          {recording && (
            <button
              className={styles.cancelRecBtn}
              onClick={onCancel}
              title="Cancel recording and discard audio"
            >
              Cancel
            </button>
          )}
        </div>

        {processing && (
          <div className={styles.processingRow}>
            <span className={styles.processingSpinner} />
            <span className={styles.processingText}>Identifying speakers…</span>
          </div>
        )}
      </section>

      {/* ── Session Actions ── */}
      {hasConversation && (
        <section className={styles.section}>
          <div className={styles.actionBtns}>
            <button className={styles.retryBtn} onClick={onRetryVoice} title="Remove last recording">
              <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fillRule="evenodd" d="M4 2a1 1 0 0 1 1 1v2.101a7.002 7.002 0 0 1 11.601 2.566 1 1 0 1 1-1.885.666A5.002 5.002 0 0 0 5.999 7H9a1 1 0 0 1 0 2H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm.008 9.057a1 1 0 0 1 1.276.61A5.002 5.002 0 0 0 14.001 13H11a1 1 0 1 1 0-2h5a1 1 0 0 1 1 1v5a1 1 0 1 1-2 0v-2.101a7.002 7.002 0 0 1-11.601-2.566 1 1 0 0 1 .61-1.276z" clipRule="evenodd"/></svg>
              Retry Last
            </button>
            <button className={styles.newBtn} onClick={onNewSession} title="Clear and start fresh">
              <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fillRule="evenodd" d="M10 3a1 1 0 0 1 1 1v5h5a1 1 0 1 1 0 2h-5v5a1 1 0 1 1-2 0v-5H4a1 1 0 1 1 0-2h5V4a1 1 0 0 1 1-1z" clipRule="evenodd"/></svg>
              New Session
            </button>
          </div>
        </section>
      )}

      {/* ── Stats ── */}
      {turnCount > 0 && (
        <section className={`${styles.section} ${styles.statsSection}`}>
          <h3 className={styles.sectionTitle}>This Session</h3>
          <p className={styles.statTotal}>
            <strong>{turnCount}</strong> turns total
          </p>
          {Object.entries(speakerCounts).map(([name, count]) => (
            <div key={name} className={styles.statRow}>
              <span className={styles.statName}>{name}</span>
              <span className={styles.statCount}>{count} turns</span>
            </div>
          ))}
        </section>
      )}
    </aside>
  );
}