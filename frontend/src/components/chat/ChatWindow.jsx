/**
 * ChatWindow.jsx  v5
 * -------------------
 * Accepts `sessions` (array of {sessionIdx, turns[]}) instead of flat conversation.
 * Sessions render in recording order. Turns within each session are pre-sorted by start time.
 * A subtle divider separates each recording session.
 *
 * Change from v4: unenrolled SPEAKER_xx labels now show as "Unknown"
 * instead of "Speaker 1", "Speaker 2" etc.
 */

import styles from "./ChatMessage.module.css";

const PALETTE = [
  "#60a5fa", "#34d399", "#f472b6", "#a78bfa",
  "#fb923c", "#22d3ee", "#facc15", "#818cf8",
];
const colorCache = {};
let colorIdx = 0;

function isRawLabel(name) {
  return !name || name === "Unknown" || /^[Ss]peaker_?\d+$/i.test(name);
}

function isMissing(name) {
  return !name || name === "Unknown" || /^SPEAKER_\d+$/i.test(name);
}

// Enrolled names (e.g. "joy", "Dr. Priya") pass through unchanged.
// Unenrolled SPEAKER_xx labels → "Unknown".
function displayName(name) {
  if (isMissing(name)) return "Unknown";
  return name;
}

function getSpeakerColor(name) {
  if (isMissing(name)) return "#64748b";
  if (!colorCache[name]) {
    colorCache[name] = PALETTE[colorIdx % PALETTE.length];
    colorIdx++;
  }
  return colorCache[name];
}

function initials(name) {
  if (isMissing(name)) return "?";
  return name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

function formatTime(s) {
  if (s == null) return "";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return m > 0 ? `${m}:${sec}` : `${parseFloat(s).toFixed(1)}s`;
}

function confidenceBadge(turn) {
  const identified = !isRawLabel(turn.speaker);
  if (!identified) {
    return { label: "not enrolled", bg: "rgba(100,116,139,0.12)", color: "#94a3b8" };
  }
  const pct = Math.round((turn.similarity ?? 0) * 100);
  const high = pct >= 75;
  return {
    label: `${pct}% match`,
    bg:    high ? "rgba(52,211,153,0.12)" : "rgba(251,146,60,0.12)",
    color: high ? "#34d399"               : "#fb923c",
  };
}

function TurnCard({ turn }) {
  const name    = displayName(turn.speaker);
  const color   = getSpeakerColor(turn.speaker);
  const badge   = confidenceBadge(turn);
  const unknown = isMissing(turn.speaker);

  return (
    <div className={`${styles.turn} ${unknown ? styles.turnUnknown : ""}`}>
      <div
        className={styles.avatar}
        style={{
          background:  unknown ? "#1e293b" : color + "22",
          color,
          borderColor: color + "44",
        }}
      >
        {initials(turn.speaker)}
      </div>

      <div className={styles.bubble}>
        <div className={styles.meta}>
          <span className={styles.speakerName} style={{ color: unknown ? "#64748b" : color }}>
            {name}
          </span>

          {turn.similarity != null && (
            <span className={styles.badge} style={{ background: badge.bg, color: badge.color }}>
              {badge.label}
            </span>
          )}

          {turn.start != null && (
            <span className={styles.timestamp}>
              {formatTime(turn.start)} – {formatTime(turn.end)}
            </span>
          )}
        </div>

        <p className={`${styles.text} ${unknown ? styles.textUnknown : ""}`}>
          {turn.text}
        </p>

        {turn.diarized_as && !isRawLabel(turn.speaker) && (
          <p className={styles.diarizedAs}>via {turn.diarized_as}</p>
        )}
      </div>
    </div>
  );
}

export function ChatWindow({ sessions = [], processing, error }) {
  const hasContent = sessions.some((s) => s.turns.length > 0);

  return (
    <main className={styles.chatWindow}>

      {!hasContent && !processing && !error && (
        <div className={styles.empty}>
          <div className={styles.emptyIcon}>🎙️</div>
          <p className={styles.emptyTitle}>Ready to transcribe</p>
          <p className={styles.emptyHint}>
            Enroll speakers on the left, then hit <strong>Record</strong>.
          </p>
        </div>
      )}

      <div className={styles.feed}>
        {sessions.map((session, sIdx) => (
          <div key={session.sessionIdx} className={styles.sessionGroup}>

            {sIdx > 0 && (
              <div className={styles.sessionDivider}>
                <span className={styles.sessionDividerLine} />
                <span className={styles.sessionDividerLabel}>Recording {sIdx + 1}</span>
                <span className={styles.sessionDividerLine} />
              </div>
            )}

            {session.turns.map((turn, tIdx) => (
              <TurnCard key={tIdx} turn={turn} />
            ))}
          </div>
        ))}
      </div>

      {processing && (
        <div className={styles.spinnerRow}>
          <div className={styles.spinner} />
          <span className={styles.spinnerLabel}>Transcribing…</span>
        </div>
      )}

      {error && (
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠</span>
          {error}
        </div>
      )}
    </main>
  );
}