

import styles from "./ChatMessage.module.css";

// ── Helpers ──────────────────────────────────────────────────────────────────

const PALETTE = [
  "#60a5fa", "#34d399", "#f472b6", "#a78bfa",
  "#fb923c", "#22d3ee", "#facc15", "#818cf8",
];
const colorCache = {};
let colorIdx = 0;

function isRawLabel(name) {
  return !name || name === "Unknown" || /^[Ss]peaker_?\d+$/i.test(name);
}

function displayName(name) {
  return isRawLabel(name) ? "Unknown" : name;
}

function getSpeakerColor(name) {
  if (isRawLabel(name)) return "#64748b";
  if (!colorCache[name]) {
    colorCache[name] = PALETTE[colorIdx % PALETTE.length];
    colorIdx++;
  }
  return colorCache[name];
}

function initials(name) {
  if (isRawLabel(name)) return "?";
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

// ── Component ─────────────────────────────────────────────────────────────────

export function ChatWindow({ conversation, processing, error }) {
  // Sort by start time — fixes out-of-order turns
  const sorted = [...conversation].sort((a, b) => (a.start ?? 0) - (b.start ?? 0));

  return (
    <main className={styles.chatWindow}>

      {/* ── Empty state ── */}
      {!sorted.length && !processing && !error && (
        <div className={styles.empty}>
          <div className={styles.emptyIcon}>🎙️</div>
          <p className={styles.emptyTitle}>Ready to transcribe</p>
          <p className={styles.emptyHint}>
            Enroll speakers on the left, then hit <strong>Record</strong>.
          </p>
        </div>
      )}

      {/* ── Turns ── */}
      <div className={styles.feed}>
        {sorted.map((turn, i) => {
          const name  = displayName(turn.speaker);
          const color = getSpeakerColor(turn.speaker);
          const badge = confidenceBadge(turn);
          const unknown = isRawLabel(turn.speaker);

          return (
            <div key={i} className={`${styles.turn} ${unknown ? styles.turnUnknown : ""}`}>

              {/* Avatar */}
              <div
                className={styles.avatar}
                style={{ background: unknown ? "#1e293b" : color + "22", color, borderColor: color + "44" }}
              >
                {initials(turn.speaker)}
              </div>

              {/* Bubble */}
              <div className={styles.bubble}>
                <div className={styles.meta}>
                  {/* Name */}
                  <span
                    className={styles.speakerName}
                    style={{ color: unknown ? "#64748b" : color }}
                  >
                    {name}
                  </span>

                  {/* Confidence badge */}
                  {turn.similarity != null && (
                    <span
                      className={styles.badge}
                      style={{ background: badge.bg, color: badge.color }}
                    >
                      {badge.label}
                    </span>
                  )}

                  {/* Timestamp — right-aligned */}
                  {turn.start != null && (
                    <span className={styles.timestamp}>
                      {formatTime(turn.start)} – {formatTime(turn.end)}
                    </span>
                  )}
                </div>

                {/* Transcript */}
                <p className={`${styles.text} ${unknown ? styles.textUnknown : ""}`}>
                  {turn.text}
                </p>

                {/* Diarized-as — only show if different from displayed name */}
                {turn.diarized_as && !isRawLabel(turn.speaker) && (
                  <p className={styles.diarizedAs}>
                    via {turn.diarized_as}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Processing indicator ── */}
      {processing && (
        <div className={styles.spinnerRow}>
          <div className={styles.spinner} />
          <span className={styles.spinnerLabel}>Transcribing…</span>
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠</span>
          {error}
        </div>
      )}
    </main>
  );
}