/**
 * VoiceTranscriber.jsx  v3
 * -------------------------
 * Fix: Each recording blob gets a session index.
 * Turns are sorted within each session, but sessions appear in recording order.
 * So: Session 1 turns (sorted by start) → Session 2 turns (sorted by start) → ...
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { useRecorder }                    from "./hooks/useRecorder";
import { transcribeAudio, enrollSpeaker } from "./services/transcribeApi";
import { Sidebar }                        from "./components/Sidebar/Sidebar";
import { ChatWindow }                     from "./components/chat/ChatWindow";
import "./styles/global.css";

export default function VoiceTranscriber() {
  // conversation is now: Array of session groups
  // Each group: { sessionIdx: number, turns: Turn[] }
  const [sessions,         setSessions]         = useState([]);   // [{sessionIdx, turns}]
  const [processing,       setProcessing]        = useState(false);
  const [error,            setError]             = useState(null);
  const [enrolledSpeakers, setEnrolledSpeakers]  = useState([]);
  const [enrollStatus,     setEnrollStatus]       = useState(null);
  const [numSpeakers,      setNumSpeakers]       = useState("auto");
  
  const sessionCountRef = useRef(0);
  const prevSessionsRef = useRef([]);
  const numSpeakersRef  = useRef("auto");

  // Keep ref in sync
  useEffect(() => {
    numSpeakersRef.current = numSpeakers;
  }, [numSpeakers]);

  // ── Transcription ──────────────────────────────────────────────────────────
  const handleBlob = useCallback(async (blob) => {
    setProcessing(true);
    setError(null);
    try {
      const data = await transcribeAudio(blob, numSpeakersRef.current);
      if (!data.conversation?.length) {
        setError("No speech detected. Speak clearly and try again.");
      } else {
        const sessionIdx = sessionCountRef.current++;

        // Sort turns within this session by start time
        const sortedTurns = [...data.conversation].sort(
          (a, b) => (a.start ?? 0) - (b.start ?? 0)
        );

        setSessions((prev) => {
          prevSessionsRef.current = prev;
          return [...prev, { sessionIdx, turns: sortedTurns }];
        });
      }
    } catch (err) {
      setError(err.message || "Could not reach the server. Is it running on port 8000?");
    } finally {
      setProcessing(false);
    }
  }, []);

  const { recording, elapsed, start, stop, cancel } = useRecorder(handleBlob);

  const handleStart = async () => {
    setError(null);
    try { await start(); } catch (err) { setError(err.message); }
  };

  const handleRetryVoice = () => {
    // Remove last session
    setSessions((prev) => prev.slice(0, -1));
    sessionCountRef.current = Math.max(0, sessionCountRef.current - 1);
    setError(null);
  };

  const handleNewSession = () => {
    setSessions([]);
    sessionCountRef.current = 0;
    setError(null);
  };

  // ── Enrollment ─────────────────────────────────────────────────────────────
  const handleEnroll = useCallback(async (blob, name) => {
    setEnrollStatus({ state: "loading", message: `Enrolling "${name}"...` });
    try {
      await enrollSpeaker(blob, name);
      setEnrolledSpeakers((prev) =>
        prev.includes(name) ? prev : [...prev, name]
      );
      setEnrollStatus({ state: "success", message: `"${name}" enrolled successfully!` });
      setTimeout(() => setEnrollStatus(null), 3000);
    } catch (err) {
      setEnrollStatus({ state: "error", message: err.message || "Enrollment failed." });
    }
  }, []);

  // ── Flatten for stats ──────────────────────────────────────────────────────
  const allTurns = sessions.flatMap((s) => s.turns);
  const turnCount = allTurns.length;
  const speakerCounts = allTurns.reduce((acc, turn) => {
    const name = turn.speaker || "Unknown";
    acc[name] = (acc[name] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="app-shell">
      <Sidebar
        recording={recording}
        elapsed={elapsed}
        processing={processing}
        onStart={handleStart}
        onStop={stop}
        onCancel={cancel}
        onRetryVoice={handleRetryVoice}
        onNewSession={handleNewSession}
        hasConversation={sessions.length > 0}
        turnCount={turnCount}
        speakerCounts={speakerCounts}
        enrolledSpeakers={enrolledSpeakers}
        enrollStatus={enrollStatus}
        onEnroll={handleEnroll}
        numSpeakers={numSpeakers}
        onNumSpeakersChange={setNumSpeakers}
      />
      <ChatWindow
        sessions={sessions}           // pass sessions, not flat conversation
        processing={processing}
        error={error}
      />
    </div>
  );
}