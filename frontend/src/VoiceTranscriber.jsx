
import { useState, useCallback, useRef } from "react";
import { useRecorder }                    from "./hooks/useRecorder";
import { transcribeAudio, enrollSpeaker } from "./services/transcribeApi";
import { Sidebar }                        from "./components/Sidebar/Sidebar";
import { ChatWindow }                     from "./components/Chat/ChatWindow";
import "./styles/global.css";

export default function VoiceTranscriber() {
  const [conversation,    setConversation]    = useState([]);
  const [processing,      setProcessing]      = useState(false);
  const [error,           setError]           = useState(null);
  const [enrolledSpeakers, setEnrolledSpeakers] = useState([]);  // list of enrolled names
  const [enrollStatus,    setEnrollStatus]    = useState(null);  // feedback msg
  const prevLengthRef = useRef(0);

  // ── Transcription ──────────────────────────────────────────────────────────
  const handleBlob = useCallback(async (blob) => {
    setProcessing(true);
    setError(null);
    try {
      const data = await transcribeAudio(blob);
      if (!data.conversation?.length) {
        setError("No speech detected. Speak clearly and try again.");
      } else {
        // In handleBlob, replace the setConversation call:
        setConversation((prev) => {
          prevLengthRef.current = prev.length;
          const merged = [...prev, ...data.conversation];
          // Sort by start time so turns always appear chronologically
          return merged.sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
        });
      }
    } catch (err) {
      setError(err.message || "Could not reach the server. Is it running on port 8000?");
    } finally {
      setProcessing(false);
    }
  }, []);

  const { recording, elapsed, start, stop } = useRecorder(handleBlob);

  const handleStart = async () => {
    setError(null);
    try { await start(); } catch (err) { setError(err.message); }
  };

  const handleRetryVoice = () => {
    setConversation((prev) => prev.slice(0, prevLengthRef.current));
    setError(null);
  };

  const handleNewSession = () => {
    setConversation([]);
    prevLengthRef.current = 0;
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

  // ── Stats from real speaker names ─────────────────────────────────────────
  // Count unique speakers who actually spoke (not Unknown)
  const speakerCounts = conversation.reduce((acc, turn) => {
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
        onRetryVoice={handleRetryVoice}
        onNewSession={handleNewSession}
        hasConversation={conversation.length > 0}
        turnCount={conversation.length}
        speakerCounts={speakerCounts}          // { "Dr. Priya": 4, "Raju": 3 }
        enrolledSpeakers={enrolledSpeakers}    // ["Dr. Priya", "Raju"]
        enrollStatus={enrollStatus}
        onEnroll={handleEnroll}
      />
      <ChatWindow
        conversation={conversation}
        processing={processing}
        error={error}
      />
    </div>
  );
}