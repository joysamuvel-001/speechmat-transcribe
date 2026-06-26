

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Transcribe a recorded audio blob.
 * Returns { text, conversation: [{speaker, diarized_as, similarity, text, start, end}] }
 */
export async function transcribeAudio(blob) {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");

  const res = await fetch(`${BASE}/api/transcribe`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  return res.json();
}

/**
 * Enroll a speaker — record a sample and give them a name.
 * The backend extracts a TitaNet embedding and saves it.
 * Call this multiple times with the same name to improve accuracy.
 */
export async function enrollSpeaker(blob, name) {
  const form = new FormData();
  form.append("audio", blob, "enrollment.webm");
  form.append("name", name);

  const res = await fetch(`${BASE}/api/enroll`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Enrollment failed ${res.status}`);
  }
  return res.json();  // { name, status: "enrolled" }
}

/**
 * Fetch list of all enrolled speakers from the backend.
 */
export async function fetchEnrolledSpeakers() {
  const res = await fetch(`${BASE}/api/speakers`);
  if (!res.ok) throw new Error("Could not fetch speakers");
  const data = await res.json();
  return data.speakers; // string[]
}

/**
 * Delete an enrolled speaker.
 */
export async function deleteSpeaker(name) {
  const res = await fetch(`${BASE}/api/speakers/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Could not delete speaker");
  return res.json();
}