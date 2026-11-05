import { useEffect, useState } from "react";
import type { Candidates, Telemetry } from "../lib/contracts";
import { AUTOPILOT_COLORS, AUTOPILOT_HELP, AUTOPILOT_LABEL, fmtMs } from "../lib/theme";
import SubjectOverlay from "./SubjectOverlay";

/** Frame age shown in amber from here: the governor goes neutral at 0.7 s (PLAN.md 4.5). */
const FRAME_AGE_WARN_MS = 700;

/**
 * Drone camera (upstream MJPEG feed) with the SAM 3.1 subject overlay (new), one status line and,
 * while the operator talks, what the drone is hearing.
 */
export default function VideoPanel({
  mjpegUrl,
  telemetry,
  stale,
  candidates,
  selected,
  onSelect,
  reconnectKey,
  caption,
}: {
  mjpegUrl: string;
  telemetry: Telemetry | null;
  stale: boolean;
  candidates: Candidates | null;
  selected: number | null;
  onSelect: (index: number) => void;
  reconnectKey: number; // bump to reopen the MJPEG stream (backend came back)
  caption?: string | null; // live voice caption, or a hint
}) {
  // MJPEG streams do not reconnect by themselves: retry on error, and when asked.
  const [retry, setRetry] = useState(0);
  const [videoError, setVideoError] = useState(false);
  useEffect(() => {
    if (!videoError) return;
    const t = setTimeout(() => {
      setVideoError(false);
      setRetry((r) => r + 1);
    }, 2000);
    return () => clearTimeout(t);
  }, [videoError]);

  const ap = telemetry?.autopilot_state ?? "IDLE";
  const color = AUTOPILOT_COLORS[ap] ?? AUTOPILOT_COLORS.IDLE;
  const subject = !stale ? telemetry?.subject ?? null : null;
  const phrase = telemetry?.mission?.intent?.subject_phrase;
  const src = `${mjpegUrl}${mjpegUrl.includes("?") ? "&" : "?"}r=${reconnectKey}.${retry}`;
  const frameAge = telemetry?.frame_age_ms;
  const lagging = typeof frameAge === "number" && frameAge >= FRAME_AGE_WARN_MS;

  return (
    <div className="video-fit relative min-h-0 flex-1">
      <div className="video-box relative mx-auto overflow-hidden rounded-2xl bg-neutral-900">
        {!videoError ? (
          <img
            src={src}
            alt="Drone video feed"
            draggable={false}
            onError={() => setVideoError(true)}
            className="absolute inset-0 h-full w-full select-none object-fill"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-neutral-500">
            Waiting for video
          </div>
        )}

        <SubjectOverlay telemetry={telemetry} stale={stale} candidates={candidates} selected={selected} onSelect={onSelect} />

        {ap !== "IDLE" && !stale && (
          <div
            className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1.5 text-[13px] backdrop-blur-md"
            title={
              subject
                ? `${subject.source} · track ${subject.track_id ?? "–"} · score ${subject.tracker_score.toFixed(2)}`
                : undefined
            }
          >
            <span className="h-2 w-2 rounded-full" style={{ background: color }} />
            <span className="font-medium text-white">{AUTOPILOT_LABEL[ap] ?? ap}</span>
            <span className="text-neutral-300">{phrase || AUTOPILOT_HELP[ap] || ""}</span>
          </div>
        )}
        <div
          className={`num pointer-events-none absolute right-4 top-4 rounded-full bg-black/60 px-2.5 py-1 text-xs backdrop-blur-md ${
            lagging ? "text-amber-300" : "text-neutral-400"
          }`}
          title="Age of the newest camera frame"
        >
          {fmtMs(frameAge)}
        </div>
        {caption && (
          <div className="pointer-events-none absolute inset-x-0 bottom-5 flex justify-center px-8">
            <div className="max-w-[80%] truncate rounded-xl bg-black/65 px-4 py-2 text-[15px] text-white backdrop-blur-md">
              {caption}
            </div>
          </div>
        )}
        {stale && (
          <div className="pointer-events-none absolute bottom-4 right-4 rounded-full bg-black/60 px-2.5 py-1 text-xs text-amber-300 backdrop-blur-md">
            Overlay hidden while telemetry is stale
          </div>
        )}
      </div>
    </div>
  );
}
