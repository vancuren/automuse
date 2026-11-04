import { useEffect, useState } from "react";
import { FRAME_H, FRAME_W, type Candidates, type Telemetry } from "../lib/contracts";
import { AUTOPILOT_COLORS, AUTOPILOT_HELP, fmtMs } from "../lib/theme";

/**
 * Drone camera (upstream MJPEG feed) with the SAM 3.1 subject overlay (new).
 * The overlay is an SVG in 1280x720 pixel space stretched over the 16:9 video box, so boxes and
 * mask polygons from the contracts map 1:1 onto the picture at any screen size.
 */
export default function VideoPanel({
  mjpegUrl,
  telemetry,
  stale,
  candidates,
  selected,
  onSelect,
  reconnectKey,
}: {
  mjpegUrl: string;
  telemetry: Telemetry | null;
  stale: boolean;
  candidates: Candidates | null;
  selected: number | null;
  onSelect: (index: number) => void;
  reconnectKey: number; // bump to reopen the MJPEG stream (backend came back)
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
  const box = subject?.box ?? null;
  const phrase = telemetry?.mission?.intent?.subject_phrase || "subject";
  const confirmed = subject?.state === "confirmed";
  const poly = subject?.mask_poly ?? null;
  const src = `${mjpegUrl}${mjpegUrl.includes("?") ? "&" : "?"}r=${reconnectKey}.${retry}`;

  return (
    <div className="video-fit relative min-h-0 flex-1">
      <div className="video-box relative mx-auto overflow-hidden rounded-xl border border-slate-800 bg-black">
        {!videoError ? (
          <img
            src={src}
            alt="Drone video feed"
            draggable={false}
            onError={() => setVideoError(true)}
            className="absolute inset-0 h-full w-full select-none object-fill"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-lg text-slate-500">
            no video from {mjpegUrl} · retrying
          </div>
        )}

        <svg
          data-testid="subject-overlay"
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${FRAME_W} ${FRAME_H}`}
          preserveAspectRatio="none"
        >
          {/* image center: the bearing reference the autopilot steers to */}
          <g stroke="white" strokeOpacity={0.45} strokeWidth={2}>
            <line x1={FRAME_W / 2 - 18} y1={FRAME_H / 2} x2={FRAME_W / 2 + 18} y2={FRAME_H / 2} />
            <line x1={FRAME_W / 2} y1={FRAME_H / 2 - 18} x2={FRAME_W / 2} y2={FRAME_H / 2 + 18} />
          </g>

          {poly && poly.length >= 3 && (
            <polygon
              data-testid="subject-mask"
              points={poly.map(([x, y]) => `${x},${y}`).join(" ")}
              fill={color}
              fillOpacity={0.28}
              stroke={color}
              strokeWidth={2.5}
              strokeLinejoin="round"
            />
          )}

          {box && (
            <g data-testid="subject-box" data-state={ap}>
              <line
                x1={FRAME_W / 2}
                y1={FRAME_H / 2}
                x2={(box.x1 + box.x2) / 2}
                y2={(box.y1 + box.y2) / 2}
                stroke={color}
                strokeWidth={2}
                strokeDasharray="6 8"
                strokeOpacity={0.8}
              />
              <rect
                x={box.x1}
                y={box.y1}
                width={Math.max(0, box.x2 - box.x1)}
                height={Math.max(0, box.y2 - box.y1)}
                fill="none"
                stroke={color}
                strokeWidth={4}
                strokeDasharray={confirmed ? undefined : "14 10"}
                rx={6}
              />
              <g transform={`translate(${box.x1}, ${Math.max(0, box.y1 - 34)})`}>
                <rect width={Math.max(240, phrase.length * 13 + 150)} height={30} rx={6} fill={color} />
                <text x={10} y={21} fontSize={19} fontWeight={700} fill="#0b1220">
                  {phrase} · {ap} · {Math.round((subject?.confidence ?? 0) * 100)}%
                </text>
              </g>
            </g>
          )}

          {candidates?.items.map((c) => {
            const isSel = selected === c.index;
            return (
              <g
                key={c.index}
                data-testid={`candidate-box-${c.index}`}
                className="pointer-events-auto cursor-pointer"
                onClick={() => onSelect(c.index)}
              >
                <rect
                  x={c.box.x1}
                  y={c.box.y1}
                  width={c.box.x2 - c.box.x1}
                  height={c.box.y2 - c.box.y1}
                  fill="white"
                  fillOpacity={isSel ? 0.2 : 0.06}
                  stroke={isSel ? "#22c55e" : "white"}
                  strokeWidth={4}
                  strokeDasharray="10 8"
                  rx={6}
                />
                <circle cx={c.box.x1 + 26} cy={c.box.y1 + 26} r={22} fill={isSel ? "#22c55e" : "white"} />
                <text
                  x={c.box.x1 + 26}
                  y={c.box.y1 + 35}
                  fontSize={26}
                  fontWeight={800}
                  textAnchor="middle"
                  fill="#0b1220"
                >
                  {c.index}
                </text>
              </g>
            );
          })}
        </svg>

        {/* HUD chips */}
        <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/60 px-2 py-1 text-xs text-slate-300">
          drone camera · MJPEG (turbodrone) · SAM 3.1 mask overlay (new)
        </div>
        <div className="pointer-events-none absolute right-3 top-3 rounded-md bg-black/60 px-2 py-1 font-mono text-xs text-slate-200">
          frame age {fmtMs(telemetry?.frame_age_ms)}
        </div>
        <div
          className="pointer-events-none absolute bottom-3 left-3 flex items-center gap-2 rounded-md bg-black/65 px-2.5 py-1.5 text-sm"
          style={{ color }}
        >
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: color }} />
          <span className="font-bold tracking-wider">{ap}</span>
          <span className="text-slate-300">{AUTOPILOT_HELP[ap] ?? ""}</span>
          {subject && (
            <span className="font-mono text-xs text-slate-400">
              {subject.source} · track {subject.track_id ?? "–"} · score {subject.tracker_score.toFixed(2)}
            </span>
          )}
        </div>
        {stale && (
          <div className="pointer-events-none absolute bottom-3 right-3 rounded-md bg-amber-600/90 px-2 py-1 text-xs font-semibold text-white">
            overlay hidden: telemetry stale
          </div>
        )}
      </div>
    </div>
  );
}
