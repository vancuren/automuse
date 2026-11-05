import { memo, useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import {
  COMMAND_LABEL,
  fetchFlights,
  fetchReplay,
  frameUrl,
  indexAt,
  type FlightSummary,
  type Replay,
  type ReplayMark,
} from "../lib/replay";
import { AUTOPILOT_COLORS, AUTOPILOT_LABEL, FLIGHT_STATE_STYLE, fmtClock } from "../lib/theme";
import { WireBar } from "./StickBars";
import SubjectOverlay from "./SubjectOverlay";
import UiButton from "./UiButton";

const SPEEDS = [1, 2, 4];
const MODE_LABEL: Record<string, string> = { real: "Real drone", twin: "Simulator", camera: "Camera only", mock: "Demo" };
const STICKS: { key: string; label: string }[] = [
  { key: "throttle", label: "Throttle" },
  { key: "yaw", label: "Yaw" },
  { key: "pitch", label: "Pitch" },
  { key: "roll", label: "Roll" },
];

function when(f: FlightSummary): string {
  if (!f.started_unix) return f.id;
  const d = new Date(f.started_unix * 1000);
  const today = new Date().toDateString() === d.toDateString();
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  return today ? `Today, ${time}` : `${d.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
}

/** What was said during the flight, for the list: the first voice command, else the commands. */
function gist(f: FlightSummary): string {
  const said = f.commands.find((c) => c.kind === "mission" && c.transcript);
  if (said?.transcript) return `“${said.transcript}”`;
  const names = f.commands.map((c) => (c.kind === "cmd" && c.command ? COMMAND_LABEL[c.command] : undefined)).filter(Boolean);
  return names.length ? names.slice(0, 3).join(", ") : "No commands";
}

const MARK_TONE: Record<ReplayMark["kind"], string> = {
  command: "text-neutral-100",
  voice: "text-neutral-100",
  mission: "text-neutral-300",
  reply: "text-neutral-400",
  state: "text-neutral-300",
  autopilot: "text-neutral-500",
  safety: "text-amber-300",
};

const MARK_TAG: Record<ReplayMark["kind"], string> = {
  command: "Command",
  voice: "Voice",
  mission: "",
  reply: "Drone",
  state: "Flight",
  autopilot: "Autopilot",
  safety: "Safety",
};

/** The story of the flight: every command and what it led to. Click a line to jump there. */
const MarkList = memo(function MarkList({
  marks,
  current,
  onSeek,
}: {
  marks: ReplayMark[];
  current: number;
  onSeek: (t: number) => void;
}) {
  const active = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    active.current?.scrollIntoView({ block: "nearest" });
  }, [current]);
  if (marks.length === 0) return <div className="text-[13px] text-neutral-500">Nothing was recorded.</div>;
  return (
    <div data-testid="replay-marks" className="flex flex-col">
      {marks.map((m, i) => (
        <UiButton
          key={i}
          ref={i === current ? active : undefined}
          data-testid="replay-mark"
          data-kind={m.kind}
          data-current={i === current ? "1" : "0"}
          onClick={() => onSeek(m.t)}
          className={`flex items-baseline gap-3 rounded-lg px-2.5 py-1.5 text-left text-[13px] ${
            i === current ? "bg-white/[0.07]" : "hover:bg-white/[0.04]"
          } ${i > current ? "opacity-50" : ""}`}
        >
          <span className="num w-9 shrink-0 text-xs text-neutral-600">{fmtClock(m.t / 1000)}</span>
          <span className="min-w-0 flex-1">
            <span className={m.failed ? "text-red-400" : MARK_TONE[m.kind]}>{m.label}</span>
            {(MARK_TAG[m.kind] || m.detail) && (
              <span className="block truncate text-xs text-neutral-600">
                {[MARK_TAG[m.kind], m.detail].filter(Boolean).join(" · ")}
              </span>
            )}
          </span>
        </UiButton>
      ))}
    </div>
  );
});

/**
 * Flight replay: pick a saved flight and watch it again, with the camera, the subject the drone was
 * following, the sticks it was sent, and every command on a timeline.
 * Mouse only on purpose: Esc, L and Space stay the live safety keys while this is open.
 */
export default function ReplayView({ onClose }: { onClose: () => void }) {
  const [flights, setFlights] = useState<FlightSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [t, setT] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetchFlights()
      .then((list) => {
        if (cancelled) return;
        setFlights(list);
        // Open the newest finished flight; one still being recorded has nothing to show yet.
        setSelectedId((cur) => cur ?? (list.find((f) => f.complete) ?? list[0])?.id ?? null);
      })
      .catch((e: unknown) => !cancelled && setError(String(e instanceof Error ? e.message : e)));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    setReplay(null);
    setPlaying(false);
    setT(0);
    fetchReplay(selectedId)
      .then((r) => {
        if (cancelled) return;
        setReplay(r);
        setPlaying(true);
      })
      .catch((e: unknown) => !cancelled && setError(String(e instanceof Error ? e.message : e)));
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  // The playhead: advanced by the frame clock while playing.
  const duration = replay?.durationMs ?? 0;
  useEffect(() => {
    if (!playing || duration <= 0) return;
    let raf = 0;
    let last = performance.now();
    const step = (now: number) => {
      const dt = Math.min(250, now - last);
      last = now;
      setT((cur) => {
        const next = cur + dt * speed;
        if (next >= duration) {
          setPlaying(false);
          return duration;
        }
        return next;
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, duration]);

  const seek = useCallback(
    (ms: number) => setT(Math.max(0, Math.min(duration, ms))),
    [duration],
  );

  // What the UI showed at the playhead.
  const tlmIndex = replay ? indexAt(replay.telemetry, t) : -1;
  const telemetry = replay && tlmIndex >= 0 ? replay.telemetry[tlmIndex].d : null;
  const frameIndex = replay ? indexAt(replay.frames, t) : -1;
  const wantFrame = replay && frameIndex >= 0 ? frameUrl(replay.id, replay.frames[frameIndex].i) : null;
  const markIndex = replay ? indexAt(replay.marks, t) : -1;

  // Show a frame only once it has loaded, so the picture never blinks between frames.
  const [shownFrame, setShownFrame] = useState<string | null>(null);
  useEffect(() => {
    if (!wantFrame) {
      setShownFrame(null);
      return;
    }
    let stale = false;
    const img = new Image();
    img.onload = () => !stale && setShownFrame(wantFrame);
    img.src = wantFrame;
    return () => {
      stale = true;
    };
  }, [wantFrame]);

  // Scrubber: click or drag anywhere on the track.
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const seekToPointer = (e: PointerEvent<HTMLDivElement>) => {
    const box = track.current?.getBoundingClientRect();
    if (!box || box.width <= 0) return;
    seek(((e.clientX - box.left) / box.width) * duration);
  };

  const fs = telemetry ? FLIGHT_STATE_STYLE[telemetry.flight_state] : null;
  const ap = telemetry?.autopilot_state ?? "IDLE";
  const flight = useMemo(() => flights?.find((f) => f.id === selectedId) ?? null, [flights, selectedId]);
  const pct = duration > 0 ? (t / duration) * 100 : 0;

  return (
    <div data-testid="replay-view" className="absolute inset-0 z-20 flex gap-6 bg-[#0a0a0a] p-5">
      {/* Flights */}
      <aside className="flex w-60 shrink-0 flex-col">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-[15px] font-medium text-neutral-100">Flights</h2>
          <UiButton data-testid="replay-close" onClick={onClose} className="text-[13px] text-neutral-400 hover:text-neutral-100">
            Back to live
          </UiButton>
        </div>
        <div data-testid="replay-flights" className="-mx-2 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto">
          {flights === null && !error && <div className="px-2 text-[13px] text-neutral-500">Loading</div>}
          {error && <div className="px-2 text-[13px] text-red-400">Could not load flights: {error}</div>}
          {flights?.length === 0 && (
            <div className="px-2 text-[13px] leading-relaxed text-neutral-500">
              No flights yet. Every flight is saved here automatically once it lands.
            </div>
          )}
          {flights?.map((f) => (
            <UiButton
              key={f.id}
              data-testid="replay-flight"
              data-selected={f.id === selectedId ? "1" : "0"}
              onClick={() => setSelectedId(f.id)}
              className={`flex flex-col gap-0.5 rounded-lg px-2.5 py-2 text-left ${
                f.id === selectedId ? "bg-white/[0.07]" : "hover:bg-white/[0.04]"
              }`}
            >
              <span className="flex items-baseline justify-between gap-2 text-[13px] text-neutral-100">
                {when(f)}
                <span className="num text-xs text-neutral-500">
                  {f.duration_ms ? fmtClock(f.duration_ms / 1000) : f.complete ? "" : "recording"}
                </span>
              </span>
              <span className="truncate text-xs text-neutral-500">
                {MODE_LABEL[f.flight_mode ?? ""] ?? f.flight_mode ?? ""} · {gist(f)}
              </span>
            </UiButton>
          ))}
        </div>
      </aside>

      {/* Player */}
      <section className="flex min-w-0 flex-1 flex-col gap-4">
        <div className="video-fit relative min-h-0 flex-1">
          <div className="video-box relative mx-auto overflow-hidden rounded-2xl bg-neutral-900">
            {shownFrame ? (
              <img
                data-testid="replay-frame"
                src={shownFrame}
                alt="Recorded camera frame"
                draggable={false}
                className="absolute inset-0 h-full w-full select-none object-fill"
              />
            ) : (
              <div className="absolute inset-x-0 bottom-5 flex justify-center text-[13px] text-neutral-500">
                {!selectedId
                  ? "Pick a flight"
                  : !replay
                    ? "Loading"
                    : replay.frames.length === 0
                      ? "No camera recording for this flight"
                      : ""}
              </div>
            )}
            <SubjectOverlay telemetry={telemetry} />
            {telemetry && fs && (
              <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1.5 text-[13px] backdrop-blur-md">
                <span className="h-2 w-2 rounded-full" style={{ background: ap !== "IDLE" ? AUTOPILOT_COLORS[ap] : fs.dot }} />
                <span data-testid="replay-state" data-state={telemetry.flight_state} className="font-medium text-white">
                  {fs.label}
                </span>
                {ap !== "IDLE" && (
                  <span className="text-neutral-300">
                    {AUTOPILOT_LABEL[ap]}
                    {telemetry.mission?.intent?.subject_phrase ? ` · ${telemetry.mission.intent.subject_phrase}` : ""}
                  </span>
                )}
              </div>
            )}
            <div className="pointer-events-none absolute bottom-4 right-4 rounded-full bg-black/60 px-2.5 py-1 text-xs text-neutral-300 backdrop-blur-md">
              Replay{flight ? ` · ${when(flight)}` : ""}
            </div>
          </div>
        </div>

        {/* Transport */}
        <div className="flex shrink-0 flex-col gap-3">
          <div className="flex items-center gap-4">
            <UiButton
              data-testid="replay-play"
              data-playing={playing ? "1" : "0"}
              disabled={!replay}
              onClick={() => {
                if (!playing && t >= duration) setT(0);
                setPlaying((p) => !p);
              }}
              className="h-9 w-[76px] rounded-lg bg-white text-[13px] font-semibold text-black hover:bg-neutral-200 disabled:bg-white/10 disabled:text-neutral-600"
            >
              {playing ? "Pause" : "Play"}
            </UiButton>
            <span data-testid="replay-time" className="num w-[84px] text-[13px] text-neutral-300">
              {fmtClock(t / 1000)} <span className="text-neutral-600">/ {fmtClock(duration / 1000)}</span>
            </span>
            <div
              ref={track}
              data-testid="replay-track"
              className="relative h-9 flex-1 cursor-pointer touch-none"
              onPointerDown={(e) => {
                dragging.current = true;
                track.current?.setPointerCapture(e.pointerId);
                seekToPointer(e);
              }}
              onPointerMove={(e) => dragging.current && seekToPointer(e)}
              onPointerUp={() => (dragging.current = false)}
              onPointerCancel={() => (dragging.current = false)}
            >
              <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-white/10">
                <div className="h-full rounded-full bg-white" style={{ width: `${pct}%` }} />
              </div>
              {replay?.marks
                .filter((m) => m.kind === "command" || m.kind === "voice")
                .map((m, i) => (
                  <span
                    key={i}
                    title={`${fmtClock(m.t / 1000)} · ${m.label}`}
                    className={`absolute top-1/2 h-3 w-0.5 -translate-y-1/2 rounded-full ${
                      m.kind === "voice" ? "bg-emerald-400" : m.failed ? "bg-red-400" : "bg-neutral-400"
                    }`}
                    style={{ left: `${duration > 0 ? (m.t / duration) * 100 : 0}%` }}
                  />
                ))}
              <span
                className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow"
                style={{ left: `${pct}%` }}
              />
            </div>
            <div className="flex rounded-lg bg-white/[0.06] p-0.5">
              {SPEEDS.map((s) => (
                <UiButton
                  key={s}
                  data-testid={`replay-speed-${s}`}
                  onClick={() => setSpeed(s)}
                  className={`num rounded-md px-2 py-1 text-xs font-medium ${
                    speed === s ? "bg-white/10 text-neutral-100" : "text-neutral-500 hover:text-neutral-200"
                  }`}
                >
                  {s}×
                </UiButton>
              ))}
            </div>
          </div>
          <div data-testid="replay-sticks" className="grid grid-cols-4 gap-6" title="Stick values sent to the drone at this moment (128 is centered)">
            {STICKS.map((a) => (
              <div key={a.key} className="flex items-center gap-2.5 text-xs">
                <span className="shrink-0 text-neutral-500">{a.label}</span>
                <WireBar value={telemetry?.sticks_wire?.[a.key]} />
                <span className="num w-7 shrink-0 text-right text-neutral-300">{telemetry?.sticks_wire?.[a.key] ?? "–"}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The story */}
      <aside className="flex w-[300px] shrink-0 flex-col">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-[15px] font-medium text-neutral-100">Commands</h2>
          <span className="text-xs text-neutral-500">
            {telemetry ? `flown by ${telemetry.active_source}` : ""}
          </span>
        </div>
        <div className="-mx-2.5 min-h-0 flex-1 overflow-y-auto">
          {replay ? <MarkList marks={replay.marks} current={markIndex} onSeek={seek} /> : null}
        </div>
      </aside>
    </div>
  );
}
