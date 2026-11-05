import { FRAME_H, FRAME_W, type Candidates, type Telemetry } from "../lib/contracts";
import { AUTOPILOT_COLORS } from "../lib/theme";

/**
 * The SAM 3.1 subject drawn over the camera picture (new). An SVG in 1280x720 pixel space stretched
 * over the 16:9 video box, so boxes and mask polygons from the contracts map 1:1 at any screen size.
 * Used by the live view and by the flight replay.
 */
export default function SubjectOverlay({
  telemetry,
  stale = false,
  candidates = null,
  selected = null,
  onSelect,
}: {
  telemetry: Telemetry | null;
  stale?: boolean;
  candidates?: Candidates | null;
  selected?: number | null;
  onSelect?: (index: number) => void;
}) {
  const ap = telemetry?.autopilot_state ?? "IDLE";
  const color = AUTOPILOT_COLORS[ap] ?? AUTOPILOT_COLORS.IDLE;
  const subject = !stale ? telemetry?.subject ?? null : null;
  const box = subject?.box ?? null;
  const phrase = telemetry?.mission?.intent?.subject_phrase || "subject";
  const confirmed = subject?.state === "confirmed";
  const poly = subject?.mask_poly ?? null;
  const label = `${phrase}  ${Math.round((subject?.confidence ?? 0) * 100)}%`;
  const labelW = label.length * 8.3 + 38;
  const cx = FRAME_W / 2;
  const cy = FRAME_H / 2;

  return (
    <svg
      data-testid="subject-overlay"
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${FRAME_W} ${FRAME_H}`}
      preserveAspectRatio="none"
    >
      {/* image center: the bearing reference the autopilot steers to */}
      <g stroke="white" strokeOpacity={0.5} strokeWidth={1.5}>
        <line x1={cx - 12} y1={cy} x2={cx + 12} y2={cy} />
        <line x1={cx} y1={cy - 12} x2={cx} y2={cy + 12} />
      </g>

      {poly && poly.length >= 3 && (
        <polygon
          data-testid="subject-mask"
          points={poly.map(([x, y]) => `${x},${y}`).join(" ")}
          fill={color}
          fillOpacity={0.16}
          stroke={color}
          strokeOpacity={0.9}
          strokeWidth={1.5}
          strokeLinejoin="round"
        />
      )}

      {box && (
        <g data-testid="subject-box" data-state={ap}>
          <line
            x1={cx}
            y1={cy}
            x2={(box.x1 + box.x2) / 2}
            y2={(box.y1 + box.y2) / 2}
            stroke={color}
            strokeWidth={1.5}
            strokeDasharray="3 7"
            strokeOpacity={0.6}
          />
          <rect
            x={box.x1}
            y={box.y1}
            width={Math.max(0, box.x2 - box.x1)}
            height={Math.max(0, box.y2 - box.y1)}
            fill="none"
            stroke={color}
            strokeWidth={2.5}
            strokeDasharray={confirmed ? undefined : "10 8"}
            rx={10}
          />
          {/* the label stays inside the picture, above the box (below it when the box touches the top) */}
          <g transform={`translate(${Math.max(8, Math.min(box.x1, FRAME_W - labelW - 8))}, ${box.y1 >= 44 ? box.y1 - 36 : Math.min(FRAME_H - 36, box.y2 + 8)})`}>
            <rect width={labelW} height={28} rx={14} fill="black" fillOpacity={0.72} />
            <circle cx={15} cy={14} r={4} fill={color} />
            <text x={27} y={19} fontSize={16} fontWeight={500} fill="white">
              {label}
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
            onClick={() => onSelect?.(c.index)}
          >
            <rect
              x={c.box.x1}
              y={c.box.y1}
              width={c.box.x2 - c.box.x1}
              height={c.box.y2 - c.box.y1}
              fill="white"
              fillOpacity={isSel ? 0.16 : 0.04}
              stroke={isSel ? "#22c55e" : "white"}
              strokeWidth={2.5}
              strokeDasharray="8 8"
              rx={10}
            />
            <circle cx={c.box.x1 + 24} cy={c.box.y1 + 24} r={17} fill={isSel ? "#22c55e" : "white"} />
            <text x={c.box.x1 + 24} y={c.box.y1 + 31} fontSize={20} fontWeight={600} textAnchor="middle" fill="#0a0a0a">
              {c.index}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
