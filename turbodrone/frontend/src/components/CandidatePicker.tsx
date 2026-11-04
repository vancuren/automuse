import type { Candidates } from "../lib/contracts";
import Panel from "./Panel";
import UiButton from "./UiButton";

/** Look-alike picker (new): perception found several matches; click one to send Select. */
export default function CandidatePicker({
  candidates,
  selected,
  onSelect,
}: {
  candidates: Candidates | null;
  selected: number | null;
  onSelect: (index: number) => void;
}) {
  if (!candidates || candidates.items.length === 0) return null;
  return (
    <Panel title="Which one?" testId="candidate-picker" caption="look-alikes tie · click to choose">
      <div className="flex flex-wrap gap-2">
        {candidates.items.map((c) => (
          <UiButton
            key={c.index}
            data-testid={`candidate-${c.index}`}
            onClick={() => onSelect(c.index)}
            className={`flex min-w-[120px] flex-col items-start rounded-lg border-2 px-3 py-1.5 text-left ${
              selected === c.index
                ? "border-emerald-400 bg-emerald-900/50 text-emerald-100"
                : "border-slate-600 bg-slate-800 text-slate-100 hover:border-white"
            }`}
          >
            <span className="text-2xl font-black">{c.index}</span>
            <span className="font-mono text-[11px] text-slate-300">
              conf {Math.round(c.confidence * 100)}% · color {Math.round(c.color_score * 100)}%
            </span>
          </UiButton>
        ))}
      </div>
    </Panel>
  );
}
