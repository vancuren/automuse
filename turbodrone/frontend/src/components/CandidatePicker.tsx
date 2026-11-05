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
    <Panel title="Which one?" testId="candidate-picker" caption="Click one, or press its number">
      <div className="flex flex-wrap gap-2">
        {candidates.items.map((c) => (
          <UiButton
            key={c.index}
            data-testid={`candidate-${c.index}`}
            onClick={() => onSelect(c.index)}
            title={`confidence ${Math.round(c.confidence * 100)}% · color match ${Math.round(c.color_score * 100)}%`}
            className={`flex h-11 min-w-[72px] items-center justify-center gap-2 rounded-xl px-4 text-[15px] font-medium ring-1 ring-inset ${
              selected === c.index
                ? "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40"
                : "bg-white/[0.06] text-neutral-100 ring-white/10 hover:bg-white/10"
            }`}
          >
            {c.index}
            <span className="num text-xs font-normal text-neutral-500">{Math.round(c.confidence * 100)}%</span>
          </UiButton>
        ))}
      </div>
    </Panel>
  );
}
