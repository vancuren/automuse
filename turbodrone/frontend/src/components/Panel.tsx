import type { ReactNode } from "react";

/** Dashboard panel frame: small uppercase title, optional right-side caption. */
export default function Panel({
  title,
  caption,
  children,
  className = "",
  testId,
}: {
  title: string;
  caption?: ReactNode;
  children: ReactNode;
  className?: string;
  testId?: string;
}) {
  return (
    <section
      data-testid={testId}
      className={`flex min-h-0 flex-col rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 ${className}`}
    >
      <header className="mb-1.5 flex items-baseline justify-between gap-2">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">{title}</h2>
        {caption && <div className="truncate text-[11px] text-slate-500">{caption}</div>}
      </header>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}
