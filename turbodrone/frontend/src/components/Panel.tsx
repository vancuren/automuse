import type { ReactNode } from "react";

/** A section of the side panel: a quiet title, an optional caption, no box. */
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
    <section data-testid={testId} className={`flex min-h-0 flex-col border-t border-white/[0.07] py-4 first:border-t-0 first:pt-0 ${className}`}>
      <header className="mb-2.5 flex items-baseline justify-between gap-3">
        <h2 className="text-[13px] font-medium text-neutral-200">{title}</h2>
        {caption && <div className="min-w-0 truncate text-xs text-neutral-500">{caption}</div>}
      </header>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}
