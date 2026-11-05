import { useEffect, useState } from "react";
import Panel from "./Panel";

/**
 * Twin chase view (new): the twin's third-person camera, shown only in twin mode.
 * Display only: pointer-events none and tabIndex -1, so clicking it can never move keyboard
 * focus into the iframe (Esc and Space must keep reaching this page).
 * The iframe is mounted only once the URL answers, so a stopped twin shows a quiet placeholder.
 */
export default function ChaseView({ url }: { url: string }) {
  const [reachable, setReachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const probe = async () => {
      try {
        await fetch(url, { mode: "no-cors", cache: "no-store" });
        if (!cancelled) setReachable(true);
      } catch {
        if (!cancelled) {
          setReachable(false);
          timer = setTimeout(probe, 5000);
        }
      }
    };
    void probe();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [url]);

  return (
    <Panel title="Chase view" testId="chase-view" caption="simulator">
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-neutral-900" title={url}>
        {reachable ? (
          <iframe
            src={url}
            title="Twin chase view"
            tabIndex={-1}
            className="pointer-events-none absolute inset-0 h-full w-full border-0"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-neutral-600">
            Waiting for the simulator
          </div>
        )}
      </div>
    </Panel>
  );
}
