import { useEffect, useRef } from "react";
import { isTypingInField } from "./useControls";

export interface HotkeyHandlers {
  onEstop: () => void;   // Esc, always (even while typing)
  onLand: () => void;    // L, unless typing in a text field
  onPttDown: () => void; // Space down
  onPttUp: () => void;   // Space up, and on blur / hidden tab
  onDigit?: (n: number) => boolean; // 1-9: pick a look-alike; returns true if it was used
}

/**
 * Safety hotkeys (new in this fork). Esc always wins; Space and L are ignored while typing in a
 * text field (the ARM box, which only exists on the ground before arming).
 *
 * The listeners sit on window in the capture phase, so they run before anything else on the
 * page. Space calls preventDefault on keydown and keyup: a focused button can never be
 * "clicked" by Space (browsers activate buttons on Space keyup), and the page never scrolls.
 * Buttons also never take focus (see UiButton and useNoButtonFocus).
 */
export function useHotkeys(handlers: HotkeyHandlers): void {
  // Handlers change every render; listeners read the latest through a ref.
  const h = useRef(handlers);
  h.current = handlers;

  useEffect(() => {
    let spaceHeld = false;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Escape" || e.key === "Escape") {
        e.preventDefault();
        (document.activeElement as HTMLElement | null)?.blur?.();
        if (!e.repeat) h.current.onEstop();
        return;
      }
      if (e.code === "Space") {
        if (isTypingInField() && !spaceHeld) return; // the ARM box (only shown before takeoff)
        e.preventDefault();
        if (!e.repeat && !spaceHeld) {
          spaceHeld = true;
          h.current.onPttDown();
        }
        return;
      }
      if (e.code === "KeyL" && !e.metaKey && !e.ctrlKey && !e.altKey && !isTypingInField()) {
        e.preventDefault();
        if (!e.repeat) h.current.onLand();
        return;
      }
      // Digit1..Digit9 / Numpad1..Numpad9 pick a candidate while the picker is shown.
      const digit = /^(?:Digit|Numpad)([1-9])$/.exec(e.code);
      if (digit && !e.repeat && !e.metaKey && !e.ctrlKey && !e.altKey && !isTypingInField()) {
        if (h.current.onDigit?.(Number(digit[1]))) e.preventDefault();
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        if (spaceHeld) {
          spaceHeld = false;
          h.current.onPttUp();
        }
      }
    };

    // A key held while the window loses focus never sends keyup: release push-to-talk.
    const release = () => {
      spaceHeld = false;
      h.current.onPttUp();
    };
    const onVisibility = () => {
      if (document.visibilityState !== "visible") release();
    };

    window.addEventListener("keydown", onKeyDown, { capture: true });
    window.addEventListener("keyup", onKeyUp, { capture: true });
    window.addEventListener("blur", release);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true });
      window.removeEventListener("keyup", onKeyUp, { capture: true });
      window.removeEventListener("blur", release);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);
}

/**
 * Second line of defense: if anything ever focuses a button (script, accessibility tools),
 * blur it at once so Space or Enter cannot re-fire it.
 */
export function useNoButtonFocus(): void {
  useEffect(() => {
    const onFocusIn = () => {
      const el = document.activeElement;
      if (el instanceof HTMLButtonElement) el.blur();
    };
    document.addEventListener("focusin", onFocusIn, { capture: true });
    return () => document.removeEventListener("focusin", onFocusIn, { capture: true });
  }, []);
}
