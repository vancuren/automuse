import type { ComponentProps, MouseEvent } from "react";

/**
 * Every button in the operator UI. It can never take keyboard focus:
 *  - tabIndex -1 keeps it out of the Tab order;
 *  - preventDefault on mousedown stops a click from focusing it.
 * So Space (push-to-talk) and Enter can never re-fire the last button pressed.
 * (React 19 passes ref as an ordinary prop, so HoldButton can still capture the pointer.)
 */
export default function UiButton({ className = "", onMouseDown, ...rest }: ComponentProps<"button">) {
  return (
    <button
      type="button"
      {...rest}
      tabIndex={-1}
      onMouseDown={(e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onMouseDown?.(e);
      }}
      className={`select-none outline-none ${className}`}
    />
  );
}
