/**
 * Sending OperatorCommand messages (new in this fork).
 *
 * Every command is {"type":"cmd","command":...} on the control socket (/ws).
 * LAND and E-STOP are safety commands: if the socket is down they go over HTTP POST /command
 * instead, retried for a few seconds. useControls also re-sends them over HTTP when telemetry
 * does not confirm them quickly (a socket that is open but not being read).
 */
import { URLS } from "../config";
import type { CommandName, OperatorCommand } from "./contracts";

export type Channel = "ws" | "http";

/** What the operator sees after pressing a command button or hotkey. */
export interface CommandReport {
  command: CommandName;
  channel: Channel | "none";
  ok: boolean;
  detail: string;
  at: number; // Date.now(), for display only
}

export const SAFETY_COMMANDS: ReadonlySet<CommandName> = new Set<CommandName>(["land", "estop"]);

export function buildCommand(
  command: CommandName,
  extra: { arm_phrase?: string; pack_id?: string } = {},
): OperatorCommand {
  return { type: "cmd", command, ...extra };
}

/** One POST /command with a timeout. Resolves true only on a 2xx answer. */
export async function postCommandOnce(cmd: OperatorCommand, timeoutMs = 1000): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(URLS.command, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cmd),
      signal: ctrl.signal,
      keepalive: true,
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** POST /command until it succeeds or `attempts` run out (about 250 ms apart). */
export async function postCommandWithRetry(cmd: OperatorCommand, attempts = 8): Promise<boolean> {
  for (let i = 0; i < attempts; i++) {
    if (await postCommandOnce(cmd)) return true;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}
