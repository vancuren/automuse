import { useEffect, useRef, useState } from "react";
import { BACKEND_URL, TWIN_CHASE_URL, URLS } from "./config";
import { useAgentLink } from "./hooks/useAgentLink";
import { useControls } from "./hooks/useControls";
import { useHeartbeat } from "./hooks/useHeartbeat";
import { useHotkeys, useNoButtonFocus } from "./hooks/useHotkeys";
import { useTelemetry } from "./hooks/useTelemetry";
import CandidatePicker from "./components/CandidatePicker";
import ChaseView from "./components/ChaseView";
import CommandBar from "./components/CommandBar";
import { FlightStateBadges, LinkChips, PackTimer } from "./components/FlightStatus";
import LatencyBars from "./components/LatencyBars";
import MissionCard from "./components/MissionCard";
import { LinkBanner, ModeBanner } from "./components/ModeBanner";
import SafetyFlags from "./components/SafetyFlags";
import StickBars from "./components/StickBars";
import VideoPanel from "./components/VideoPanel";
import VoicePanel from "./components/VoicePanel";

/**
 * Operator dashboard: one screen that tells the whole story.
 * The shell (MJPEG feed, keyboard/gamepad input) is turbodrone's; the rest is new.
 */
function App() {
  const tlm = useTelemetry();
  const hb = useHeartbeat();
  const agent = useAgentLink();
  const flightState = tlm.telemetry && !tlm.stale ? tlm.telemetry.flight_state : null;
  const controls = useControls({ flightState });

  useNoButtonFocus();

  // Mission card: the backend's mission (telemetry) unless the agent announced a newer one.
  const tlmMission = tlm.telemetry?.mission ?? null;
  const tlmMissionId = tlmMission?.mission_id ?? null;
  const [tlmMissionAt, setTlmMissionAt] = useState(0);
  useEffect(() => {
    setTlmMissionAt(Date.now());
  }, [tlmMissionId]);
  const agentIsNewer =
    agent.agentMission !== null &&
    agent.agentMission.mission_id !== tlmMissionId &&
    agent.agentMissionAt > tlmMissionAt;
  // Telemetry clearing its mission (null) clears the card too, unless the agent spoke since.
  const mission = agentIsNewer ? agent.agentMission : tlmMission;

  // Look-alike picker: once the autopilot follows, or the mission changes, the question is moot.
  const apState = tlm.telemetry?.autopilot_state ?? null;
  const { clearCandidates } = agent;
  useEffect(() => {
    if (apState === "FOLLOW") clearCandidates();
  }, [apState, clearCandidates]);
  const candidates =
    agent.candidates && (!tlmMissionId || agent.candidates.mission_id === tlmMissionId) ? agent.candidates : null;

  // Safety hotkeys: Esc E-STOP, L LAND, Space push-to-talk; 1-9 pick a look-alike while shown.
  useHotkeys({
    onEstop: controls.emergencyStop,
    onLand: controls.land,
    onPttDown: agent.pttDown,
    onPttUp: agent.pttUp,
    onDigit: (n) => (candidates?.items.some((c) => c.index === n) ? agent.selectCandidate(n) : false),
  });

  // Reopen the MJPEG stream when the control link comes back after a drop.
  const [videoKey, setVideoKey] = useState(0);
  const wasOpen = useRef<boolean | null>(null);
  useEffect(() => {
    const open = controls.linkStatus === "open";
    if (open && wasOpen.current === false) setVideoKey((k) => k + 1);
    if (open) wasOpen.current = true;
    else if (wasOpen.current) wasOpen.current = false;
  }, [controls.linkStatus]);

  const twin = tlm.telemetry?.flight_mode === "twin";

  // Heartbeat trouble while the control link itself is fine: the worker is dead or its socket
  // is down, or the governor says it is not getting our heartbeat. Give the worker 3 s to start.
  const [startedAt] = useState(() => Date.now());
  const [, setTick] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setTick((n) => n + 1), 3100);
    return () => clearTimeout(t);
  }, []);
  const governorMissesHb = !tlm.stale && (tlm.telemetry?.safety_flags ?? []).includes("heartbeat_missing");
  let heartbeatProblem: string | null = null;
  if (hb.error) {
    heartbeatProblem = `heartbeat worker failed: ${hb.error}`;
  } else if (controls.linkStatus === "open" && Date.now() - startedAt > 3000) {
    if (!hb.open) heartbeatProblem = "heartbeat socket down";
    else if (governorMissesHb) heartbeatProblem = "the governor reports no UI heartbeat";
  }

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#070b14] text-slate-200">
      <ModeBanner telemetry={tlm.telemetry} stale={tlm.stale} />
      <LinkBanner
        controlStatus={controls.linkStatus}
        attempts={controls.reconnectAttempts}
        telemetryStale={tlm.stale}
        backendUrl={BACKEND_URL}
        heartbeatProblem={heartbeatProblem}
      />

      <header className="flex h-[66px] shrink-0 items-center justify-between gap-4 border-b border-slate-800 px-4">
        <div className="min-w-0">
          <div className="truncate text-lg font-black tracking-[0.08em] text-white">VOICE FOLLOW DRONE · OPERATOR</div>
          <div className="truncate text-xs text-slate-400">
            voice → mission (Muse Glimmer + LoRA / Muse Spark) → SAM 3.1 subject → autopilot inside a safety governor
          </div>
        </div>
        <FlightStateBadges telemetry={tlm.telemetry} stale={tlm.stale} />
        <div className="flex items-center gap-3">
          <PackTimer
            telemetry={tlm.telemetry}
            stale={tlm.stale}
            onPackSwap={(id) => controls.sendCommand("pack_swap", { pack_id: id })}
          />
          <LinkChips
            control={controls.linkStatus}
            hbOpen={hb.open}
            hbRate={hb.rateHz}
            telemetryStale={tlm.stale}
            telemetryRate={tlm.rateHz}
            agent={agent.status}
          />
        </div>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_clamp(400px,29vw,560px)] gap-3 p-3">
        <div className="flex min-h-0 flex-col gap-3">
          <VideoPanel
            mjpegUrl={URLS.mjpeg}
            telemetry={tlm.telemetry}
            stale={tlm.stale}
            candidates={candidates}
            selected={agent.selected}
            onSelect={agent.selectCandidate}
            reconnectKey={videoKey}
          />
          <div className="grid shrink-0 grid-cols-3 gap-3">
            <StickBars
              telemetry={tlm.telemetry}
              stale={tlm.stale}
              localAxes={controls.axes}
              sendingAxes={controls.sendingAxes}
              mode={controls.mode}
              setMode={controls.setMode}
              gamepadConnected={controls.gamepadConnected}
            />
            <LatencyBars telemetry={tlm.telemetry} stale={tlm.stale} />
            <SafetyFlags telemetry={tlm.telemetry} stale={tlm.stale} events={agent.events} />
          </div>
        </div>

        <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto [&>*]:shrink-0">
          <VoicePanel
            ptt={agent.ptt}
            partial={agent.partial}
            lines={agent.lines}
            agentStatus={agent.status}
            health={agent.health}
            pttError={agent.pttError}
          />
          <MissionCard mission={mission} provisional={agentIsNewer} />
          <CandidatePicker candidates={candidates} selected={agent.selected} onSelect={agent.selectCandidate} />
          {twin && <ChaseView url={TWIN_CHASE_URL} />}
        </aside>
      </main>

      <CommandBar
        telemetry={tlm.telemetry}
        stale={tlm.stale}
        control={controls.linkStatus}
        caps={controls.commandCapabilities}
        send={controls.sendCommand}
        onTakeoff={controls.takeOff}
        onLand={controls.land}
        onEstop={controls.emergencyStop}
        ptt={agent.ptt}
        onPttDown={agent.pttDown}
        onPttUp={agent.pttUp}
        tilt={controls.cameraTiltDirection}
        setTilt={controls.setCameraTiltDirection}
        lastCommand={controls.lastCommand}
      />

      <footer
        data-testid="footer"
        className="flex h-6 shrink-0 items-center justify-between border-t border-slate-900 bg-black px-4 text-[11px] text-slate-500"
      >
        <span>
          UI shell from turbodrone (Apache-2.0) · new: voice, mission, SAM overlay, safety, latency and twin panels
        </span>
        <span className="font-mono">
          backend {BACKEND_URL} · {controls.droneType}
        </span>
      </footer>
    </div>
  );
}

export default App;
