import { useEffect, useRef, useState } from "react";
import { BACKEND_URL, TWIN_CHASE_URL, URLS } from "./config";
import { isCameraOnly } from "./lib/gates";
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
import ReplayView from "./components/ReplayView";
import SafetyFlags from "./components/SafetyFlags";
import StickBars from "./components/StickBars";
import UiButton from "./components/UiButton";
import VideoPanel from "./components/VideoPanel";
import VoicePanel from "./components/VoicePanel";

/**
 * The operator's screen: the camera, what the drone heard and is doing, and three actions
 * (take off, talk, land) with Emergency stop always in the same corner.
 * The shell (MJPEG feed, keyboard/gamepad input) is turbodrone's; the rest is new.
 */
const DETAILS_KEY = "mhd.details";

function readDetails(): boolean {
  try {
    return window.localStorage.getItem(DETAILS_KEY) === "1";
  } catch {
    return false;
  }
}

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

  // Camera-only runs are twin mode with the real camera and no twin server: no chase view.
  const twin = tlm.telemetry?.flight_mode === "twin" && !isCameraOnly(tlm.telemetry);

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

  // Sticks and latency are for debugging: off by default, remembered per browser.
  const [details, setDetails] = useState(readDetails);
  const toggleDetails = () =>
    setDetails((on) => {
      try {
        window.localStorage.setItem(DETAILS_KEY, on ? "0" : "1");
      } catch {
        /* private mode: the choice just is not remembered */
      }
      return !on;
    });

  // Flight replay covers the camera only; the header and the command bar stay live under it.
  // It is for the ground: taking off closes it, so the live camera is never hidden in the air.
  const [replayOpen, setReplayOpen] = useState(false);
  const flying = flightState === "TAKING_OFF" || flightState === "AIRBORNE" || flightState === "LANDING";
  useEffect(() => {
    if (flying) setReplayOpen(false);
  }, [flying]);

  // Over the camera: what the drone is hearing while the operator talks, else the next step.
  const airborne = flightState === "AIRBORNE";
  const caption = agent.ptt
    ? agent.partial
      ? `${agent.partial}…`
      : "Listening"
    : airborne && !mission
      ? "Hold Space and say what to follow"
      : null;
  const headerBtn = "h-8 rounded-lg px-3 text-[13px] font-medium";

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#0a0a0a] text-neutral-200">
      <header className="flex h-14 shrink-0 items-center justify-between gap-6 border-b border-white/[0.07] px-5">
        <div className="flex min-w-0 items-center gap-4">
          <div className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-white">
            <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
              <circle cx="9" cy="9" r="8" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="9" cy="9" r="2.5" fill="currentColor" />
            </svg>
            Follow
          </div>
          <ModeBanner telemetry={tlm.telemetry} stale={tlm.stale} />
          <FlightStateBadges telemetry={tlm.telemetry} stale={tlm.stale} />
        </div>
        <div className="flex items-center gap-5">
          <LinkChips
            control={controls.linkStatus}
            hbOpen={hb.open}
            hbRate={hb.rateHz}
            telemetryStale={tlm.stale}
            telemetryRate={tlm.rateHz}
            agent={agent.status}
          />
          <PackTimer
            telemetry={tlm.telemetry}
            stale={tlm.stale}
            onPackSwap={(id) => controls.sendCommand("pack_swap", { pack_id: id })}
          />
          <div className="flex items-center gap-1">
            <UiButton
              data-testid="details-toggle"
              data-on={details ? "1" : "0"}
              onClick={toggleDetails}
              title="Show or hide sticks and latency"
              className={`${headerBtn} ${details ? "bg-white/10 text-neutral-100" : "text-neutral-400 hover:text-neutral-100"}`}
            >
              Details
            </UiButton>
            <UiButton
              data-testid="replay-open"
              disabled={flying}
              onClick={() => setReplayOpen((open) => !open)}
              title={flying ? "Replay is available on the ground" : "Watch a saved flight again"}
              className={`${headerBtn} ${
                replayOpen
                  ? "bg-white text-black"
                  : flying
                    ? "cursor-not-allowed text-neutral-700"
                    : "bg-white/[0.06] text-neutral-100 hover:bg-white/10"
              }`}
            >
              Flights
            </UiButton>
          </div>
        </div>
      </header>
      <LinkBanner
        controlStatus={controls.linkStatus}
        attempts={controls.reconnectAttempts}
        telemetryStale={tlm.stale}
        backendUrl={BACKEND_URL}
        heartbeatProblem={heartbeatProblem}
      />

      <main className="relative grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_clamp(320px,24vw,400px)] gap-6 p-5">
        <div className="flex min-h-0 flex-col">
          <VideoPanel
            mjpegUrl={URLS.mjpeg}
            telemetry={tlm.telemetry}
            stale={tlm.stale}
            candidates={candidates}
            selected={agent.selected}
            onSelect={agent.selectCandidate}
            reconnectKey={videoKey}
            caption={caption}
          />
        </div>

        <aside className="flex min-h-0 flex-col overflow-y-auto pr-1 [&>*]:shrink-0">
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
          <SafetyFlags telemetry={tlm.telemetry} stale={tlm.stale} events={agent.events} />
          {twin && <ChaseView url={TWIN_CHASE_URL} />}
          {details && (
            <>
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
            </>
          )}
        </aside>

        {replayOpen && <ReplayView onClose={() => setReplayOpen(false)} />}
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
        className="flex h-7 shrink-0 items-center justify-between px-5 text-[11px] text-neutral-600"
      >
        <span>UI shell from turbodrone (Apache-2.0)</span>
        <span>
          {BACKEND_URL} · {controls.droneType}
        </span>
      </footer>
    </div>
  );
}

export default App;
