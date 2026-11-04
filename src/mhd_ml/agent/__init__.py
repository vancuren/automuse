"""Voice and intent agent (PLAN.md section 4.6).

Modules:
- safety_listener: always-on local listener for stop / hover / land / abort (silero VAD + Omnilingual CTC).
- asr_meta:        Muse Voice Transcribe clients (realtime WebSocket, warm persistent session, file endpoint).
- asr_local:       Omnilingual ASR CTC-300M offline transcription for push-to-talk in edge mode.
- intent:          keyword grammar, Glimmer and Spark clients, and the hedged router.
- tts:             MMS-TTS spoken replies, falling back to macOS `say`.
- backend_link:    WebSocket link to the drone backend (/ws/agent?role=agent) with HTTP fallbacks.
- service:         the agent process (CLI).
- fake_backend, mock_llm: local stand-ins used by tests and dry runs.

Heavy imports (sherpa_onnx, torch, transformers, sounddevice) happen lazily inside the functions that
need them, so importing this package is cheap and works without a microphone.
"""
