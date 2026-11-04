"""Backend glue for the patched turbodrone web_server.py: FLIGHT_MODE gate, capture-time estimate,
frame store, and the hub that routes every UI, perception and agent message to the governor.

Importing mhd.backend.capture or mhd.backend.hub needs turbodrone/backend on sys.path (the backend
runs from that directory).
"""
