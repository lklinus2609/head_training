"""FastAPI server for the FLAME expression viewer.

Serves the static web viewer and provides WebSocket streaming of expression weights.

Usage:
    # Basic static file serving (load sequences from .npy files):
    python server.py

    # With live model inference streaming:
    python server.py --checkpoint $WORK/checkpoints/d4head/stage3_best.pt

    Open http://localhost:8765 in your browser.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="FLAME Expression Viewer")

# Global state
SEQUENCES_DIR = Path(__file__).parent / "static" / "sequences"
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/sequences")
async def list_sequences():
    """List all available .npy sequence files."""
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    sequences = []
    for f in sorted(SEQUENCES_DIR.glob("*.npy")):
        data = np.load(str(f))
        sequences.append({
            "name": f.stem,
            "filename": f.name,
            "frames": data.shape[0],
            "dims": data.shape[1] if data.ndim > 1 else 1,
            "duration_s": data.shape[0] / 30.0,
        })
    return {"sequences": sequences}


@app.get("/api/sequences/{filename}")
async def get_sequence(filename: str):
    """Load a .npy sequence file and return as binary float32 data."""
    filepath = SEQUENCES_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".npy":
        return Response(status_code=404, content="Sequence not found")
    data = np.load(str(filepath)).astype(np.float32)
    # Return raw binary: first 8 bytes are (rows, cols) as uint32, then float32 data
    header = np.array([data.shape[0], data.shape[1]], dtype=np.uint32).tobytes()
    return Response(content=header + data.tobytes(), media_type="application/octet-stream")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time expression weight streaming.

    Protocol:
    - Server sends JSON frames: {"frame": int, "weights": [100 floats], "fps": 30}
    - Client can send: {"command": "play"/"pause"/"seek", "frame": int}
    """
    await websocket.accept()

    try:
        # Wait for client to specify a sequence
        init_msg = await websocket.receive_json()
        sequence_name = init_msg.get("sequence", "demo")
        fps = init_msg.get("fps", 30)

        filepath = SEQUENCES_DIR / f"{sequence_name}.npy"
        if not filepath.exists():
            await websocket.send_json({"error": f"Sequence '{sequence_name}' not found"})
            await websocket.close()
            return

        data = np.load(str(filepath)).astype(np.float32)
        T = data.shape[0]

        await websocket.send_json({
            "type": "init",
            "frames": T,
            "dims": int(data.shape[1]),
            "fps": fps,
        })

        # Stream frames at the specified FPS
        frame_interval = 1.0 / fps
        current_frame = 0
        playing = True

        while True:
            # Check for client commands (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                cmd = msg.get("command")
                if cmd == "pause":
                    playing = False
                elif cmd == "play":
                    playing = True
                elif cmd == "seek":
                    current_frame = max(0, min(msg.get("frame", 0), T - 1))
                elif cmd == "stop":
                    break
            except asyncio.TimeoutError:
                pass

            if playing and current_frame < T:
                weights = data[current_frame].tolist()
                await websocket.send_json({
                    "type": "frame",
                    "frame": current_frame,
                    "weights": weights,
                })
                current_frame += 1
                if current_frame >= T:
                    current_frame = 0  # Loop
                await asyncio.sleep(frame_interval)
            else:
                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass


# Mount static files last (after API routes)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    parser = argparse.ArgumentParser(description="FLAME Expression Viewer Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Starting FLAME Expression Viewer at http://{args.host}:{args.port}")
    print(f"Sequences directory: {SEQUENCES_DIR}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
