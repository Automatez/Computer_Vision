import asyncio
import json
import websockets

clients = set()

async def handler(websocket):
    addr = getattr(websocket, "remote_address", "unknown")
    print(f"\nClient connected: {addr}")
    clients.add(websocket)

    # Optional: prove the server can send something immediately
    try:
        await websocket.send(json.dumps({"type": "display", "text": "CONNECTED"}))
    except Exception as e:
        print(f"Initial send failed to {addr}: {e}")

    try:
        async for message in websocket:
            print(f"Received from {addr}: {message}")

            try:
                data = json.loads(message)
                if data.get("type") == "display" and data.get("text") is not None:
                    await broadcast(str(data["text"]))
            except Exception:
                await broadcast(message.strip())

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client closed: {addr}, code={e.code}, reason={e.reason!r}")

    except Exception as e:
        print(f"Client error from {addr}: {e}")

    finally:
        clients.discard(websocket)
        print(f"Client disconnected: {addr}")


async def broadcast(text: str, exclude=None):
    msg = json.dumps({"type": "display", "text": text})
    dead = []

    for ws in list(clients):
        if ws is exclude:
            continue

        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)

    for ws in dead:
        clients.discard(ws)

async def console_loop():
    while True:
        # IMPORTANT: input() must not block the event loop
        text = await asyncio.to_thread(input, "Enter display text: ")
        await broadcast(text)

async def main():
    print("Starting WebSocket server on ws://0.0.0.0:8765")
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await console_loop()

if __name__ == "__main__":
    asyncio.run(main())
