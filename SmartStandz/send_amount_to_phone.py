# put this file in the same folder on Jetson as orderBoard.py, and then import the push_amount function to use in the workflow that processes the order total amount, and call it with the IP address of the desktop computer running the phone app and the text of the amount to display on the phone screen

# note - be sure to limit the frequency of messages sent to avoid overwhelming the websocket server and client app on the phone
# maybe only send once per 0.5 seconds, and also try not to send if there's no change from last sent amount

# for more info on how to get the app running to receive this info see my Google Doc: My Drive > Roboflow > SmartStandz > App for Digital Display

# TODO: this opens the websocket every time, so might want to implement a way to keep it open and just send messages through it, but for now this is simpler and should work fine as long as we don't send too many messages too quickly

import websockets
import json

async def push_amount(server_ip: str, amount_text: str, open_timeout: float = 0.5):
    uri = f"ws://{server_ip}:8765"
    async with websockets.connect(uri, open_timeout=open_timeout, close_timeout=0.2) as ws:
        await ws.send(json.dumps({"type": "display", "text": amount_text}))

# Example usage:
# asyncio.run(push_amount("192.168.68.85", "$5.75"))
