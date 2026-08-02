# THIS VERSION was the one used for the demo and the LinkedIn post (slightly slower run time)
# it is also saved as a version in Github with tags for all relevant files

#loading to Jetson:
# scp "orderBoard.py" automatez@192.168.68.91:/home/automatez/smartstandz

#Setting up Jetson:
# physical set up - Basler directly overhead with Order Board square to frame, turn on extra overhead light, turn Basler on, then Jetson on.
# ssh login via terminal
# cd smartstandz
# docker start inference-server
# source .venv/bin/activate
# export ROBOFLOW_API_KEY="your_api_key"
# be sure send_amount_to_phone.py is in the same folder and has the correct IP address for the desktop computer running the phone app
# use Google Drive > Roboflow > SmartStandz > App for Digital Display doc for info on how to get the phone app running to receive the display info
# python orderBoard.py

import os
from inference import InferencePipeline
from inference_sdk import InferenceHTTPClient
from pathlib import Path
import base64
import mimetypes
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from inference.core.interfaces.camera.entities import VideoFrame, VideoFrameProducer, SourceProperties  # optional (for type hints)
from pypylon import pylon
import numpy as np
import asyncio
from send_amount_to_phone import push_amount

# region variable assignments
RAK = os.getenv("ROBOFLOW_API_KEY")
if not RAK:
    raise ValueError("ROBOFLOW_API_KEY is not set in the environment.")


COUNT_TO_LIMIT_RUN = 0
FRAME_STRIDE = 2
FAILS_BEFORE_ERR = 6
FRAME_COUNTER = 0
CONSECUTIVE_FAILURES = 0
LAST_TOTAL_ORDER: Optional[float] = None

# ---- Config for Workflow B ----
HOST = "http://localhost:9001"          # inference server in Docker on Jetson
WORKSPACE = "manufacturing-n8ggq"
WORKFLOW_B_ID = "count-sliders"
_WORKFLOW_B_CLIENT = InferenceHTTPClient(api_url=HOST, api_key=RAK)
DEBUG_CROP_DIR = Path("debug_crops")
DEBUG_CROP_DIR.mkdir(exist_ok=True)
DEBUG_FAIL_DIR = Path("debug_fail_frames")
DEBUG_FAIL_DIR.mkdir(exist_ok=True)
DEBUG_PREVIEW_DIR = Path("debug_rail_preview")
DEBUG_PREVIEW_DIR.mkdir(exist_ok=True)

PREVIEW_CANVAS_W = 1280
PREVIEW_CANVAS_H = 960
PREVIEW_FPS = 2.0  # 2 FPS => 0.5 seconds per frame
_PREVIEW_WRITER: Optional[cv2.VideoWriter] = None
_PREVIEW_WRITER_PATH: Optional[Path] = None

MAX_PROCESSED_FRAMES = 20
PROCESSED_FRAME_COUNTER = 0

DISPLAY_SERVER_IP = os.getenv("SMARTSTANDZ_DISPLAY_SERVER_IP", "192.168.68.85")  # default to this IP if env var not set, but be sure to set it to the correct IP for your desktop computer running the phone app
LAST_SENT_DISPLAY_TEXT = None

# endregion

#NOTE: only for static frame testing, not needed with video
def file_to_data_uri(image_path: str) -> str:
    """
    Not needed in production - only used for testing with a static image file instead of a video frame.
    Convert an image file to a data URI suitable for Roboflow workflow 'image' inputs.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "image/jpeg"

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

#NOTE: pass in cropped orderboard image and location of a rail to count sliders for, return slider count
def get_count_of_sliders(
    image_data_uri: str,
    x_center: int,
    y_center: int,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    
    """
    Use Roboflow SDK method to run a second workflow 
    This will detect and count the sliders on a given rail
    Pass in the image and the coordinates of the right side of a rail that you want to use
    Will be based on the cropped order board image and the ArUco tag coordinates for that rail as inputs
    This function will be called in the sink function of the first workflow for each detected rail to get the slider count per rail which will then be used to calculate total order cost.
    """
    return _WORKFLOW_B_CLIENT.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_B_ID,
        images={"image": image_data_uri},
        parameters={
            "x_center": int(x_center),
            "y_center": int(y_center),
            "width": int(width),
            "height": int(height),
        },
        use_cache=False,
    )

#NOTE: pass in a video frame, return cropped order board image
def crop_order_board_videoframe_to_data_uri(
    video_frame: "VideoFrame",
    order_board_coords: List[Dict[str, Any]],
    cls_name: str = "orderboard",
    pad_px: int = 0,
    out_ext: str = ".jpg",   # ".jpg" or ".png"
    jpeg_quality: int = 95,
) -> Tuple[str, np.ndarray]:
    """
    - Uses video_frame.image (np.ndarray, OpenCV BGR) as the source image
    - Finds the first bbox with class == cls_name in aruco_tag_coords
    - Crops using center-based bbox (x,y center; width,height size)
    - Returns (data_uri, crop_bgr)
    """

    # 1) find the order_board dict
    order_board: Optional[Dict[str, Any]] = next(
        (d for d in order_board_coords if isinstance(d, dict) and d.get("class") == cls_name),
        None,
    )
    if order_board is None:
        raise ValueError(f"No '{cls_name}' box found in order_board_coords.")

    # 2) get frame as numpy array (OpenCV BGR)
    frame_bgr: np.ndarray = video_frame.image  # VideoFrame.image is np.ndarray

    h_img, w_img = frame_bgr.shape[:2]

    print(f"[DEBUG] frame size: {w_img}x{h_img}, order_board data: {order_board}")
    
    # 3) center bbox -> corners
    cx = float(order_board["x"])
    cy = float(order_board["y"])
    bw = float(order_board["width"])
    bh = float(order_board["height"])

    print(f"[DEBUG] order_board bbox: center=({cx},{cy}), size=({bw}x{bh}), pad={pad_px}px")  #this is not looking correct for an abacus detection, am I passing the right frame in?
    

    x1 = int(round(cx - bw / 2)) - pad_px
    y1 = int(round(cy - bh / 2)) - pad_px
    x2 = int(round(cx + bw / 2)) + pad_px
    y2 = int(round(cy + bh / 2)) + pad_px

    # clamp to image bounds
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(1, min(w_img, x2))
    y2 = max(1, min(h_img, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop after clamping: ({x1},{y1})-({x2},{y2})")

    crop = frame_bgr[y1:y2, x1:x2]

    # 4) encode to bytes
    out_ext = out_ext.lower()
    if out_ext not in (".jpg", ".jpeg", ".png"):
        raise ValueError("out_ext must be '.jpg'/.jpeg or '.png'")

    if out_ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
        ok, buf = cv2.imencode(out_ext, crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        mime = "image/png"
        ok, buf = cv2.imencode(out_ext, crop)

    if not ok:
        raise RuntimeError("cv2.imencode failed")

    # 5) build data URI
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}", crop

#NOTE: debug showing order totals
def _emit_total_order() -> None:
    if CONSECUTIVE_FAILURES >= FAILS_BEFORE_ERR:
        print("Total Order: ERR")
    elif LAST_TOTAL_ORDER is not None:
        print(f"Total Order: ${LAST_TOTAL_ORDER:.2f}")
    else:
        print("Total Order: N/A")


def _center_pad_to_canvas(image_bgr: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    img_h, img_w = image_bgr.shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    copy_w = min(img_w, canvas_w)
    copy_h = min(img_h, canvas_h)

    src_x1 = max((img_w - copy_w) // 2, 0)
    src_y1 = max((img_h - copy_h) // 2, 0)
    src_x2 = src_x1 + copy_w
    src_y2 = src_y1 + copy_h

    dst_x1 = (canvas_w - copy_w) // 2
    dst_y1 = (canvas_h - copy_h) // 2
    dst_x2 = dst_x1 + copy_w
    dst_y2 = dst_y1 + copy_h

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image_bgr[src_y1:src_y2, src_x1:src_x2]
    return canvas


def _get_preview_writer() -> cv2.VideoWriter:
    global _PREVIEW_WRITER
    global _PREVIEW_WRITER_PATH

    if _PREVIEW_WRITER is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _PREVIEW_WRITER_PATH = DEBUG_PREVIEW_DIR / f"rail_preview_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _PREVIEW_WRITER = cv2.VideoWriter(
            str(_PREVIEW_WRITER_PATH),
            fourcc,
            PREVIEW_FPS,
            (PREVIEW_CANVAS_W, PREVIEW_CANVAS_H),
        )
        if not _PREVIEW_WRITER.isOpened():
            raise RuntimeError(f"Unable to open preview writer at {_PREVIEW_WRITER_PATH}")
        print(f"[DEBUG] Writing rail preview video to: {_PREVIEW_WRITER_PATH}")

    return _PREVIEW_WRITER

def _release_preview_writer() -> None:
    global _PREVIEW_WRITER
    global _PREVIEW_WRITER_PATH

    if _PREVIEW_WRITER is not None:
        _PREVIEW_WRITER.release()
        _PREVIEW_WRITER = None

        if _PREVIEW_WRITER_PATH is not None:
            print(f"[DEBUG] Saved rail preview video: {_PREVIEW_WRITER_PATH}")

def _extract_slider_box(
    slider_count_result: List[Dict[str, Any]],
    fallback_x: int,
    fallback_y: int,
    fallback_w: int,
    fallback_h: int,
) -> Tuple[int, int, int, int]:
    """
    Try to pull a box from workflow output, while safely falling back to the rail box used as input.
    Returns center-based box: (x_center, y_center, width, height)
    """
    x_center = int(fallback_x)
    y_center = int(fallback_y)
    width = int(fallback_w)
    height = int(fallback_h)

    if isinstance(slider_count_result, list) and slider_count_result and isinstance(slider_count_result[0], dict):
        first = slider_count_result[0]

        # Common direct shape: {"x":..., "y":..., "width":..., "height":...}
        if all(k in first for k in ("x", "y", "width", "height")):
            x_center = int(first["x"])
            y_center = int(first["y"])
            width = int(first["width"])
            height = int(first["height"])
            return x_center, y_center, width, height

        # Common nested shape: {"predictions":[{"x":...,"y":...,"width":...,"height":...}, ...]}
        preds = first.get("predictions")
        if isinstance(preds, list) and preds and isinstance(preds[0], dict):
            p0 = preds[0]
            if all(k in p0 for k in ("x", "y", "width", "height")):
                x_center = int(p0["x"])
                y_center = int(p0["y"])
                width = int(p0["width"])
                height = int(p0["height"])

    return x_center, y_center, width, height

def maybe_send_total_to_phone(total_order_cost: float) -> None:
    global LAST_SENT_DISPLAY_TEXT

    amount_text = f"${total_order_cost:.2f}"

    # Don't resend the same amount over and over
    if amount_text == LAST_SENT_DISPLAY_TEXT:
        return

    LAST_SENT_DISPLAY_TEXT = amount_text

    try:
        asyncio.run(push_amount(DISPLAY_SERVER_IP, amount_text))
        print(f"[DISPLAY] Sent to phone: {amount_text}")
    except Exception as e:
        print(f"[DISPLAY] Failed to send amount: {e}")

def my_sink(result, video_frame):
    """
    Sink function for processing results from the first workflow.
    Also includes running of a second workflow for counting sliders and calculating order totals.
    Runs the full processing pipeline once every FRAME_STRIDE frames.
    """
    global COUNT_TO_LIMIT_RUN
    global FRAME_COUNTER
    global CONSECUTIVE_FAILURES
    global LAST_TOTAL_ORDER
    global PROCESSED_FRAME_COUNTER

    # Optional test limiter (disabled by default unless you change this condition).
    COUNT_TO_LIMIT_RUN += 1

    FRAME_COUNTER += 1
    if FRAME_COUNTER % FRAME_STRIDE != 0:
        _emit_total_order()
        return

    PROCESSED_FRAME_COUNTER += 1    #to end the video stream after a certain number of 
    print(f"[DEBUG] Processed frame {PROCESSED_FRAME_COUNTER}/{MAX_PROCESSED_FRAMES}")

    try:
        if isinstance(result, list):
            if not result:
                raise ValueError("Empty result list")
            result = result[0]
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected result type: {type(result)}")

        # NOTE: STEP 1 - return cropped order board
        # the first workflow (capture-detection-coordinates) gives us:
        #  - the coordinates for cropping the order board from the video frame
        #  - the ArUco tag coordinates BASED ON THE CROPPED ORDER BOARD IMAGE for finding the rails on the order board
        # We need to capture both of these outputs in order to run the second workflow (count-sliders) which takes in the cropped order board image and uses the ArUco tag coords to find each rail and count the sliders on each rail.
        
        # region for STEP 1

        # NOTE: capture the order board coords from the workflow that detected it in the video frame
        order_board_coords = result.get("order_board_coords", [])
        if isinstance(order_board_coords, list) and order_board_coords:
            if isinstance(order_board_coords[0], list):
                flat_order_board_coords = [
                    d for group in order_board_coords for d in group if isinstance(d, dict)
                ]
            else:
                flat_order_board_coords = [d for d in order_board_coords if isinstance(d, dict)]
        else:
            flat_order_board_coords = []

        if not flat_order_board_coords:
            print("[DEBUG] result keys:", list(result.keys()) if isinstance(result, dict) else type(result))
            print("[DEBUG] order_board_coords type/value:", type(order_board_coords), order_board_coords)
            fail_path = DEBUG_FAIL_DIR / f"no_order_board_frame_{FRAME_COUNTER:06d}.jpg"
            cv2.imwrite(str(fail_path), video_frame.image)
            print(f"[DEBUG] Saved no-detection frame: {fail_path}")
            raise ValueError(f"No usable order_board_coords. Raw: {order_board_coords}")

        # NOTE: use the order board coords to manually (via python) crop the order board from the video frame, convert to data URI, so it can be passed into the slider count workflow as an image input
        crop_uri, crop_img = crop_order_board_videoframe_to_data_uri(video_frame, flat_order_board_coords)
        h, w = crop_img.shape[:2]
        _ = (h, w)  # keep for debug parity with existing flow

        #endregion


        # NOTE: Step 2 - Calc cost for each rail - pass in the cropped order board image we built and the detected ArUco tag coordinates and 
        # get back the count of sliders for each rail and calc the costs

        # region for STEP 2
        
        aruco_tag_coords = result.get("aruco_coords", [])
        if not (isinstance(aruco_tag_coords, list) and aruco_tag_coords and isinstance(aruco_tag_coords[0], list)):
            raise ValueError(f"Invalid aruco_tag_coords shape: {aruco_tag_coords}")

        # OUTPUT of aruco_tag_coords: [[{'x': 415.5572847723961, 'y': 299.48401272296906, 'width': 830.0873347520828, 'height': 593.0319745540619, 'class': 'order_board'}, {'x': 76.86630630493164, 'y': 250.95183563232422, 'width': 67.3298568725586, 'height': 56.88020324707031, 'class': 'aruco3'}, {'x': 82.40351486206055, 'y': 469.42283630371094, 'width': 67.73717498779297, 'height': 56.424896240234375, 'class': 'aruco6'}, {'x': 74.54020500183105, 'y': 176.44049835205078, 'width': 66.55055618286133, 'height': 62.64158630371094, 'class': 'aruco2'}, {'x': 72.96613502502441, 'y': 101.71524429321289, 'width': 69.20707321166992, 'height': 61.42687225341797, 'class': 'aruco1'}, {'x': 80.61264991760254, 'y': 396.167724609375, 'width': 67.64889144897461, 'height': 56.8077392578125, 'class': 'aruco5'}, {'x': 78.42835998535156, 'y': 322.7617950439453, 'width': 67.484375, 'height': 57.613983154296875, 'class': 'aruco4'}]]

        length_of_aruco_coords = len(aruco_tag_coords[0]) - 1
        if length_of_aruco_coords <= 0:
            raise ValueError("No ArUco markers available for slider counting")

        # TODO: do I bundle this in a function? 
        # NOTE: for each ArUco detection in aruco_tag_coords:
        #  - run the second workflow to count the sliders on a given rail and return as a list where list[0] is Rail1 slider count

        count_per_rail = [0] * length_of_aruco_coords

        for i in range(1, length_of_aruco_coords + 1):
            cls_name = f"aruco{i}"
            marker = next((d for d in aruco_tag_coords[0] if d.get("class") == cls_name), None)

            if not marker:
                print(f"{cls_name} not found")
                continue

            rail_x_center = 590
            ar_y_center = marker["y"]
            rail_width = 300
            ar_height = marker["height"] + 60  # Add a small buffer because ArUco markers were not as tall as the sliders

            # Call slider count workflow for the current rail
            slider_count_result = get_count_of_sliders(
                image_data_uri=crop_uri,
                x_center=rail_x_center,
                y_center=ar_y_center,
                width=rail_width,
                height=ar_height,
            )

            # Build per-rail debug frame on a fresh copy, then write one 0.5s frame to video.
            draw_x, draw_y, draw_w, draw_h = _extract_slider_box(
                slider_count_result=slider_count_result,
                fallback_x=rail_x_center,
                fallback_y=int(ar_y_center),
                fallback_w=rail_width,
                fallback_h=int(ar_height),
            )
            rail_preview = crop_img.copy()
            x1 = int(round(draw_x - draw_w / 2))
            y1 = int(round(draw_y - draw_h / 2))
            x2 = int(round(draw_x + draw_w / 2))
            y2 = int(round(draw_y + draw_h / 2))

            ph, pw = rail_preview.shape[:2]
            x1 = max(0, min(pw - 1, x1))
            y1 = max(0, min(ph - 1, y1))
            x2 = max(1, min(pw, x2))
            y2 = max(1, min(ph, y2))

            slider_count = int(slider_count_result[0]["bead_count"])  #had to leave this as 'bead count' because that's the output name in the workflow
            count_per_rail[i - 1] = slider_count
            
            rail_box_text = f"{cls_name}: {slider_count} sliders"

            cv2.rectangle(rail_preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_x = max(0, x1 - 120)
            label_y = max(20, y1 + 20)
            cv2.putText(
                rail_preview,
                rail_box_text,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            padded_preview = _center_pad_to_canvas(
                rail_preview,
                PREVIEW_CANVAS_W,
                PREVIEW_CANVAS_H,
            )
            _get_preview_writer().write(padded_preview)

        #NOTE: now multiply the slider count per row by the cost per item on that row and sum for total order cost to display

        #rail 1 = cost_per_item[0] = cheeseburgers
        #rail 2 = nachos
        #rail 3 = pickle
        #rail 4 = candy
        #rail 5 = popcorn
        #rail 6 = cookie_dough
        #rail 7 = soda
        #rail 8 = water

        total_order_cost = 0.0
        cost_per_item = [5, 4, 2, 1.5, 2, 4, 3, 2]
        for i, count in enumerate(count_per_rail):
            if i < len(cost_per_item):
                total_order_cost += count * cost_per_item[i]

        LAST_TOTAL_ORDER = total_order_cost
        CONSECUTIVE_FAILURES = 0
        _emit_total_order()
        maybe_send_total_to_phone(total_order_cost) #also send to phone via WebSocket

        #endregion

    except Exception as e:
        CONSECUTIVE_FAILURES += 1
        print("Processing failed:", e)
        _emit_total_order()
    
    if PROCESSED_FRAME_COUNTER >= MAX_PROCESSED_FRAMES:
        print(f"[DEBUG] Reached {MAX_PROCESSED_FRAMES} processed frames. Stopping pipeline...")
        _release_preview_writer()
        pipeline.terminate()

#region Build a video producer that pulls fromt the Basler camera feed to create a new video feed that can be read in InferencePipeline as the video_reference

class BaslerFrameProducer(VideoFrameProducer):
    def __init__(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        # Turn off auto exposure so your manual setting sticks
        # if hasattr(self.camera, "ExposureAuto"):
        #     self.camera.ExposureAuto.SetValue("Off")

        # Set exposure time (microseconds)
        #self.camera.ExposureTime.SetValue(50000.0)  #trying 50k to see if that gets better detections than auto, but I know 500k was definitely good (it just means 2 fps which is a bummer)
        
        #this section converts the capture to get RGB and thus capture frames in color, which should also help improve detections over the grayscale I was dealing with by default
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened and self.camera.IsGrabbing()

    def grab(self) -> bool:
        # Basler grabs in retrieve, but we can check readiness here
        return self.isOpened()

    def retrieve(self):
        if not self.isOpened():
            return False, None

        grab = self.camera.RetrieveResult(
            5000,
            pylon.TimeoutHandling_ThrowException
        )

        try:
            if grab.GrabSucceeded():
                image = self.converter.Convert(grab)
                img = image.GetArray()  # true BGR color image for OpenCV
                return True, img

            return False, None

        finally:
            grab.Release()

    def discover_source_properties(self) -> SourceProperties:
        w = self.camera.Width.Value
        h = self.camera.Height.Value
        fps = self.camera.ResultingFrameRate.Value if hasattr(self.camera, 'ResultingFrameRate') else 30.0
        return SourceProperties(width=w, height=h, total_frames=-1, is_file=False, fps=fps, is_reconnectable=True)

    def initialize_source_properties(self, properties: dict) -> None:
        for k, v in properties.items():
            if hasattr(self.camera, k):
                setattr(self.camera, k, int(v))

    def release(self):
        if self._opened:
            self.camera.StopGrabbing()
            self.camera.Close()
            self._opened = False

def basler_source_factory() -> VideoFrameProducer:
    return BaslerFrameProducer()

#endregion

#region Set up and run main Roboflow pipeline with primary workflow (capture-detection-coordinates) and sink function to run workflow B (count-sliders) and calculate order cost based on detected slider counts per rail
pipeline = InferencePipeline.init_with_workflow(
    api_key=RAK,
    workspace_name="manufacturing-n8ggq",
    workflow_id="capture-detection-coordinates",
    video_reference=basler_source_factory,  # video for testing - "basler_capture.avi",
    max_fps=2,
    on_prediction=my_sink,
)

pipeline.start()
pipeline.join()
_release_preview_writer()


#endregion