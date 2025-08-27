# This uses a video of a single lane bridge to do the following: (livestream code from https://blog.roboflow.com/video-stream-analysis/)
# - uses an off-the-shelf vehicle object detection to find vehicles
# - uses a time-in-zone feature to determine if traffic gets stopped on the bridge
# - builds a dictionary of tracking IDs that could be used for a count of vehicles that crossed the bridge
# - overlays the time-in-zone and a simple dashboard on the video output (note - dashboard is a mock-up just for demo purposes)

# PROCESS to use a video in this folder
# 
# 1. should have done cd to get to the folder, so now activate the venv (which was created with Python 3.11 and has pkgs listed in requirements.txt))
#     confirm >python --version #should be 3.11 
# 
# 2. Verify whether Docker is running an inference server.
#    If server is NOT running restart an old one.  If you need a new one run the below code
#       > inference server start   #check if it's done by checking that Docker shows a running container
# 3. set your API_key as an environment variable (replace the xxx first)
#       > $env:ROBOFLOW_API_KEY="xxx"

# 4. Update your code below to use the local video file
#   video_reference="bridge.mkv",     #this pulls from my mediamtx 8554 local RTSP server which pulls from Streamlink 1935/youtube which is an HTTP>RTMP converted stream of a YouTube channel as specified in the streamlink I started from a command line

# 5. run this python file
#    >python Vehicle_timeInZone_count.py



import os
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import signal
import sys
import threading
import cv2
import time


# Set the ONNX Runtime execution providers just to eliminate warnings at runtime
#os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = '[CUDAExecutionProvider]'  # this is now set in the powershell before starting the container
os.environ["ORT_LOG_LEVEL"] = "WARNING" #"VERBOSE"

# grab the api_key from the environment variable 
RAK = os.getenv("ROBOFLOW_API_KEY") 
if not RAK: 
    raise ValueError("ROBOFLOW_API_KEY is not set in the environment.") 

_frame_counter = {"n": 0}          # dict keeps it mutable without global keyword
PRINT_EVERY = 60                   # change to whatever cadence you want

#1920 x 1080 is the video size
object_times = {}       # track_id -> {last_time, in_zone_status}
car_count_today = 147.2        # just a starting number for demo purposes
car_count_average = 96.3  # just a starting number for demo purposes


def my_sink(result, video_frame):
    global car_count_today
    global car_count_average

    # region capture and store cumulative time in zone for each object id and calc avg
    frame_data = result['time_in_zone']
    if frame_data is None:
        return  # nothing to do this frame
    
    if frame_data.tracker_id is None:
        return
    else:
        tid = frame_data.tracker_id
    
    if frame_data.data['time_in_zone'] is None:
        return  # nothing to do this frame
    else:
        tiz = frame_data.data['time_in_zone']  # array of time in zone for each detected object in this frame
    
    #capture all times in zone
    if len(tiz) > 0:
        for i in range(len(tiz)):
            #save ids with time in zone but overwrite if you get a more recent one
            tid_i = int(tid[i])
            tiz_i = float(tiz[i])
            object_times[tid_i] = tiz_i # the id of detected object and the time in zone (seconds) it was in the zone at this point, could add a check to only replace if the latest time is greater but I'll assume the tracking works
            avg_time = sum(object_times.values()) / len(object_times)
    
    # endregion

    if result.get("label_visualization"): # Display an image from the workflow response
        img = result["label_visualization"].numpy_image   # draw on workflow viz
        anchory = 650
        # --- overlay live feed for text of average time (before resizing) ---
        text = f"Bridge Crossing Time"
        org  = (130, anchory + 0)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)   # outline
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 1, cv2.LINE_AA)
        
        text = f"Today's Rate: {avg_time:.2f}s"
        org  = (150, anchory + 30)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)   # outline
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 1, cv2.LINE_AA)

        # --- temp dashboard for example purposes ---
        text = f"Average Rate: 2.9s"
        org  = (150, anchory + 60)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)   # outline
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 1, cv2.LINE_AA)

        text = f"Traffic Volume (for this time of day)"
        org  = (130, anchory + 100)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)   # outline
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 1, cv2.LINE_AA)

        text = f"Today's Rate: {car_count_today:.1f} cars"
        org  = (150, anchory + 130)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)   # outline
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 1, cv2.LINE_AA)
        car_count_today += 0.005  # just for demo purposes, would be incremented by line crossing logic in a full workflow
        
        text = f"Average Rate: {car_count_average:.1f} cars"
        org  = (150, anchory + 160)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)   # outline
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 1, cv2.LINE_AA)
        car_count_average += 0.002  # just for demo purposes, would be incremented by line crossing logic in a full workflow
        
        # end dashboard overlay

        #img = result["label_visualization"].numpy_image
        disp = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # 50% size
        cv2.imshow("Workflow Image", disp)
        cv2.waitKey(1)
        #cv2.imshow("Workflow Image", result["label_visualization"].numpy_image)
        #cv2.waitKey(1)
    
    #print('my_sink says')
    _frame_counter["n"] += 1
    if _frame_counter["n"] % PRINT_EVERY == 0:   # only on every Nth call
        print(f"[frame {_frame_counter['n']}]")
        print(result)
        printable_tiz = result['time_in_zone'].data['time_in_zone']
        my_check = f'TIZ: {printable_tiz}'
        print(my_check)



shutdown_event = threading.Event()

def graceful_exit(signum=None, frame=None):
    if shutdown_event.is_set():
        return
    shutdown_event.set()
    print("\nGracefully shutting down...")
    print(len(object_times))
    try:
        pipeline.terminate()   # sets _stop and terminates video sources
        pipeline.join()        # waits for inference & dispatch to finish; triggers on_pipeline_end (executor shutdown)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    # Now it is safe to end the program; no need to sys.exit here—let main return.

signal.signal(signal.SIGINT, graceful_exit)   # Handle CTRL+C
signal.signal(signal.SIGTERM, graceful_exit)  # Handle kill signals

# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key=RAK,
    workspace_name="manufacturing-n8ggq",
    workflow_id="keypointzoneonly", # for just checking zone logic, and note it's actual object det, not keypoint
    video_reference="bridge.mkv", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start(use_main_thread=False)

# Keep the main thread alive until we’re asked to stop
try:
    while not shutdown_event.is_set():
        time.sleep(0.2)
except KeyboardInterrupt:
    graceful_exit()

# 5) Optional: if you want a clean fallthrough when the video ends naturally
if not shutdown_event.is_set():
    graceful_exit()



# =======================================
#
#        DATA SAMPLES
#
# =======================================
# result data example:
# [frame 780]
# {'bounding_box_visualization': <inference.core.workflows.execution_engine.entities.base.WorkflowImageData object at 0x0000020C37DBB050>, 
# 'label_visualization': <inference.core.workflows.execution_engine.entities.base.WorkflowImageData object at 0x0000020C37E49650>, 
# 'time_in_zone': 
# 	Detections(
# 		xyxy=array([[ 999.,  554., 1195.,  668.], [ 575.,  372.,  787.,  507.]]), 
# 		mask=None, 
# 		confidence=array([0.71180147, 0.44315791]), 
# 		class_id=array([1, 6]), 
# 		tracker_id=array([30, 23]), 
# 		data={
# 			'time_in_zone': array([0.15      , 2.18333333]), 
# 			'root_parent_dimensions': array([[1080, 1920],[1080, 1920]]), 
# 			'parent_coordinates': array([[0, 0], [0, 0]]), 
# 			'inference_id': array(['9592fe52-7475-4088-af19-293d3685286f','9592fe52-7475-4088-af19-293d3685286f'], dtype='<U36'), 
# 			'class_name': array(['car', 'van'], dtype='<U3'), 
# 			'root_parent_id': array(['image.[0]', 'image.[0]'], dtype='<U9'), 
# 			'parent_dimensions': array([[1080, 1920],[1080, 1920]]), 
# 			'detection_id': array(['9b62309a-f786-46ea-a947-ee8315d61500', 'ed9eeb91-1cb8-413f-8a04-49564f582998'], dtype='<U36'), 
# 			'root_parent_coordinates': array([[0, 0], [0, 0]]), 
# 			'parent_id': array(['image.[0]', 'image.[0]'], dtype='<U9'), 
# 			'prediction_type': array(['object-detection', 'object-detection'], dtype='<U16'), 
# 			'image_dimensions': array([[1080, 1920], [1080, 1920]])
# 		}, 
# 		metadata={}
# 	), 
# 'model_1_predictions': 
# 	Detections(
# 		xyxy=array([[ 124.,  340.,  358.,  435.], [ 999.,  554., 1195.,  668.], [ 575.,  372.,  787.,  507.]]), 
# 		mask=None, 
# 		confidence=array([0.81376302, 0.71180147, 0.44315791]), 
# 		class_id=array([1, 1, 6]), 
# 		tracker_id=None, 
# 		data={
# 			'class_name': array(['car', 'car', 'van'], dtype='<U3'), 
# 			'detection_id': array(['bf1399d9-6f81-42d9-8605-72a3505c695c', '9b62309a-f786-46ea-a947-ee8315d61500', 'ed9eeb91-1cb8-413f-8a04-49564f582998'], dtype='<U36'), 
# 			'parent_id': array(['image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'), 
# 			'image_dimensions': array([[1080, 1920], [1080, 1920], [1080, 1920]]), 
# 			'inference_id': array(['9592fe52-7475-4088-af19-293d3685286f', '9592fe52-7475-4088-af19-293d3685286f', '9592fe52-7475-4088-af19-293d3685286f'], dtype='<U36'), 
# 			'prediction_type': array(['object-detection', 'object-detection', 'object-detection'], dtype='<U16'), 
# 			'root_parent_id': array(['image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'), 
# 			'root_parent_coordinates': array([[0, 0], [0, 0], [0, 0]]), 
# 			'root_parent_dimensions': array([[1080, 1920], [1080, 1920], [1080, 1920]]), 
# 			'parent_coordinates': array([[0, 0], [0, 0], [0, 0]]), 
# 			'parent_dimensions': array([[1080, 1920], [1080, 1920], [1080, 1920]])
# 		}, 
# 		metadata={}
# 	)
# }