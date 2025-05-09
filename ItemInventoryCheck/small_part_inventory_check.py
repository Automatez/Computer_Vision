
# 1. cd to C:\Users\zacra\Documents\python\training\roboflow\inference
# 2. Open Docker
# 3. start the venv >.\workflow_venv\Scripts\activate.ps1
# 3.5 set your API_key as an environment variable (replace the xxx first)
# > $env:ROBOFLOW_API_KEY="xxx"
# 4. start the inference server before working on the workflow blocks in Roboflow
#  >inference server start


import os
import json
import numpy as np
import cv2 as cv
from inference_sdk import InferenceHTTPClient
import re

DEBUG = False #to run testing code flip to True

# grab the api_key from the environment variable
RAK = os.getenv("ROBOFLOW_API_KEY")
if not RAK:
    raise ValueError("ROBOFLOW_API_KEY is not set in the environment.")

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=RAK
)

# Sample JSON file containing detections (and this has code to use this for testing)
with open("InstSeg_outputOfDetections.txt", "r") as f:
    data = json.load(f)

bin = [] # to hold the bin detections and their position in the tray
grid = [] # to hold all the separate item detections and their position in the grid
            # both as list of lists like: 
            # [gridID (1t,2t... for top and 1b,2b... for bottom), 
            # topleftx, 
            # toplefty, 
            # topbottrow, 
            # quad (quadrilateral coordinates as array([],[],[],[],type=int32)),
            # item_num_or_count (item num -from ocr)]

all_y_coords_grid = [] # to hold all the y coordinates of the top left corners of the detections for finding top and bottom row break
all_y_coords_bin = [] # to hold all the y coordinates of the top left corners of the bin detections for finding top and bottom row break

# Load the image on which to plot the results
img = cv.imread("small_inventory.jpg")
if img is None:
    raise FileNotFoundError("Could not load image 'small_inventory.jpg'.")


#########################################
# GET THE GRID LABEL DETECTIONS AND OCR FOR ITEM NUMBERS
#########################################

wf_grid_result = client.run_workflow(
    workspace_name="manufacturing-n8ggq",
    workflow_id="smallpartgridfinder",
    images={
        "image": img
    },
    use_cache=True # cache workflow definition for 15 minutes
)

# Extract detections from the JSON structure
detections_grid = wf_grid_result[0]["model_predictions"]["predictions"]

displayImg = img.copy()

# Loop over each detection, reduce jagged segmentation to a computed 4-point outline
# Save it for analysis and draw it on the image
def capture_dynamic_zone(detections, boxes_to_detect, y_coords_group, gridcolor):
    for detection in detections:
        pts = detection["points"]
        # Convert list of point dictionaries into a NumPy array of shape (N, 2)
        points = np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)
        
        # Compute the convex hull of the points
        hull = cv.convexHull(points)
        
        # Compute the perimeter of the convex hull
        peri = cv.arcLength(hull, True)
        
        # Approximate the convex hull to a polygon with an accuracy factor
        approx = cv.approxPolyDP(hull, 0.02 * peri, True)
        
        # If the approximation yields 4 points, use them; otherwise, use a bounding rectangle as fallback
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
        else:
            x, y, w, h = cv.boundingRect(hull)
            quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        
        # for coord in quad:
        #     print(coord)

        # print("quad.shape:", quad.shape)
        #Save all top left corners like [[ID, topleftx, toplefty, topbottrow, position_in_row(1tox left to right), itemnum]]
        topleftx = 50000
        toplefty = 50000
        for item in quad:
            if item[0] < topleftx:
                topleftx = item[0]
            if item[1] < toplefty:
                toplefty = item[1]

        #add this item to the grid and y coord list
        boxes_to_detect.append(["gridID", topleftx, toplefty, "topbottrow", quad, "itemnumct"])
        y_coords_group.append([toplefty])

        #breakpoint()

        # Draw the quadrilateral on the image using red color and thickness of 2
        # cv.polylines requires the points in the shape (number_of_points, 1, 2)
        quad = quad.reshape((-1, 1, 2))
        cv.polylines(displayImg, [quad], isClosed=True, color=gridcolor, thickness=10)
        
        #Test line, note that 0,0 is top left of image
        # testpts = [[2400, 900], [2400,1200]]
        # pts = np.array(testpts, np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=10)

        # Optionally, you can also put the detection ID near the outline:
        # Get the top-left point of the quadrilateral for the text position
        # text_pos = tuple(quad[0][0])
        # cv.putText(img, detection["detection_id"], text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

capture_dynamic_zone(detections_grid, grid, all_y_coords_grid, (0,0,255))

# Place all detections in the top or bottom row and then sort by x value
def sort_detections(boxes_to_detect, y_coords_group):
    
     # First, extract the numeric values:
    y_values = [y[0] if isinstance(y, list) else y for y in y_coords_group]

    # Now sort the numbers:
    sorted_y = sorted(y_values)

    # Compute differences between consecutive sorted values
    gaps = [sorted_y[i+1] - sorted_y[i] for i in range(len(sorted_y)-1)]
    # Find the index of the largest gap
    max_gap_index = np.argmax(gaps)
    # Compute cutoff as the average of the values around the largest gap
    cutoff = (sorted_y[max_gap_index] + sorted_y[max_gap_index+1]) / 2.0

    # Now update the grid listing with the indicator based on the cutoff:
    for row in boxes_to_detect:
        if row[2] < cutoff:
            row[3] = "top"
        else:
            row[3] = "bottom"

    # Sort the grid by top or bottom then by the x value within each row to get the grids from left to right numbered 1 and up
    boxes_to_detect.sort(key=lambda x: (x[3], x[1]))

    # Now number the grid from left to right within each row
    for i in range(len(boxes_to_detect)):
        if i == 0:
            boxes_to_detect[i][0] = "1" + boxes_to_detect[i][3][0] #first item in the top row
        elif boxes_to_detect[i][3] == boxes_to_detect[i-1][3]: #same row as previous item
            boxes_to_detect[i][0] = str(int(boxes_to_detect[i-1][0][0]) + 1) + boxes_to_detect[i][3][0]
        else:
            boxes_to_detect[i][0] = "1" + boxes_to_detect[i][3][0] #first item in the next row

sort_detections(grid, all_y_coords_grid)

#region Now capture the item numbers from the OCR and add them to the grid by cropping and OCRing each grid item

for itemrow in grid:
    #Crop the image to the quadrilateral
    # Extract the polygon coordinates from the list (they're the 5th element)
    polygon = itemrow[4]  # This is a NumPy array of shape (4, 2)

    # Compute the bounding rectangle around the polygon
    x, y, w, h = cv.boundingRect(polygon)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # Fill the polygon area on the mask with white (255)
    cv.fillPoly(mask, [polygon], 255)

    # Crop the area using slicing
    cropped_image = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    final_cropped = cv.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

    #Run the final_cropped through OCR and get the item number from the end of the string
    
    ocr_result = client.run_workflow(
        workspace_name="manufacturing-n8ggq",
        workflow_id="ocr-for-item-number",
        images={
            "image": final_cropped
        },
        use_cache=True # cache workflow definition for 15 minutes
    )

    # ocr_result looks like this:
    # [{'model': {'result': '8-32 X318 Uses 9164 HexiNrench $ 55EA 08236 02689 3184-3',
    #                     'parent_id': 'image',
    #                     'prediction_type': 'ocr',
    #                     'root_parent_id': 'image'}}]

    # Extract the result string
    result_str = ocr_result[0]['model']['result']

    # Define the regex pattern: exactly 4 digits, a hyphen, and one alpha character.
    pattern = r'\b\d{4}-[A-Za-z0-9]\b'

    # Search for the pattern in the string
    match = re.search(pattern, result_str)

    # for troubleshooting
    # if match:
    #     print("Found:", match.group(0))
    # else:
    #     print("No match found:", result_str)

    # Define the text to write, update item num in the row, and set the position (bottom-left corner of the text)
    text = match.group(0)
    
    itemrow[5] = text

    if DEBUG:
        print(itemrow)

# endregion

    # region Put the OCR result on the image

    position = (20, 40)  # (x, y) coordinates

    # Choose a font from OpenCV's available fonts
    font = cv.FONT_HERSHEY_SIMPLEX

    # Set the font scale (size), color (B, G, R), thickness, and line type
    font_scale = 1
    color = (0, 0, 255)  # Red text
    thickness = 4
    line_type = cv.LINE_AA

    # Write the text on the image
    cv.putText(final_cropped, text, position, font, font_scale, color, thickness, line_type)

    if DEBUG:
        # Display each cropped image and OCR item num value as this loop runs
        cv.imshow("Cropped Area", final_cropped)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # endregion




#########################################
#  GET THE BIN DETECTIONS AND COUNT FOR ITEMS
#########################################


wf_bin_result = client.run_workflow(
    workspace_name="manufacturing-n8ggq",
    workflow_id="small-part-bin-finder",
    images={
        "image": img
    },
    use_cache=True # cache workflow definition for 15 minutes
)

# Extract detections from the JSON structure
detections_bin = wf_bin_result[0]["model_predictions"]["predictions"]

capture_dynamic_zone(detections_bin, bin, all_y_coords_bin, (255,0,0))

sort_detections(bin, all_y_coords_bin)

# region Now capture the item counts from each bin and add them to the bin by cropping and counting each bin item

#ver = 1 #for saving the cropped images to disk when initially training
for itemrow in bin:
    #Crop the image to the quadrilateral
    # Extract the polygon coordinates from the list (they're the 5th element)
    polygon = itemrow[4]  # This is a NumPy array of shape (4, 2)

    # Compute the bounding rectangle around the polygon
    x, y, w, h = cv.boundingRect(polygon)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # Fill the polygon area on the mask with white (255)
    cv.fillPoly(mask, [polygon], 255)

    # Crop the area using slicing
    cropped_image = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    final_cropped = cv.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

    if DEBUG:
        # Display each cropped image and OCR item num value as this loop runs
        cv.imshow("Cropped Area", final_cropped)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    #To train a model, save the cropped images to disk
    # cv.imwrite(f"cropped_area{ver}.jpg", final_cropped)
    # ver += 1

    #Run the final_cropped through COUNT detections and save that value
    
    count_result = client.run_workflow(
        workspace_name="manufacturing-n8ggq",
        workflow_id="smallpartpartscounter",
        images={
            "image": final_cropped
        },
        use_cache=True # cache workflow definition for 15 minutes
    )

    itemrow[5] = count_result[0]["property_definition"]

    #Send a warning if under a given threshold (could pull this from a db to be specific to each part num)
    threshold = 5
    if (itemrow[5] < threshold):
        grid_itemnum = next((sublist[5] for sublist in grid if sublist[0] == itemrow[0]), None)
        print(f'Need to reorder part {grid_itemnum} as there are only {itemrow[5]} left in bin.')


# endregion

# Display the annotated image so user can confirm all of grid was identified (should see red bounding poly around each one)
cv.namedWindow("Annotated Image", cv.WINDOW_NORMAL)
cv.imshow("Annotated Image", displayImg)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the annotated image
cv.imwrite("annotated_inventory.jpg", displayImg)
