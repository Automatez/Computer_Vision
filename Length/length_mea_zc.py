# This finds the length based on largest distance between 2 points of an instance segmentation
# finds the width as the maximum perpindicular distance to the length
# also put the length on the image at the middle of the length bar
# INPUT - the results of an instance segmentation model with a reference object
# OUTPUT - the length and width of the objects in your units, and the length and width lines on the image

# 1. set your API_key (manufacturing) as an environment variable (replace the xxx first)
#    >$env:ROBOFLOW_API_KEY="xxx"
# 2. Open Docker
# 3. navigate to this directory:  C:\Users\zacra\Documents\python\training\roboflow\inference\workflow_size
# 4. start the venv >.\size_venv\Scripts\activate.ps1
# 5. start the inference server before using workflow blocks or inference in Roboflow
#    >inference server start


import cv2
import os
import inference
import numpy as np
import json
from scipy.spatial.distance import pdist, squareform
from math import radians, sin, acos

# Sample image (replace with your own)
img = cv2.imread("spores.jpg")

#Manual version Load the sample JSON file
# with open("img_segm.json", "r") as f:
#     data = json.load(f)

#Get the length of the reference object from the user, units are irrelevant as long as they are consistent

user_confidence_threshold = 0.152
class_to_measure = 'spores'
class_of_reference = 'ten_micrometers'
ref_length = 10

# user_confidence_threshold = float(input("Enter the confidence level (0.0 - 1.0): "))
# class_to_measure = input("Enter the class to measure (e.g., spores): ")
# class_of_reference = input("Enter the class of the reference object (e.g., ten_micrometers): ")
# ref_length = float(input("Enter the length of the reference object: "))

# grab the api_key from the environment variable
RAK = os.getenv("ROBOFLOW_API_KEY")
if not RAK:
    raise ValueError("ROBOFLOW_API_KEY is not set in the environment.")

#Run the instance segmentation model on the image and save the json to a variable
model = inference.get_model("measure_with_reference_private/8")  
results = model.infer(image=img, confidence=user_confidence_threshold)

#Find the reference object and get the pixel length of it and calc the scale factor
#all_predictions = results[0].predictions[0].class_name
reference_detection_count = 0
print("List of predictions for quick reference:")
for response in results:                      # one image / frame
    for pred in response.predictions:           # one detection in that image
        print("   ",pred.class_name, " with confidence ", pred.confidence)
        if pred.class_name == class_of_reference:
            #print("Ref Class at", pred.x, pred.y, "size =", pred.width, pred.height, "confidence =", pred.confidence)
            pixels_per_unit_of_length = pred.width / ref_length
            print(f"Pixels per a unit of length: {pixels_per_unit_of_length:.2f}")
            reference_detection_count += 1

if reference_detection_count == 0:
    raise ValueError(f"No reference object of class '{class_of_reference}' found in the image.")  
elif reference_detection_count > 1:
    raise ValueError(f"Multiple reference objects of class '{class_of_reference}' found in the image. Please ensure only one is present.")

#breakpoint()



#MANUAL FOR INITIAL TESTING
# # Get the first prediction with class "spores"
# predictions = data[0]["predictions"]["predictions"]
# spores_predictions = [p for p in predictions if p["class"] == "spores"]

# # Get the first set of points
# first_spore_points = spores_predictions[0]["points"]

# # Convert to numpy array in the required shape (N, 1, 2)
# points_np = np.array([[p["x"], p["y"]] for p in first_spore_points], dtype=np.int32).reshape(-1, 1, 2)

# #print(points_np.shape)  # Just to verify shape is correct



##Example of points from the contour in the format almost ready for the cv function
# points = np.array([
#     [100, 200],
#     [120, 280],
#     [180, 250],
#     [160, 180]
# ], dtype=np.int32)


def object_length_from_contour(contour_points):
    """
    Given a set of (x, y) points representing a contour, 
    returns the maximum Euclidean distance between any two points,
    and the pair of points that gave that distance.
    
    Args:
        contour_points (ndarray): Nx2 array of (x, y) coordinates
    
    Returns:
        tuple: (max_distance: float, point1: (x, y), point2: (x, y))
    """
    # Compute pairwise distances
    dist_matrix = squareform(pdist(contour_points, metric='euclidean'))

    # Find indices of the max distance in the matrix
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

    # Get the points
    pt1 = tuple(contour_points[i])
    pt2 = tuple(contour_points[j])
    max_dist = dist_matrix[i, j]

    return max_dist, pt1, pt2

def object_width_from_contour(contour_points, point1, point2):
    # length endpoints already found
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)

    L   = p2 - p1                       # length vector
    u_L = L / np.linalg.norm(L)         # unit length vector

    # one exact perpendicular unit vector (rotate 90°)
    u_W = np.array([-u_L[1], u_L[0]])   # width direction


    #Find the width of the object
    proj = contour_points @ u_W         # dot product with width axis
    imin = proj.argmin()
    imax = proj.argmax()

    width_pt1 = tuple(contour_points[imin])
    width_pt2 = tuple(contour_points[imax])
    width     = np.linalg.norm(contour_points[imax] - contour_points[imin])

    return width, width_pt1, width_pt2 # tuple(w1), tuple(w2)

spore_detection_count = 0
#Loop through the predictions to label each spore
for response in results:                      # one image / frame
    for pred in response.predictions:           # one detection in that image
        if pred.class_name == "spores":
            spore_points = pred.points
            points_np = np.array([[p.x, p.y] for p in spore_points], dtype=np.int32).reshape(-1, 1, 2)
            length, point1, point2 = object_length_from_contour(points_np.reshape(-1, 2))
            width, width_pt1, width_pt2 = object_width_from_contour(points_np.reshape(-1, 2), point1, point2)
            print(f"Object length: {length:.2f}")
            print(f"Farthest points: {point1} and {point2}")
            print(f"Object width: {width:.2f}")
            print(f"Width points: {width_pt1} and {width_pt2}")

            real_length = length / pixels_per_unit_of_length
            real_width = width / pixels_per_unit_of_length
            # Draw the length line
            cv2.line(img, point1, point2, color=(255, 0, 0), thickness=2)
            cv2.line(img, width_pt1, width_pt2, color=(0, 255, 0), thickness=2)
            cv2.putText(img, f"#{spore_detection_count} len {real_length:.2f}", (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            spore_detection_count += 1
        else:
            # optional: ignore or log other classes
            pass



# Show the image
cv2.imshow("Length in reference units (with width indicator)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
