
<p align="right">
  <img src="https://github.com/user-attachments/assets/c13f4bd3-06af-4707-a587-8e756ffd6e39"
       alt="Automatez logo"
       width="100">
</p>

This provides the time a vehicle is traveling through a specified zone in an image as well as a count of vehicles crossing the zone.

A basic dashboard is also generated (with mainly manufactured data except for the changing 'time in zone') on the video.

Process:
1. references a video file (but could point to a stream instead)
2. runs a workflow out of Roboflow which has an object detection model for vehicles and tracking mechanisms
3. collects time in zone and count of vehicles and outputs to a dashboard on the video (written per frame)

<img width="1879" height="1063" alt="image" src="https://github.com/user-attachments/assets/083f2ba9-6d2a-4078-b286-624867a33277" />


