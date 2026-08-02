# SmartStandz

SmartStandz is a computer vision proof of concept that automatically recognizes concession stand items and displays the running order total on a phone.

The project combines a camera, an object detection model, Python, WebSockets, and a simple Android display application to demonstrate how a traditional concession stand could be made faster and easier to operate.

## How It Works

1. A Basler camera captures the concession stand area.
2. A Python application processes the camera feed.
3. An RF-DETR object detection model identifies the selected items.
4. Python calculates the current order total.
5. The total is sent over a WebSocket connection.
6. An Android phone displays the updated amount.

## Demo

SmartStandz was built as a working minimum viable product to demonstrate an end-to-end computer vision system.

The current version focuses on:

* Detecting concession stand items
* Calculating a running total
* Sending results from Python to an Android phone
* Updating the phone display in near real time

## Technology

* **Python** — camera processing, model inference, pricing logic, and WebSocket server
* **Roboflow RF-DETR** — custom object detection model
* **Basler Camera** — image capture
* **WebSockets** — communication between the Python application and phone
* **Kotlin / Android Studio** — Android display application
* **NVIDIA Jetson** — edge inference testing and deployment

## System Architecture

```text
Basler Camera
      |
      v
Python Application
      |
      +--> RF-DETR Object Detection
      |
      +--> Pricing Logic
      |
      v
WebSocket Server
      |
      v
Android Phone Display
```

## Repository Overview

```text
SmartStandz/
├── server.py                 # Python WebSocket server
├── requirements.txt          # Python dependencies
├── android/                  # Android Studio application
├── models/                   # Model files or model configuration
├── assets/                   # Images, screenshots, and demo media
└── README.md
```

The exact folder structure may vary depending on the current version of the project.

## Getting Started

### Prerequisites

You will need:

* Python 3.10 or newer
* A trained object detection model
* A compatible camera
* An Android device or emulator
* Android Studio
* Both devices connected to the same local network

### Install Python Dependencies

Clone the repository:

```bash
git clone https://github.com/Automatez/smartstandz.git
cd smartstandz
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Python Server

Start the WebSocket server:

```powershell
python server.py
```

The default WebSocket port is:

```text
8765
```

The server should listen on the computer’s local network address so the Android phone can connect to it.

## Configuring the Android App

In the Android project, locate the WebSocket server address in `MainActivity.kt`.

Update it to match the local IP address of the computer running `server.py`:

```kotlin
ws://192.168.x.x:8765
```

For example:

```kotlin
ws://192.168.68.85:8765
```

The computer and phone must be connected to the same network.

To find the computer’s local IP address on Windows, run:

```powershell
ipconfig
```

Look for the **IPv4 Address** under the active Wi-Fi or Ethernet connection.

## Current Limitations

SmartStandz is an MVP and is not intended for production use.

Current limitations include:

* The computer’s local IP address may change
* Detection accuracy depends on lighting and camera position
* The phone and computer must be on the same local network
* Pricing information is currently configured in the application
* The system has not yet been integrated with a payment processor or point-of-sale platform
* Performance may vary depending on the inference hardware

## Potential Improvements

Future versions could include:

* Automatic discovery of the WebSocket server
* A fixed hostname instead of a changing IP address
* Improved inference speed and reduced display latency
* Support for additional concession items
* A configurable product and pricing database
* Order confirmation and correction controls
* Integration with a point-of-sale system
* Payment processing
* Inventory tracking
* Multiple display devices
* Improved low-light detection
* Cloud or remote monitoring

## Project Purpose

SmartStandz was created as a portfolio project to demonstrate how Python and computer vision can connect physical hardware, machine learning, networking, and a mobile application into a complete working system.

The project emphasizes practical system integration rather than model training alone.

## Author

**Zac Crane**

* GitHub: [Automatez](https://github.com/Automatez)
* LinkedIn: [Zachary Crane](https://www.linkedin.com/in/zachary-crane-automatez)

## Acknowledgments

SmartStandz was built using tools and technology from:

* [Python](https://www.python.org/)
* [Roboflow](https://roboflow.com/)
* [Basler](https://www.baslerweb.com/)
* [Android Studio](https://developer.android.com/studio)
* [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing)



<img width="1618" height="972" alt="SmartStandz_flowchart" src="https://github.com/user-attachments/assets/6b27ee82-6fe6-4a9e-8002-f17609ec0fec" />

