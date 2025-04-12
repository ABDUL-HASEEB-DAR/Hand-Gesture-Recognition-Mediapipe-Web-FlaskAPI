# MediaPipe hand gesture recognition on Web using Flask API

This project uses MediaPipe for hand gesture recognition and Flask for backend processing. It controls a robot by recognizing gestures from a camera feed.

## Installation

### Frontend and Backend
The frontend for this project is contained in the `index.html` file. This file handles the camera feed, integrates with MediaPipe for hand gesture detection, and displays the recognized gesture. You can open this file in your browser to interact with the hand gesture recognition system.

For the full content of the frontend, refer to the `index.html` file in the project directory. 
For backend follow steps below.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. Install modules
   ```bash
   pip install Flask flask-cors tensorflow
   
3. Run server
   ```bash
   python app.py
  The backend server will start at http://127.0.0.1:5000.

## Available Gestures

The following gestures are recognized by the system:

- Open
- Close
- Pointer
- OK
- Peace
  
