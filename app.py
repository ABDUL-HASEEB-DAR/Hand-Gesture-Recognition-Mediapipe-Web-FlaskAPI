import eventlet
eventlet.monkey_patch()
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
from model.keypoint_classifier import KeyPointClassifier

from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the classifier
classifier = KeyPointClassifier('model/keypoint_classifier.tflite')

# Socket.IO user tracking
connected_users = {}  # sid: username mapping

@app.route("/")
def home():
    return render_template("index.html")  # serves your web app

# Socket.IO event handlers
@socketio.on('connect')
def on_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    username = connected_users.pop(request.sid, 'Unknown')
    print(f"{username} disconnected")

@socketio.on('join')
def on_join(data):
    username = data['username']
    connected_users[request.sid] = username
    print(f"User {username} has joined. Current users: {connected_users}")


# New event for gesture recognition via Socket.IO
@socketio.on('gesture_data')
def handle_gesture(data):
    try:
        landmarks = data.get('landmarks', None)
        
        
        if landmarks is None or not isinstance(landmarks, list):
            emit('gesture_result', {'error': 'Invalid landmark data'})
            return
            
        result = classifier(landmarks)
        
        if isinstance(result, (np.integer, np.floating)):
            result = result.item()
        elif isinstance(result, np.ndarray):
            result = result.tolist()
            
        # Broadcast the gesture to all users
        username = connected_users.get(request.sid, 'Unknown')
        print('gesture detected and result is: {result}')
        emit('gesture_result', {
            'user': username,
            'prediction': result
        }, broadcast=True)
        
    except Exception as e:
        emit('gesture_result', {'error': f'Classifier error: {str(e)}'})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)