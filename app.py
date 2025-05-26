import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, render_template
from flask_sock import Sock
import numpy as np
import json
from model.keypoint_classifier import KeyPointClassifier

app = Flask(__name__)
sock = Sock(app)
app.config['SECRET_KEY'] = 'secret!'

# Initialize the classifier
classifier = KeyPointClassifier('model/keypoint_classifier.tflite')

# Track connected clients (you can use set or dict for advanced tracking)
connected_clients = set()

@app.route("/")
def home():
    return render_template("index.html")

@sock.route('/ws')
def websocket_handler(ws):
    print("Client connected")
    connected_clients.add(ws)

    try:
        while True:
            message = ws.receive()
            if message is None:
                break  # client disconnected

            try:
                data = json.loads(message)

                if data.get("type") == "join":
                    username = data.get("username", "Unknown")
                    print(f"{username} joined.")
                    ws.send(json.dumps({
                        'type': 'join_ack',
                        'message': f'Welcome {username}!',
                        'status': 'success'
                    }))
                    continue

                if data.get("type") == "gesture":
                    landmarks = data.get("landmarks")
                    if landmarks is None:
                        ws.send(json.dumps({'error': 'Missing landmark data'}))
                        continue

                    result = classifier(landmarks)

                    if isinstance(result, (np.integer, np.floating)):
                        result = result.item()
                    elif isinstance(result, np.ndarray):
                        result = result.tolist()

                    payload = json.dumps({'prediction': result})
                    for client in connected_clients.copy():
                        try:
                            client.send(payload)
                        except Exception:
                            connected_clients.discard(client)

            except Exception as e:
                ws.send(json.dumps({'error': f'Error: {str(e)}'}))

    finally:
        print("Client disconnected")
        connected_clients.discard(ws)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)