from fastapi import FastAPI
import socketio
import cv2
import numpy as np
import base64
import eventlet

# Create a FastAPI app and a Socket.IO server
app = FastAPI()
sio = socketio.Server(cors_allowed_origins="*")
sio_app = socketio.WSGIApp(sio, app)
frame = 0


@sio.event
def connect(sid, environ):
    print("Client connected:", sid)


@sio.event
def disconnect(sid):
    print("Client disconnected:", sid)


@sio.event
def video_frame(sid, data):
    # Decode the JPEG image
    jpg_original = base64.b64decode(data)
    np_array = np.frombuffer(jpg_original, np.uint8)
    np_array = np_array.reshape((480, 640, -1))
    frame = cv2.imdecode(np_array, cv2.IMREAD_ANYCOLOR)

    # Display the frame
    # cv2.imshow("Received Video", np_array)
    cv2.imwrite(f"out/frame-{frame}.jpg", np_array)


# Start the FastAPI server with Socket.IO
if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 8000)), sio_app)
