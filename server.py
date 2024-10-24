from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from socketio import ASGIApp, AsyncServer
import uvicorn
import base64

import cv2
import numpy as np
import torch

from pathlib import Path
import os
import sys
import json
import time

from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

app = FastAPI()
device = "cuda:0"
weights = ROOT / "best.pt"
imgsz = (640, 640)
seen, windows, dt = 0, [], None
bs = 1  # batch size
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
classes = None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS

headcoords: torch.Tensor = torch.tensor([])

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a Socket.IO server
sio = AsyncServer(async_mode="asgi")


# Define a simple event handler
@sio.event
def connect(sid, environ):
    print(f"Client {sid} connected")


@sio.event
def disconnect(sid):
    print(f"Client {sid} disconnected")


@sio.event
def video_frame(sid, data):
    # Decode the JPEG image
    decoded_array = base64.b64decode(data)
    np_array = np.frombuffer(decoded_array, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    shape = frame.shape

    global dt, model, device, conf_thres, iou_thres, classes, agnostic_nms, max_det, seen, headcoords

    with dt[0]:
        # Apply transformations to frame
        im0 = [frame]
        im = np.stack(
            [letterbox(x, imgsz, stride=model.stride, auto=model.pt)[0] for x in im0]
        )  # resize
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

    # Inference
    with dt[1]:
        pred = model(im)

    # NMS
    with dt[2]:
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )

    # Process predictions
    det = pred[0]
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()
        cv2.imwrite("static/output.jpg", frame)

        det = det[:, :4]
        headcoords = (det[:, :2] + det[:, 2:]) / 2
        print(shape[1], shape[0])
        headcoords[:, 0] /= shape[1]
        headcoords[:, 1] /= shape[0]


@sio.event
def message(sid, data):
    print(f"Message from {sid}: {data}")


@sio.event
async def get_headcoords(sid):
    await sio.emit(
        "headcoords",
        json.dumps(
            {
                "startRatio": [0.68715697, 0.585173502],
                "endRatio": [0.81778265, 0.681388013],
                "diffRatio": [0.13062568, 0.096214511],
                "headcoords": headcoords.tolist(),
            }
        ),
        room=sid,
    )


@app.get("/headcoords")
def get_headcoords() -> dict:
    return {
        "startRatio": [0.68715697, 0.585173502],
        "endRatio": [0.81778265, 0.681388013],
        "diffRatio": [0.13062568, 0.096214511],
        "headcoords": headcoords.tolist(),
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")


def load_model() -> None:
    global model, device, imgsz, seen, windows, dt
    device = select_device(device)
    model = DetectMultiBackend(weights, device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = (
        0,
        [],
        (Profile(device=device), Profile(device=device), Profile(device=device)),
    )


# Wrap the FastAPI app with the Socket.IO WSGI app
app = ASGIApp(sio, app)

# Run the app with `uvicorn` if this script is executed directly
if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
