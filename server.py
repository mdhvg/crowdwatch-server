from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
device = "cpu"
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
videos = None

maps = {
    "LT": {
        "path": "lt.png",
        "videos": [
            {
                "path": "lt.mp4",
                "xFlip": True,
                "yFlip": False,
                "startRatio": [493 / 911.0, 330 / 634.0],
                "diffRatio": [247 / 911.0, 219 / 634.0],
            },
            {
                "path": "lt2.mp4",
                "xFlip": True,
                "yFlip": False,
                "startRatio": [267 / 911.0, 361 / 634.0],
                "diffRatio": [153 / 911.0, 193 / 634.0],
            },
        ],
    },
    "Jaggi": {
        "path": "Jaggi.jpg",
        "videos": [
            {
                "path": "jaggi.mp4",
                "xFlip": False,
                "yFlip": True,
                "startRatio": [294 / 694.0, 294 / 936.0],
                "diffRatio": [245 / 694.0, 234 / 936.0],
            }
        ],
    },
}

currentMap = "Jaggi"

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

        det = det[:, :4]
        headcoords = (det[:, :2] + det[:, 2:]) / 2
        # put circle at headcoords
        for x, y in headcoords:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.imwrite("static/output.jpg", frame)
        headcoords[:, 0] /= shape[1]
        headcoords[:, 1] /= shape[0]
        headcoords[:, 1] = 1 - headcoords[:, 1]


@sio.event
def message(sid, data):
    print(f"Message from {sid}: {data}")


@sio.event
async def get_headcoords(sid):
    global videos, dt, model, device, conf_thres, iou_thres, classes, agnostic_nms, max_det, seen

    if videos is None:
        videos = [
            cv2.VideoCapture(os.path.join("videos", v["path"]))
            for v in maps[currentMap]["videos"]
        ]

    res = []
    headcoords = None

    for i, v in enumerate(videos):
        _, frame = v.read()

        shape = frame.shape

        with dt[0]:
            # Apply transformations to frame
            im0 = [frame]
            im = np.stack(
                [
                    letterbox(x, imgsz, stride=model.stride, auto=model.pt)[0]
                    for x in im0
                ]
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

            det = det[:, :4]
            headcoords = (det[:, :2] + det[:, 2:]) / 2
            # put circle at headcoords
            for x, y in headcoords:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            headcoords[:, 0] /= shape[1]
            headcoords[:, 1] /= shape[0]
            headcoords[:, 1] = 1 - headcoords[:, 1]

        res.append(
            {
                "startRatio": maps[currentMap]["videos"][i]["startRatio"],
                "diffRatio": maps[currentMap]["videos"][i]["diffRatio"],
                "headCoords": headcoords.tolist(),
                "xFlip": maps[currentMap]["videos"][i]["xFlip"],
                "yFlip": maps[currentMap]["videos"][i]["yFlip"],
            }
            if headcoords is not None
            else {
                "startRatio": maps[currentMap]["videos"][i]["startRatio"],
                "diffRatio": maps[currentMap]["videos"][i]["diffRatio"],
                "headCoords": [],
                "xFlip": maps[currentMap]["videos"][i]["xFlip"],
                "yFlip": maps[currentMap]["videos"][i]["yFlip"],
            }
        )
        headcoords = None

    await sio.emit(
        "headCoords",
        json.dumps(res),
        room=sid,
    )


@app.get("/map")
def get_map() -> dict:
    return FileResponse(os.path.join("static", maps[currentMap]["path"]))


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
