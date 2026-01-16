#!/usr/bin/env python3
"""
Simple YOLO inference script using the ultralytics package.
Supports images, a directory of images, video files, or webcam (use 0,1,...).
Saves annotated outputs to ./runs/detect by default.
"""

# HOW TO USE:
"""
Run on an image:
python detect.py --model yolov8n.pt --source path/to/image.jpg
Run on a folder:
python detect.py --source path/to/images_dir
Webcam:
python detect.py --source 0
Video:
python detect.py --source demo.mp4
To use GPU: pass --device 0 or --device cuda:0
What this gives you

Annotated images/videos saved in runs/detect
Class names and confidences drawn on boxes
Easy switch of model weights (yolov8n, yolov8s, custom weights)
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def draw_boxes(img, boxes, scores, classes, names, conf_thres=0.25):
    for (xyxy, conf, cls) in zip(boxes, scores, classes):
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_image_files(model, paths, out_dir, conf, device):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Could not read {p}, skipping.")
            continue
        results = model(img, device=device, conf=conf)  # list-like Results
        # Use first result
        res = results[0]
        if hasattr(res, "boxes"):
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.array([])
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
            classes = res.boxes.cls.cpu().numpy() if len(res.boxes) else np.array([])
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        draw_boxes(img, boxes, scores, classes, model.names, conf_thres=conf)
        out_path = out_dir / f"{Path(p).stem}_det{Path(p).suffix}"
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path}")


def process_video_or_cam(model, source, out_dir, conf, device):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, device=device, conf=conf)
        res = results[0]
        if hasattr(res, "boxes"):
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.array([])
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
            classes = res.boxes.cls.cpu().numpy() if len(res.boxes) else np.array([])
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        draw_boxes(frame, boxes, scores, classes, model.names, conf_thres=conf)
        writer.write(frame)
        cv2.imshow("YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved video to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO image/video/webcam detection (ultralytics)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="path or model name (yolov8n.pt/yolov8s.pt etc.)")
    parser.add_argument("--source", type=str, default="0", help="image, dir, video file, or webcam (0)")
    parser.add_argument("--out", type=str, default="runs/detect", help="output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--device", type=str, default="", help="'' for CPU, '0' for GPU 0, 'cpu' or 'cuda:0'")
    args = parser.parse_args()

    model = YOLO(args.model)
    source = args.source
    # detect whether source is webcam index
    is_numeric = False
    try:
        int(source)
        is_numeric = True
    except Exception:
        is_numeric = False

    if is_numeric or source.startswith(("rtsp://", "http://", "https://")) or source.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        process_video_or_cam(model, int(source) if is_numeric else source, args.out, args.conf, args.device)
    else:
        # single image or directory
        p = Path(source)
        if p.is_dir():
            files = [str(x) for x in p.glob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        elif p.is_file():
            files = [str(p)]
        else:
            # try glob wildcard
            import glob
            files = glob.glob(source)
        if not files:
            print("No images found for source:", source)
            return
        process_image_files(model, files, args.out, args.conf, args.device)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Simple YOLO inference script using the ultralytics package.
Supports images, a directory of images, video files, or webcam (use 0,1,...).
Saves annotated outputs to ./runs/detect by default.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def draw_boxes(img, boxes, scores, classes, names, conf_thres=0.25):
    for (xyxy, conf, cls) in zip(boxes, scores, classes):
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_image_files(model, paths, out_dir, conf, device):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Could not read {p}, skipping.")
            continue
        results = model(img, device=device, conf=conf)  # list-like Results
        # Use first result
        res = results[0]
        if hasattr(res, "boxes"):
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.array([])
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
            classes = res.boxes.cls.cpu().numpy() if len(res.boxes) else np.array([])
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        draw_boxes(img, boxes, scores, classes, model.names, conf_thres=conf)
        out_path = out_dir / f"{Path(p).stem}_det{Path(p).suffix}"
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path}")


def process_video_or_cam(model, source, out_dir, conf, device):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, device=device, conf=conf)
        res = results[0]
        if hasattr(res, "boxes"):
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.array([])
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
            classes = res.boxes.cls.cpu().numpy() if len(res.boxes) else np.array([])
        else:
            boxes, scores, classes = np.array([]), np.array([]), np.array([])

        draw_boxes(frame, boxes, scores, classes, model.names, conf_thres=conf)
        writer.write(frame)
        cv2.imshow("YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved video to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO image/video/webcam detection (ultralytics)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="path or model name (yolov8n.pt/yolov8s.pt etc.)")
    parser.add_argument("--source", type=str, default="0", help="image, dir, video file, or webcam (0)")
    parser.add_argument("--out", type=str, default="runs/detect", help="output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--device", type=str, default="", help="'' for CPU, '0' for GPU 0, 'cpu' or 'cuda:0'")
    args = parser.parse_args()

    model = YOLO(args.model)
    source = args.source
    # detect whether source is webcam index
    is_numeric = False
    try:
        int(source)
        is_numeric = True
    except Exception:
        is_numeric = False

    if is_numeric or source.startswith(("rtsp://", "http://", "https://")) or source.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        process_video_or_cam(model, int(source) if is_numeric else source, args.out, args.conf, args.device)
    else:
        # single image or directory
        p = Path(source)
        if p.is_dir():
            files = [str(x) for x in p.glob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        elif p.is_file():
            files = [str(p)]
        else:
            # try glob wildcard
            import glob
            files = glob.glob(source)
        if not files:
            print("No images found for source:", source)
            return
        process_image_files(model, files, args.out, args.conf, args.device)


if __name__ == "__main__":
    main()