"""Stage a synthetic open-day scene: an empty room, then a group walks in.

Shared by render_test.py, soak_test.py and validate_demo.py. People are cut out
of real photographs with the segmenter itself, scaled to stand at the distance
the camera will see them from, and pasted into an empty room -- so the same
room exists both with and without people, which is what the background plate
(and therefore the vanish effect) needs to be tested honestly.

Fixtures come from tools/fetch_assets.py (assets/testimg/, not committed).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTIMG = os.path.join(ROOT, "assets/testimg")
W, H = 1280, 720

ROOM = "coco_000000000384.jpg"
PEOPLE = ("coco_000000000294.jpg", "coco_000000000446.jpg",
          "coco_000000000113.jpg", "coco_000000000165.jpg")


def load(name: str) -> np.ndarray | None:
    return cv2.imread(os.path.join(TESTIMG, name))


def room() -> np.ndarray:
    img = load(ROOM)
    if img is None:
        img = np.full((H, W, 3), 90, np.uint8)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def cutouts(seg_weights: str, device: str = "cpu", max_n: int = 3
            ) -> list[np.ndarray]:
    """Largest person of each photo as a tight BGRA crop."""
    from ultralytics import YOLO
    model = YOLO(seg_weights)
    out = []
    for name in PEOPLE:
        img = load(name)
        if img is None:
            continue
        r = model.predict(img, classes=[0], conf=0.4, retina_masks=True,
                          device=device, verbose=False)[0]
        if r.masks is None or not len(r.boxes):
            continue
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.cpu().numpy()]
        k = int(np.argmax(areas))
        m = (r.masks.data[k].cpu().numpy() > 0.5).astype(np.uint8) * 255
        x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[k].cpu().numpy())
        crop = np.dstack([img[y1:y2, x1:x2], m[y1:y2, x1:x2]])
        if crop.size:
            out.append(crop)
        if len(out) >= max_n:
            break
    return out


def compose(bg: np.ndarray, people: list[np.ndarray], height_frac: float = 0.85,
            xs: tuple[float, ...] = (0.24, 0.52, 0.78)
            ) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Paste the cut-outs standing in a row; return frame and their boxes."""
    frame = bg.copy()
    boxes = []
    for p, xc in zip(people, xs):
        ph, pw = p.shape[:2]
        th = int(H * height_frac)
        tw = max(1, int(pw * th / ph))
        p = cv2.resize(p, (tw, th), interpolation=cv2.INTER_AREA)
        x1 = int(xc * W - tw / 2)
        y1 = H - th + int(0.06 * H)          # feet just below the frame edge
        _alpha_paste(frame, p, x1, y1)
        boxes.append((max(0, x1), max(0, y1), min(W, x1 + tw), min(H, y1 + th)))
    return frame, boxes


def paste_poster(frame: np.ndarray, box, poster: np.ndarray, side: int,
                 tilt: float = 0.0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Hold `poster` at chest height on the person in `box`, `side` px wide."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, y1 + 0.42 * (y2 - y1)
    h = side / 2
    d = tilt * h
    dst = np.float32([[cx - h + d, cy - h], [cx + h + d, cy - h + 0.15 * d],
                      [cx + h - d, cy + h], [cx - h - d, cy + h - 0.15 * d]])
    src = np.float32([[0, 0], [poster.shape[1] - 1, 0],
                      [poster.shape[1] - 1, poster.shape[0] - 1],
                      [0, poster.shape[0] - 1]])
    out = frame.copy()
    cv2.warpPerspective(poster, cv2.getPerspectiveTransform(src, dst),
                        (frame.shape[1], frame.shape[0]), dst=out,
                        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
    return out, (int(cx - h), int(cy - h), int(cx + h), int(cy + h))


def _alpha_paste(dst: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    h, w = rgba.shape[:2]
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    src = rgba[Y0 - y:Y1 - y, X0 - x:X1 - x]
    a = cv2.GaussianBlur(src[:, :, 3], (5, 5), 0)[..., None].astype(np.float32) / 255
    roi = dst[Y0:Y1, X0:X1]
    roi[:] = (roi * (1 - a) + src[:, :, :3] * a).astype(np.uint8)
