"""Build the fallback loop: camera-like frames for when the camera dies.

These are NOT pre-rendered screens. They are plain frames -- a room, a group
walking in, one of them raising poster A, then poster B -- and the app feeds
them through the live pipeline exactly as it would camera frames. So whatever
the audience sees during a fallback (segmentation, trigger, effect) is really
being computed, not replayed.

    .venv/bin/python tools/make_fallback.py              # staged scene
    .venv/bin/python tools/make_fallback.py --img-dir D  # your own photos

With --img-dir, every photo with a person in it becomes a 3-second shot (as is,
then with poster A on the largest person). Photos of real people stay out of
git: assets/fallback/ is ignored.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scene  # noqa: E402

W, H, FPS = 1280, 720, 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--n", type=int, default=6, help="max photos with --img-dir")
    ap.add_argument("--side", type=int, default=200, help="poster width, px")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    seg_w = os.path.join(ROOT, "assets/models/yolov8n-seg.pt")
    pa = cv2.imread(os.path.join(ROOT, "assets/patches/magic_dog.png"))
    pb = cv2.imread(os.path.join(ROOT, "assets/decoys/dog_photo.jpg"))
    shots: list[tuple[np.ndarray, float]] = []          # (frame, seconds)

    if args.img_dir:
        from ultralytics import YOLO
        y = YOLO(seg_w)
        for fn in sorted(os.listdir(args.img_dir)):
            if len(shots) >= 2 * args.n:
                break
            img = cv2.imread(os.path.join(args.img_dir, fn))
            if img is None:
                continue
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            r = y.predict(img, classes=[0], conf=0.4, device=args.device,
                          verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            b = max(r.boxes.xyxy.cpu().numpy(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            shots += [(img, 3.0), (scene.paste_poster(img, b, pa, args.side)[0], 3.0)]
    else:
        room = scene.room()
        group, boxes = scene.compose(room, scene.cutouts(seg_w, args.device))
        holder = boxes[len(boxes) // 2]
        shots = [(room, 3.0), (group, 3.0),
                 (scene.paste_poster(group, holder, pa, args.side)[0], 4.0),
                 (group, 2.0),
                 (scene.paste_poster(group, holder, pb, args.side)[0], 4.0)]

    if not shots:
        sys.exit("no usable frames")
    out_dir = os.path.join(ROOT, "assets/fallback")
    os.makedirs(out_dir, exist_ok=True)
    clip = os.path.join(out_dir, "loop.mp4")
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(0)
    for frame, secs in shots:
        for _ in range(int(secs * FPS)):
            # a little sensor noise, so the networks see a live-like signal
            noisy = np.clip(frame + rng.normal(0, 2.0, frame.shape), 0, 255)
            vw.write(noisy.astype(np.uint8))
    vw.release()
    total = sum(s for _, s in shots)
    print(f"  {os.path.relpath(clip, ROOT)}  ({total:.0f}s, "
          f"{os.path.getsize(clip) // 1024} KB)")


if __name__ == "__main__":
    main()
