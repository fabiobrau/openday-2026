"""Does poster A move any network other than the one it was optimised on?

Poster A was optimised against `yolov8n` (detection) to light up class 78.
This pastes it -- and decoy B -- at chest height on the largest person of each
photo, at demo framing, and reads class 78 from every model given. It is the
measurement behind the two-network design (docs/OPENDAY_NOTES.md).

    .venv/bin/python tools/validate_transfer.py --img-dir /path/to/photos \\
        --models yolov8n,yolov8n-seg,yolo11n-seg --imgsz 1280,640

Models are looked up in assets/models/ (fetch extras with
`tools/fetch_assets.py --all-models`). Any folder of photos of people works;
Ultralytics' coco128 sample is what the notes used.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framing import close_framing  # noqa: E402
from scene import paste_poster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=os.path.join(ROOT, "assets/testimg"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--models", default="yolov8n,yolov8n-seg")
    ap.add_argument("--imgsz", default="1280")
    ap.add_argument("--sides", default="100,160,240")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from ultralytics import YOLO
    finder = YOLO(os.path.join(ROOT, "assets/models/yolov8n.pt"))
    samples = []
    for fn in sorted(os.listdir(args.img_dir)):
        img = cv2.imread(os.path.join(args.img_dir, fn))
        if img is None:
            continue
        r = finder.predict(img, classes=[0], conf=0.5, device=args.device,
                           verbose=False)[0]
        if not len(r.boxes):
            continue
        b = max(r.boxes.xyxy.cpu().numpy(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if (b[3] - b[1]) < 0.45 * img.shape[0]:
            continue                                  # want a close-up person
        framed, nb = close_framing(img, list(map(float, b)))
        if framed is not None:
            samples.append((framed, nb))
        if len(samples) >= args.n:
            break
    if not samples:
        sys.exit("no photo with a large person in --img-dir")

    posters = {"A": cv2.imread(os.path.join(ROOT, "assets/patches/magic_dog.png")),
               "B": cv2.imread(os.path.join(ROOT, "assets/decoys/dog_photo.jpg"))}
    models = {m: YOLO(os.path.join(ROOT, "assets/models", f"{m}.pt"))
              for m in args.models.split(",")}
    print(f"{len(samples)} photos; cells = mean class-78 conf / share >= 0.20 "
          f"(person conf in brackets)")
    for imgsz in (int(s) for s in args.imgsz.split(",")):
        print(f"\nimgsz {imgsz}")
        for side in (int(s) for s in args.sides.split(",")):
            for pk, pimg in posters.items():
                row = f"  {side:4d}px {pk}:"
                for name, m in models.items():
                    c78, cp = [], []
                    for fr, nb in samples:
                        x = paste_poster(fr, nb, pimg, side)[0]
                        r = m.predict(x, imgsz=imgsz, conf=0.01, classes=[0, 78],
                                      device=args.device, verbose=False)[0]
                        cls = r.boxes.cls.cpu().numpy().astype(int)
                        cf = r.boxes.conf.cpu().numpy()
                        c78.append(cf[cls == 78].max() if (cls == 78).any() else 0.0)
                        cp.append(cf[cls == 0].max() if (cls == 0).any() else 0.0)
                    v = np.array(c78)
                    row += (f"   {name} {v.mean():.2f}/{(v >= 0.2).mean():4.0%}"
                            f" ({np.mean(cp):.2f})")
                print(row, flush=True)


if __name__ == "__main__":
    main()
