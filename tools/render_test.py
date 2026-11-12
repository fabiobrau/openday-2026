"""Render the full screen offscreen, in every show state, to PNGs.

Drives the REAL pipeline -- both networks, the background plate, the effects --
on a staged scene (tools/scene.py): an empty room, then three people walk in,
then the middle one raises poster A (and, as a control, poster B). Lets us check
layout, wording and both languages without a camera or a fullscreen window.

    .venv/bin/python tools/render_test.py --out /tmp/screens [--device cpu]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as A  # noqa: E402
import scene  # noqa: E402


class Clock:
    """Fake monotonic clock: one tick per frame at a steady 24 fps, so the
    chart, latch and fade behave as they would live, at any render speed."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/screens")
    ap.add_argument("--device", default=None, help="override config device")
    ap.add_argument("--side", type=int, default=200,
                    help="poster width in camera pixels (1280-wide frame)")
    ap.add_argument("--lang", default="it,en")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = A.load_config(os.path.join(ROOT, "config.yaml"))
    cfg["ui"]["fullscreen"] = False
    if args.device:
        cfg["device"] = args.device
    dev = cfg["device"]

    room = scene.room()
    people = scene.cutouts(os.path.join(ROOT, cfg["segmentation"]["weights"]), dev)
    group, boxes = scene.compose(room, people)
    pa = cv2.imread(os.path.join(ROOT, cfg["patch"]["images"]["adversarial"]))
    pb = cv2.imread(os.path.join(ROOT, cfg["patch"]["images"]["dog"]))
    holder = boxes[len(boxes) // 2] if boxes else (500, 100, 780, 720)
    with_a, _ = scene.paste_poster(group, holder, pa, args.side)
    with_b, _ = scene.paste_poster(group, holder, pb, args.side)

    clock = Clock()
    time.monotonic = clock                     # app, trigger and effects agree

    for lang in args.lang.split(","):
        show = A.Show(cfg, A.build_parser().parse_args(
            ["--lang", lang, "--windowed"]))
        show._fps = 24.0

        def run(frame, n):
            for _ in range(n):
                clock.t += 1 / 24
                clean, model = show._process(frame)
            return clean, model

        def save(name, frame, n=1):
            clean, model = run(frame, n)
            cv2.imwrite(os.path.join(args.out, f"{lang}_{name}.png"),
                        show._compose(clean, model, False))
            print(f"  {lang}_{name}.png   trigger {show.trigger.score:.2f}  "
                  f"level {show._level:.2f}  seg {show._seg.ms:.0f} ms")

        save("1_empty_room", room, 24)              # the plate learns the room
        save("2_group", group, 48)
        save("3_vanish_A", with_a, 24)
        save("4_back", group, 24)
        save("5_decoy_B", with_b, 48)               # control: nothing happens
        show.effect = "relabel"
        save("6_relabel_A", with_a, 36)
        show.view = "mask"
        save("7_mask_view", with_a, 6)
        show.view = "matrix"
        save("8_matrix_view", with_a, 6)
        show.view, show.effect, show.backstage = "overlay", "vanish", True
        save("9_backstage", with_a, 12)
        cv2.imwrite(os.path.join(args.out, f"{lang}_10_nocamera.png"),
                    show._compose_waiting())
        print(f"  {lang}_10_nocamera.png")
        show.shutdown()


if __name__ == "__main__":
    main()
