"""Headless soak test: run the real frame loop for N frames, no window.

Checks the things that matter for an unattended afternoon: throughput, memory
growth, and that no frame ever raises. The fake camera cycles the staged scene
(tools/scene.py): empty room, a group, the group with poster A, with poster B,
and drops the odd frame the way a real webcam does. Keys are pressed along the
way so every view and effect is exercised.

    .venv/bin/python tools/soak_test.py --frames 1200
"""
from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as A  # noqa: E402
import scene  # noqa: E402


class FakeCamera:
    def __init__(self, frames: list) -> None:
        self.frames = frames
        self.n = 0

    def read(self):
        self.n += 1
        if self.n % 97 == 0:
            return None                       # simulate a dropped frame
        return self.frames[(self.n // 60) % len(self.frames)].copy()

    def release(self):
        pass


def rss_mb() -> float:
    """Peak RSS (ru_maxrss never decreases -- good for a rising high-water mark)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def rss_now_mb() -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / 1024
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Headless: swallow every window call, and press a key now and then.
    presses = {150: "m", 300: "m", 450: "e", 600: "d", 750: "m", 900: "e",
               1000: "d", 1100: "t", 1150: "t"}
    tick = {"i": 0}
    cv2.imshow = lambda *a, **k: None
    cv2.namedWindow = lambda *a, **k: None
    cv2.setWindowProperty = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: ord(presses.get(tick["i"], "\xff")[0]) \
        if tick["i"] in presses else 255

    cfg = A.load_config(os.path.join(ROOT, "config.yaml"))
    cfg["ui"]["fullscreen"] = False
    if args.device:
        cfg["device"] = args.device

    room = scene.room()
    people = scene.cutouts(os.path.join(ROOT, cfg["segmentation"]["weights"]),
                           cfg["device"])
    group, boxes = scene.compose(room, people)
    holder = boxes[len(boxes) // 2]
    pa = cv2.imread(os.path.join(ROOT, cfg["patch"]["images"]["adversarial"]))
    pb = cv2.imread(os.path.join(ROOT, cfg["patch"]["images"]["dog"]))
    frames = [room, group, scene.paste_poster(group, holder, pa, 200)[0],
              group, scene.paste_poster(group, holder, pb, 200)[0]]

    show = A.Show(cfg, A.build_parser().parse_args(["--windowed"]))
    show.camera = FakeCamera(frames)
    show.segmenter.warmup()
    show.trigger.warmup()

    rss0 = rss_mb()
    t0 = time.monotonic()
    errors = fired = 0
    for i in range(args.frames):
        tick["i"] = i
        try:
            show._tick()
        except Exception as exc:
            errors += 1
            print(f"  FRAME {i} RAISED: {type(exc).__name__}: {exc}")
        fired += int(show._level > 0.99)
        if i and i % 300 == 0:
            print(f"  {i:5d} frames  {i / (time.monotonic() - t0):5.1f} fps  "
                  f"rss now {rss_now_mb():6.1f} MB  peak {rss_mb():6.1f} MB  "
                  f"view {show.view:<7} effect {show.effect}")

    dt = time.monotonic() - t0
    rss1 = rss_mb()
    show.shutdown()

    print(f"\n  frames        : {args.frames}")
    print(f"  throughput    : {args.frames / dt:.1f} fps  ({dt:.1f}s)")
    print(f"  peak rss      : {rss0:.1f} -> {rss1:.1f} MB  (+{rss1 - rss0:.1f})")
    print(f"  current rss   : {rss_now_mb():.1f} MB")
    print(f"  errors        : {errors}")
    print(f"  effect frames : {fired} ({fired / args.frames:.0%}; poster A is up "
          f"~{1 / len(frames):.0%} of the time)")
    ok = errors == 0 and (rss1 - rss0) < 150 and fired > 0
    print("  RESULT        :", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
