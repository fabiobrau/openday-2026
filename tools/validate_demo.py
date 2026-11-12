"""End-to-end check of the shipping path, on a staged group.

For every combination of poster (A magic / B decoy / none), poster size and
which person in the group holds it, this builds the frame (tools/scene.py),
runs the REAL pipeline -- Trigger -> Segmenter -> effects.pick_holder -- and
reports three numbers:

  fires      how often the trigger crosses its threshold (A should be ~100%,
             B and none must be 0%)
  holder     when it fires, how often the effect lands on the person actually
             holding the poster, not a friend (the point of instance masks)
  seg kept   how often the segmenter still finds the holder with the poster up
             (the honest number: the segmenter is never the one fooled)

    .venv/bin/python tools/validate_demo.py [--device cpu] [--sides 120,160,220]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as A  # noqa: E402
import scene  # noqa: E402
from effects import iou, persons_of, pick_holder  # noqa: E402
from segmenter import Segmenter  # noqa: E402
from trigger import Trigger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--sides", default="100,140,200,260",
                    help="poster width in camera pixels (1280-wide frame)")
    ap.add_argument("--tilts", default="0,0.25")
    args = ap.parse_args()

    cfg = A.load_config(os.path.join(ROOT, "config.yaml"))
    dev = args.device or cfg["device"]
    sg, tg = cfg["segmentation"], cfg["trigger"]
    seg = Segmenter(os.path.join(ROOT, sg["weights"]), dev, int(sg["imgsz"]),
                    float(sg["conf"]), [int(c) for c in sg["classes"]])
    trg = Trigger(os.path.join(ROOT, tg["weights"]), dev, int(tg["imgsz"]),
                  int(tg["target_class"]), float(tg["threshold"]))

    room = scene.room()
    people = scene.cutouts(os.path.join(ROOT, sg["weights"]), dev)
    posters = {"A magic": cv2.imread(os.path.join(ROOT, cfg["patch"]["images"]["adversarial"])),
               "B decoy": cv2.imread(os.path.join(ROOT, cfg["patch"]["images"]["dog"]))}
    sides = [int(s) for s in args.sides.split(",")]
    tilts = [float(t) for t in args.tilts.split(",")]
    layouts = [(0.24, 0.52, 0.78), (0.30, 0.55, 0.80), (0.20, 0.47, 0.74)]

    print(f"trigger {trg.name}@{trg.imgsz} thr {trg.threshold}  |  "
          f"segmenter {seg.name}@{seg.imgsz}  |  {len(people)} people, "
          f"{len(layouts)} layouts x {len(tilts)} tilts x each holder\n")
    print(f"  {'poster':<9} {'side':>5}   {'fires':>6}  {'holder':>7}  {'seg kept':>8}"
          f"  {'trigger score':>13}")
    for name, poster in [*posters.items(), ("none", None)]:
        for side in (sides if poster is not None else [0]):
            n = fired = right = kept = 0
            scores = []
            for xs in layouts:
                group, boxes = scene.compose(room, people, xs=xs)
                for h, hb in enumerate(boxes):
                    for tilt in tilts:
                        frame = group if poster is None else \
                            scene.paste_poster(group, hb, poster, side, tilt)[0]
                        trg._last_hit = -1e9
                        trg.update(frame, now=0.0)
                        r = seg.segment(frame)
                        persons = persons_of(r.instances)
                        n += 1
                        scores.append(trg.score)
                        # the instance that IS the holder: best IoU with its box
                        truth = max(persons, key=lambda p: iou(p[1].box, hb),
                                    default=None)
                        if truth is not None and iou(truth[1].box, hb) > 0.4:
                            kept += 1
                        if trg.fired_at(0.0):
                            fired += 1
                            k = pick_holder(persons, trg.box)
                            right += int(truth is not None and k == truth[0])
            sc = sorted(scores)
            print(f"  {name:<9} {side or '-':>5}   {fired / n:6.0%}  "
                  f"{(right / fired if fired else float('nan')):7.0%}  "
                  f"{kept / n:8.0%}  {sc[len(sc) // 2]:6.2f} (max {sc[-1]:.2f})")
    print("\n  holder = share of firing frames where the effect picks the right person")


if __name__ == "__main__":
    main()
