"""The trigger: read the one output neuron poster A was optimised to drive.

Poster A is not a trick picture. It was optimised by gradient descent against
the exact `yolov8n` weights in assets/models/ so that one output class -- 78,
`hair drier` -- fires as hard as possible wherever the poster appears. No hair
drier will ever be in front of the camera at an open day, so that class turning
on is an unambiguous signal: *someone is holding poster A, and it is here.*

The patch is model-specific. It moves this detector's class 78, not the
segmentation network's (measured: see docs/OPENDAY_NOTES.md), which is why the
trigger runs its own small detector next to the segmenter.

What happens after the signal -- the person vanishing or changing class -- is
the app's doing, not the model's. The segmenter keeps seeing the person the
whole time; the confidence chart under the panels shows exactly that.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

log = logging.getLogger("openday.trigger")


class Trigger:
    def __init__(self, weights: str, device: str = "mps", imgsz: int = 1280,
                 target_class: int = 78, threshold: float = 0.20,
                 latch_seconds: float = 0.5, every_n_frames: int = 1,
                 floor: float = 0.05) -> None:
        from ultralytics import YOLO  # imported late: keeps --help instant

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"model weights not found: {weights}\n"
                "Run tools/fetch_assets.py before the event.")
        self.name = os.path.splitext(os.path.basename(weights))[0]
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.target_class = target_class
        self.threshold = threshold
        self.latch_seconds = latch_seconds
        self.every = max(1, int(every_n_frames))
        self.floor = floor                  # read below threshold, for the chart

        self.score = 0.0                    # strongest class-78 box, last run
        self.peak = 0.0                     # strongest score while latched
        self.box: tuple[float, float, float, float] | None = None  # last hit
        self.ms = 0.0
        self._last_hit = -1e9
        self._n = 0
        self._fail = 0

    def warmup(self, shape: tuple[int, int] = (720, 1280)) -> None:
        self._run(np.zeros((shape[0], shape[1], 3), np.uint8))

    def update(self, frame: np.ndarray, now: float | None = None) -> None:
        """Run the detector every `every` frames; latch the result in time."""
        now = time.monotonic() if now is None else now
        self._n += 1
        if (self._n - 1) % self.every == 0:
            self.score, box = self._run(frame)
            if self.score >= self.threshold:
                self._last_hit = now
                self.box = box
                self.peak = max(self.peak, self.score)
        if not self.fired_at(now):
            self.peak = 0.0

    def fired_at(self, now: float) -> bool:
        # The class fires on roughly half the frames, not all of them; holding
        # it for half a second turns an intermittent signal into a steady one.
        return now - self._last_hit <= self.latch_seconds

    @property
    def fired(self) -> bool:
        return self.fired_at(time.monotonic())

    def _run(self, frame: np.ndarray):
        t0 = time.perf_counter()
        try:
            r = self.model.predict(frame, imgsz=self.imgsz, conf=self.floor,
                                   classes=[self.target_class],
                                   device=self.device, verbose=False)[0]
            self._fail = 0
            best, box = 0.0, None
            if r.boxes is not None and len(r.boxes):
                confs = r.boxes.conf.cpu().numpy()
                k = int(np.argmax(confs))
                best = float(confs[k])
                box = tuple(float(v) for v in r.boxes.xyxy[k].cpu().numpy())
        except Exception as exc:
            self._fail += 1
            if self._fail in (1, 10, 100):
                log.error("trigger inference failed (%dx): %s", self._fail, exc)
            best, box = 0.0, None
        self.ms = (time.perf_counter() - t0) * 1000.0
        return best, box
