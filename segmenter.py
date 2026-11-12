"""YOLOv8n-seg wrapper: instance masks -> one per-pixel class map.

YOLO-seg is an *instance* segmentation model: it returns one mask per object,
each with a class and a confidence. The demo shows *semantic* segmentation --
every pixel gets exactly one class -- so the instances are flattened into a
single label map, the most confident instance on top. Keeping the instances
around as well is what lets the effect pick out *one* person in a group.

Masks are kept at the model's own resolution (e.g. 360x640 for a 1280x720
camera at `imgsz: 640`), with the letterbox padding already cut away, so they
cover exactly the camera frame and can be resized straight onto any panel.

Like everything in the frame loop, `segment()` never raises.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("openday.segmenter")

NONE = 255          # label-map value for "no class"
PERSON = 0


@dataclass
class Instance:
    cls: int
    conf: float
    box: tuple[float, float, float, float]   # camera-frame pixels, x1 y1 x2 y2
    mask: np.ndarray                          # bool, SegResult.mask_shape

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class SegResult:
    instances: list[Instance]
    frame_shape: tuple[int, int]   # (h, w) of the camera frame
    mask_shape: tuple[int, int]    # (h, w) of every instance mask
    ms: float = 0.0                # inference + post-processing time

    @property
    def persons(self) -> list[Instance]:
        return [i for i in self.instances if i.cls == PERSON]

    def label_map(self, override: dict[int, int] | None = None) -> np.ndarray:
        """Flatten instances into one uint8 map at mask_shape.

        `override` maps an instance *index* to the label to paint instead --
        NONE removes it from the map, any other value relabels it. Painting goes
        in ascending confidence, so where two masks overlap the surer one wins.
        """
        out = np.full(self.mask_shape, NONE, np.uint8)
        override = override or {}
        order = sorted(range(len(self.instances)),
                       key=lambda k: self.instances[k].conf)
        for k in order:
            lab = override.get(k, self.instances[k].cls)
            if lab == NONE:
                continue
            out[self.instances[k].mask] = lab
        return out

    @staticmethod
    def empty(frame_shape: tuple[int, int]) -> "SegResult":
        h, w = frame_shape
        return SegResult([], (h, w), (max(1, h // 2), max(1, w // 2)))


class Segmenter:
    def __init__(self, weights: str, device: str = "mps", imgsz: int = 640,
                 conf: float = 0.4, classes: list[int] | None = None,
                 min_area_frac: float = 0.002) -> None:
        from ultralytics import YOLO  # imported late: keeps --help instant

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"model weights not found: {weights}\n"
                "Run tools/fetch_assets.py before the event -- the venue wifi "
                "will fail.")
        self.name = os.path.splitext(os.path.basename(weights))[0]
        self.imgsz = imgsz
        self.conf = conf
        self.classes = classes
        self.device = device
        self.min_area_frac = min_area_frac
        self.model = YOLO(weights)
        self.model_input: tuple[int, int] | None = None   # (h, w) the net saw
        self._fail = 0
        self._calls = 0

    def warmup(self, shape: tuple[int, int] = (720, 1280)) -> None:
        self.segment(np.zeros((shape[0], shape[1], 3), np.uint8))

    def segment(self, frame: np.ndarray) -> SegResult:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        try:
            res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                     classes=self.classes, device=self.device,
                                     retina_masks=False, verbose=False)[0]
            out = self._unpack(res, h, w)
            self._fail = 0
            self._calls += 1
            # Torch's MPS allocator creeps upward over a long afternoon; evict
            # now and then. Cheap, and a no-op on other devices.
            if self._calls % 600 == 0 and self.device == "mps":
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass
        except Exception as exc:
            self._fail += 1
            if self._fail in (1, 10, 100):
                log.error("segmentation failed (%dx): %s", self._fail, exc)
            out = SegResult.empty((h, w))
        out.ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _unpack(self, res, h: int, w: int) -> SegResult:
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            shape = self._crop_shape(h, w)
            return SegResult([], (h, w), shape)

        m = res.masks.data                       # (N, mh, mw), letterboxed
        mh, mw = int(m.shape[1]), int(m.shape[2])
        self.model_input = (mh, mw)
        top, bottom, left, right = _unpad(mh, mw, h, w)
        m = (m[:, top:bottom, left:right] > 0.5).cpu().numpy()

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        min_px = self.min_area_frac * m.shape[1] * m.shape[2]
        inst = []
        for k in range(len(clss)):
            if m[k].sum() < min_px:
                continue                          # specks: not worth a colour
            inst.append(Instance(int(clss[k]), float(confs[k]),
                                 tuple(float(v) for v in xyxy[k]), m[k]))
        return SegResult(inst, (h, w), (m.shape[1], m.shape[2]))

    def _crop_shape(self, h: int, w: int) -> tuple[int, int]:
        if self.model_input is None:
            return max(1, h // 2), max(1, w // 2)
        mh, mw = self.model_input
        t, b, l, r = _unpad(mh, mw, h, w)
        return b - t, r - l


def _unpad(mh: int, mw: int, h: int, w: int) -> tuple[int, int, int, int]:
    """The rows/cols of a letterboxed (mh, mw) map that hold the (h, w) frame.

    Same arithmetic as ultralytics' own scale_image, so the cut lands on the
    same pixel the network padded.
    """
    gain = min(mh / h, mw / w)
    pw, ph = (mw - w * gain) / 2, (mh - h * gain) / 2
    top, left = int(round(ph - 0.1)), int(round(pw - 0.1))
    bottom, right = int(round(mh - ph + 0.1)), int(round(mw - pw + 0.1))
    return top, max(top + 1, bottom), left, max(left + 1, right)


def resize_mask(mask: np.ndarray, w: int, h: int, soft: bool = False) -> np.ndarray:
    """Bool mask -> (h, w). soft=True returns a float 0..1 edge for blending."""
    m = mask.astype(np.uint8) * 255
    if soft:
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR
                          ).astype(np.float32) / 255.0
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
