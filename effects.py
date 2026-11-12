"""What happens to the person holding poster A: vanish, or change class.

Both effects work on the segmentation output, at panel resolution:

  vanish   -- the person's pixels are replaced by the background, learned from
              every frame in which that patch of the scene was *not* covered by
              a person. Their mask leaves the class map too.
  relabel  -- the mask stays exactly where it is, but is painted, named and
              counted as another class.

Neither is something the model does. The segmenter still segments the person
perfectly well; the app reads poster A's signal (trigger.py) and edits the
output. That is the honest sentence to say out loud, and the backstage view
(D) draws the real mask on top to prove it.
"""
from __future__ import annotations

import cv2
import numpy as np

from segmenter import PERSON, Instance


class BackgroundPlate:
    """A clean plate of the scene, learned from the pixels nobody stands on.

    Every frame, pixels outside the (dilated) person masks are blended into a
    running average. A pixel that has never been seen uncovered has no plate
    yet and is in-painted from its neighbours instead -- so at setup, let the
    camera see the empty scene for a few seconds, and press B to start over if
    the camera moves.

    Masks may come in at any resolution. Every morphological step runs at
    quarter size: a 31-px dilation at 800x450 costs ~3 ms, and three of them a
    frame was a tenth of the frame budget for edges nobody can see.
    """

    Q = 4                                       # work resolution divisor

    def __init__(self, w: int, h: int, rate: float = 0.04, dilate_px: int = 15,
                 feather_px: int = 9) -> None:
        self.w, self.h = w, h
        self.sw, self.sh = max(8, w // self.Q), max(8, h // self.Q)
        self.rate = rate
        r = max(1, round(dilate_px / self.Q))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * r + 1, 2 * r + 1))
        self.sigma = max(0.5, feather_px / self.Q / 2)
        self.reset()

    def reset(self) -> None:
        self.plate = np.zeros((self.h, self.w, 3), np.float32)
        self.seen = np.zeros((self.h, self.w), bool)

    @property
    def coverage(self) -> float:
        """Share of the scene that has a learned background (0..1)."""
        return float(self.seen.mean())

    def _grow(self, mask: np.ndarray) -> np.ndarray:
        """Bool mask (any size) -> dilated uint8 0/255 at quarter size."""
        small = cv2.resize(mask.astype(np.uint8) * 255, (self.sw, self.sh),
                           interpolation=cv2.INTER_AREA)
        return cv2.dilate((small > 0).astype(np.uint8) * 255, self.kernel)

    def update(self, img: np.ndarray, persons: np.ndarray) -> None:
        """`img` panel-sized BGR, `persons` bool mask of every person."""
        busy = cv2.resize(self._grow(persons), (self.w, self.h),
                          interpolation=cv2.INTER_NEAREST) > 0
        free = ~busy
        new = free & ~self.seen
        if new.any():                                  # first sight: copy
            self.plate[new] = img[new]
            self.seen |= new
        learn = (free & ~new).astype(np.uint8)
        if learn.any():
            cv2.accumulateWeighted(img, self.plate, self.rate, learn)

    def fill(self, img: np.ndarray, region: np.ndarray) -> np.ndarray:
        """The plate, with any never-seen pixels inside `region` in-painted.

        `region` must be panel-sized here: it is compared with `seen`.
        """
        out = self.plate.astype(np.uint8)
        hole = region & ~self.seen
        if hole.any():
            # In-paint at quarter resolution: a full-size cv2.inpaint costs tens
            # of ms, and the result is a soft smear either way.
            small = cv2.resize(img, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
            patched = cv2.inpaint(small, self._grow(region), 5, cv2.INPAINT_TELEA)
            big = cv2.resize(patched, (self.w, self.h),
                             interpolation=cv2.INTER_LINEAR)
            cv2.copyTo(big, hole.astype(np.uint8), out)
        return out

    def soft(self, region: np.ndarray) -> np.ndarray:
        """Feathered float alpha for `region`, so the cut-out has no hard edge."""
        small = cv2.GaussianBlur(self._grow(region), (0, 0), self.sigma)
        return cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR
                          ).astype(np.float32) / 255.0


def pick_holder(persons: list[tuple[int, Instance]],
                anchor: tuple[float, float, float, float] | None,
                ) -> int | None:
    """Which person is holding the poster? Returns the instance index.

    `anchor` is the latched class-78 box -- the poster itself. Held at the
    chest, it sits inside its holder's box; in a group, the friend standing
    behind overlaps it less. Score = share of the poster inside each person's
    box, ties broken by distance to the poster centre.
    """
    if not persons:
        return None
    if anchor is None:
        return max(persons, key=lambda p: p[1].area)[0]
    ax1, ay1, ax2, ay2 = anchor
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))

    def score(p: tuple[int, Instance]) -> tuple[float, float]:
        x1, y1, x2, y2 = p[1].box
        iw = max(0.0, min(x2, ax2) - max(x1, ax1))
        ih = max(0.0, min(y2, ay2) - max(y1, ay1))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return round(iw * ih / a_area, 1), -np.hypot(cx - acx, cy - acy)

    return max(persons, key=score)[0]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def follow(persons: list[tuple[int, Instance]],
           last_box: tuple[float, float, float, float] | None,
           min_iou: float = 0.3) -> int | None:
    """Keep hold of the same person while the effect fades out."""
    if last_box is None or not persons:
        return None
    k, best = max(((k, iou(i.box, last_box)) for k, i in persons),
                  key=lambda t: t[1])
    return k if best >= min_iou else None


def persons_of(instances: list[Instance]) -> list[tuple[int, Instance]]:
    return [(k, i) for k, i in enumerate(instances) if i.cls == PERSON]
