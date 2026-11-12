"""OPEN DAY 2026 -- a segmentation network, and a poster that rewrites its output.

A YOLOv8n-seg network labels every pixel of the camera feed live (person,
backpack, phone, ...). One student raises a printed poster; that student either
vanishes from the network's output or turns into another class, while their
friends stay exactly as they were.

    python app.py --lang it

How it really works -- say this out loud, it is the point of the demo: poster A
was optimised to drive ONE output neuron of a second, small detector (class 78,
`hair drier`). The app reads that neuron, finds the person holding the poster,
and edits the segmentation output. The segmenter itself is never fooled; the
signal chart and the backstage view (D) show it still sees the student.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Some MPS kernels still fall back to CPU; set this before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import yaml

import overlay
from effects import BackgroundPlate, follow, persons_of, pick_holder
from segmenter import NONE, Segmenter, SegResult, resize_mask
from strings import Strings
from trigger import Trigger

ROOT = os.path.dirname(os.path.abspath(__file__))
WND = "OPEN DAY 2026"

CANVAS_W, CANVAS_H = 1600, 900
PANEL_W, PANEL_H = 800, 450
BRAND_H = 78                        # top strip: logos left, event name right
HEAD_H = 52                         # panel titles
PANEL_Y = BRAND_H + HEAD_H          # top of the two camera panels
BAR_Y = PANEL_Y + PANEL_H           # top of the metrics bar

VIEWS = ("overlay", "mask", "matrix")
EFFECTS = ("vanish", "relabel")
TARGETS = ("holder", "all")

log = logging.getLogger("openday")


class PatchModelMismatch(RuntimeError):
    """Raised at startup when the poster was not made for the trigger model."""


# --------------------------------------------------------------- camera ---

class Camera:
    """Webcam with automatic reconnection. Never raises at read time."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index = index
        self.width, self.height, self.fps = width, height, fps
        self.cap: cv2.VideoCapture | None = None
        self.failures = 0
        self._next_retry = 0.0
        self.open()

    def open(self) -> bool:
        self.release()
        try:
            cap = cv2.VideoCapture(self.index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not cap.isOpened():
                cap.release()
                return False
            self.cap = cap
            self.failures = 0
            log.info("camera %d opened (%dx%d)", self.index,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            return True
        except Exception as exc:
            log.error("camera open failed: %s", exc)
            return False

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            if time.monotonic() >= self._next_retry:
                self._next_retry = time.monotonic() + 2.0
                self.open()
            return None
        try:
            ok, frame = self.cap.read()
        except Exception as exc:
            log.error("camera read raised: %s", exc)
            ok, frame = False, None
        if not ok or frame is None:
            self.failures += 1
            if self.failures > 15:
                log.warning("camera lost; reconnecting")
                self.release()
                self._next_retry = time.monotonic() + 1.0
            return None
        self.failures = 0
        # NOTE: no flip here. Mirroring is a *display* transform applied in
        # Show._process; the trigger must see the world as the camera does, or
        # the printed poster reaches it horizontally flipped and weakens.
        return frame

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class FallbackSource:
    """Loop a recorded clip, or cycle stills, through the live pipeline."""

    def __init__(self, video: str, stills_dir: str) -> None:
        self.cap = None
        self.stills: list[np.ndarray] = []
        self._i = 0
        self._last = 0.0
        if os.path.exists(video):
            cap = cv2.VideoCapture(video)
            if cap.isOpened():
                self.cap = cap
                log.info("fallback clip: %s", video)
        if os.path.isdir(stills_dir):
            for fn in sorted(os.listdir(stills_dir)):
                img = cv2.imread(os.path.join(stills_dir, fn))
                if img is not None:
                    self.stills.append(img)
            if self.stills:
                log.info("fallback stills: %d", len(self.stills))

    def read(self) -> np.ndarray | None:
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if ok:
                return frame
        if self.stills:
            now = time.monotonic()
            if now - self._last > 3.0:
                self._last = now
                self._i = (self._i + 1) % len(self.stills)
            return self.stills[self._i].copy()
        return None


# ------------------------------------------------------------------ show ---

def _bgr(v) -> tuple[int, int, int]:
    return tuple(int(x) for x in v)


class Show:
    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        ui = cfg["ui"]
        self.colors = {k: _bgr(v) for k, v in ui["colors"].items()}
        overlay.set_font(os.path.join(ROOT, ui["font"]))

        device = str(cfg.get("device", "mps"))
        sg = cfg["segmentation"]
        self.classes = [int(c) for c in sg["classes"]]
        self.segmenter = Segmenter(os.path.join(ROOT, sg["weights"]), device,
                                   int(sg["imgsz"]), float(sg["conf"]),
                                   self.classes,
                                   float(sg.get("min_area_frac", 0.002)))
        tg = cfg["trigger"]
        self.trigger = Trigger(os.path.join(ROOT, tg["weights"]), device,
                               int(tg["imgsz"]), int(tg["target_class"]),
                               float(tg["threshold"]),
                               float(tg.get("latch_seconds", 0.5)),
                               int(tg.get("every_n_frames", 1)))
        self._fade = float(tg.get("fade_seconds", 0.35))

        names = getattr(self.segmenter.model, "names", {}) or {}
        self.S = Strings(args.lang or ui["lang"], dict(names))

        ef = cfg["effect"]
        self.effect = args.effect or str(ef.get("mode", "vanish"))
        self.relabel_to = int(ef.get("relabel_to", tg["target_class"]))
        self.target_mode = str(ef.get("target", "holder"))
        self.plate = BackgroundPlate(PANEL_W, PANEL_H,
                                     float(ef.get("plate_rate", 0.04)),
                                     int(ef.get("plate_dilate_px", 15)))

        self.class_colors = {int(k): _bgr(v) for k, v in ui["class_colors"].items()}
        self.lut = overlay.color_lut(self.class_colors)
        self.view = args.view or str(ui.get("view", "overlay"))
        self.mask_alpha = float(ui.get("mask_alpha", 0.5))
        self.mirror = bool(cfg["camera"]["mirror"])
        self.fullscreen = bool(ui["fullscreen"])
        self.backstage = False
        self._force = False                 # operator override (V)

        self.matrix = overlay.MatrixView(PANEL_W, PANEL_H,
                                         ghost=float(ui.get("matrix_ghost", 0.0)))
        self.chart = overlay.SignalChart(float(ui.get("chart_seconds", 10)))
        brand = cfg.get("branding", {})
        entries = []
        for e in brand.get("logos", []) or []:
            e = dict(e)
            e["path"] = os.path.join(ROOT, e.get("path", ""))
            entries.append(e)
        self.branding = overlay.Branding(entries, str(brand.get("text", "")))

        self.camera: Camera | None = None
        self.fallback = FallbackSource(os.path.join(ROOT, cfg["fallback"]["video"]),
                                       os.path.join(ROOT, cfg["fallback"]["stills_dir"]))

        # per-frame state, read by the drawing code
        self._level = 0.0                   # 0 = untouched output, 1 = full effect
        self._target_box = None             # box of the person being edited
        self._seg: SegResult | None = None
        self._shown_labels: np.ndarray | None = None   # after the effect
        self._person_conf = 0.0
        self._last_frame: np.ndarray | None = None
        self._last_frame_at = 0.0
        self._canvas = np.full((CANVAS_H, CANVAS_W, 3), self.colors["panel_bg"],
                               np.uint8)
        self._fps = 0.0
        self._t_last = time.monotonic()
        self._running = True

        log.warning("TRIGGER: the effect fires when class %d of %s rises above "
                    "%.2f -- the segmenter is not being fooled.",
                    self.trigger.target_class, self.trigger.name,
                    self.trigger.threshold)

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        if not self.args.fallback:
            cam = self.cfg["camera"]
            self.camera = Camera(int(self.args.camera if self.args.camera is not None
                                     else cam["index"]),
                                 int(cam["width"]), int(cam["height"]),
                                 int(cam["fps"]))
        self.segmenter.warmup()
        self.trigger.warmup()

        cv2.namedWindow(WND, cv2.WINDOW_NORMAL)
        self._apply_fullscreen()

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as exc:  # the show must never die in front of an audience
                log.exception("frame failed, continuing: %s", exc)
                time.sleep(0.05)

        self.shutdown()
        return 0

    def _tick(self) -> None:
        frame = self.camera.read() if self.camera else None
        using_fallback = False
        now = time.monotonic()

        if frame is not None:
            self._last_frame = frame
            self._last_frame_at = now
        else:
            # A webcam drops the odd frame: ride it out on the last good one.
            if self._last_frame is not None and now - self._last_frame_at < 1.0:
                frame = self._last_frame
            else:
                fb = self.fallback.read()
                if fb is not None:
                    frame, using_fallback = fb, True

        if frame is None:
            canvas = self._compose_waiting()
        else:
            clean, model = self._process(frame)
            canvas = self._compose(clean, model, using_fallback)
        cv2.imshow(WND, canvas)

        self._handle_keys()

        now = time.monotonic()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

    # -- the important part ------------------------------------------------
    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (camera panel, model panel), both mirrored for display.

        Both networks see the raw, unmirrored frame. The effect is applied to
        the segmentation *output* afterwards -- never to the network's input.
        """
        now = time.monotonic()
        self.trigger.update(frame, now)
        seg = self.segmenter.segment(frame)
        self._seg = seg

        fired = self.trigger.fired_at(now)
        want = self._force or fired
        self._ease(want)
        targets = self._pick_targets(seg, want, fired)

        persons = persons_of(seg.instances)
        if targets:
            self._person_conf = seg.instances[targets[0]].conf
        elif persons:
            self._person_conf = max(persons, key=lambda p: p[1].area)[1].conf
        else:
            self._person_conf = 0.0
        self.chart.push(now, self._person_conf, self.trigger.score, self._level)

        # One downscale serves both panels. INTER_LINEAR, not AREA: at a
        # non-integer 1.6x it is several times faster and looks the same.
        small = cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        model = self._render_model(small, seg, targets, now)
        clean = cv2.flip(small, 1) if self.mirror else small
        return clean, model

    def _ease(self, want: bool) -> None:
        # Ease in and out instead of cutting: an instant switch reads as fake.
        step = 1.0 / max(1e-3, self._fade * max(self._fps, 1.0))
        target = 1.0 if want else 0.0
        if self._level < target:
            self._level = min(target, self._level + step)
        elif self._level > target:
            self._level = max(target, self._level - step)

    def _pick_targets(self, seg: SegResult, want: bool, fired: bool) -> list[int]:
        """Instance indices the effect applies to this frame."""
        if self._level <= 0.0:
            self._target_box = None
            return []
        persons = persons_of(seg.instances)
        if self.target_mode == "all":
            return [k for k, _ in persons]
        if fired:
            k = pick_holder(persons, self.trigger.box)
        elif want:                                   # manual override, no poster
            k = follow(persons, self._target_box)
            if k is None:
                k = pick_holder(persons, None)
        else:                                        # fading out: same person
            k = follow(persons, self._target_box)
        if k is None:
            return []
        self._target_box = seg.instances[k].box
        return [k]

    def _render_model(self, base: np.ndarray, seg: SegResult,
                      targets: list[int], now: float) -> np.ndarray:
        """`base` is the camera frame already at panel size, unmirrored."""
        pw, ph = PANEL_W, PANEL_H

        # learn the background wherever nobody is standing
        persons = np.zeros(seg.mask_shape, bool)
        for i in seg.persons:
            persons |= i.mask
        self.plate.update(base, persons)

        labels = seg.label_map()
        level = self._level
        effect_on = bool(targets) and level > 0.0
        if effect_on:
            new = NONE if self.effect == "vanish" else self.relabel_to
            labels_fx = seg.label_map({k: new for k in targets})
            region = np.zeros(seg.mask_shape, bool)
            for k in targets:
                region |= seg.instances[k].mask
            region = resize_mask(region, pw, ph)
            alpha = self.plate.soft(region) * level
            if self.effect == "vanish":
                base_fx = self.plate.fill(base, region)
            else:
                base_fx = base
            self._shown_labels = labels_fx if level >= 0.5 else labels
        else:
            labels_fx, region, alpha, base_fx = labels, None, None, base
            self._shown_labels = labels

        L0 = cv2.resize(labels, (pw, ph), interpolation=cv2.INTER_NEAREST)
        if self.view == "matrix":
            img = base
            if effect_on:
                img = cv2.blendLinear(base, base_fx, 1.0 - alpha, alpha)
                if self.effect == "vanish" and level >= 0.5 and self._glitch_ready:
                    self._glitch(seg, targets, now)
            if not effect_on or level < 0.5:
                self._glitch_ready = True
            rain = self.matrix.render(img, now)
            L = cv2.resize(self._shown_labels, (pw, ph),
                           interpolation=cv2.INTER_NEAREST)
            out = overlay.paint_classes(rain, L, self.lut, "matrix")
        else:
            out = overlay.paint_classes(base, L0, self.lut, self.view,
                                        self.mask_alpha)
            if effect_on:
                L1 = cv2.resize(labels_fx, (pw, ph), interpolation=cv2.INTER_NEAREST)
                fx = overlay.paint_classes(base_fx, L1, self.lut, self.view,
                                           self.mask_alpha)
                out = cv2.blendLinear(out, fx, 1.0 - alpha, alpha)
            self._draw_outlines(out, seg, targets, level)

        if self.backstage:
            self._draw_backstage_shapes(out, seg, targets)
        if self.mirror:
            out = cv2.flip(out, 1)
        self._draw_labels(out, seg, targets, level)
        if self.backstage:
            self._draw_backstage_text(out)
        return out

    _glitch_ready = True

    def _glitch(self, seg: SegResult, targets: list[int], now: float) -> None:
        """Scramble the Matrix cells where the student just vanished."""
        self._glitch_ready = False
        h, w = seg.frame_shape
        x1, y1, x2, y2 = seg.instances[targets[0]].box
        self.matrix.glitch((x1 / w, y1 / h, x2 / w, y2 / h), now)

    # -- drawing on the model panel ----------------------------------------
    def _draw_outlines(self, img, seg: SegResult, targets, level) -> None:
        for k, inst in enumerate(seg.instances):
            col = self.class_colors.get(inst.cls, (230, 230, 230))
            if k in targets and level >= 0.5:
                if self.effect == "vanish":
                    continue
                col = self.class_colors.get(self.relabel_to, col)
            overlay.outline(img, inst.mask, _lighter(col), 2)

    def _draw_labels(self, img, seg: SegResult, targets, level) -> None:
        """Class name and confidence on each sizeable object, mirrored to match."""
        pw, ph = img.shape[1], img.shape[0]
        fh, fw = seg.frame_shape
        sx, sy = pw / fw, ph / fh
        min_area = 0.01 * fw * fh
        shown = sorted(((k, i) for k, i in enumerate(seg.instances)
                        if i.area >= min_area), key=lambda t: -t[1].area)[:8]
        placed: list[tuple[int, int, int, int]] = []
        for k, inst in shown:
            cls, conf = inst.cls, inst.conf
            if k in targets and level >= 0.5:
                if self.effect == "vanish":
                    continue
                cls = self.relabel_to
                conf = max(self.trigger.peak, self.trigger.score) or conf
            x1, y1, x2, y2 = inst.box
            if self.mirror:
                x1, x2 = fw - x2, fw - x1
            col = self.class_colors.get(cls, (235, 235, 235))
            text = f"{self.S.cls(cls)} {int(round(conf * 100))}%"
            tw, th = overlay.text_size(text, 26, 3)
            tx = int(min(max(4, x1 * sx + 4), pw - tw - 4))
            ty = int(max(30, y1 * sy + 4))
            # In a group the labels collide: step down until this one is clear.
            for _ in range(6):
                if not any(tx < px2 and px1 < tx + tw and ty < py2 and py1 < ty + th
                           for px1, py1, px2, py2 in placed):
                    break
                ty += th
            placed.append((tx, ty, tx + tw, ty + th))
            overlay.draw_text(img, text, tx, ty, 26, _lighter(col), "tl", 3,
                              weight=600)

    def _draw_backstage_shapes(self, img, seg: SegResult, targets) -> None:
        """The truth, drawn over the show: the real mask and the poster box."""
        real = [seg.instances[k] for k in targets] or seg.persons
        for inst in real:
            overlay.outline(img, inst.mask, (255, 255, 255), 2, dashed=True)
        box = self.trigger.box if self.trigger.fired else None
        if box is not None:
            fh, fw = seg.frame_shape
            sx, sy = img.shape[1] / fw, img.shape[0] / fh
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                          self.colors["neuron"], 3, cv2.LINE_AA)

    def _draw_backstage_text(self, img) -> None:
        w, h = img.shape[1], img.shape[0]
        overlay.draw_text(img, self.S.get("backstage"), w - 12, 10, 26,
                          self.colors["neuron"], "tr", 3, weight=700)
        box = self.trigger.box if self.trigger.fired else None
        seg = self._seg
        if box is not None and seg is not None:
            fh, fw = seg.frame_shape
            x1, y1, x2, y2 = box
            if self.mirror:
                x1, x2 = fw - x2, fw - x1
            overlay.draw_text(
                img, self.S.get("backstage_poster",
                                conf=int(round(max(self.trigger.peak,
                                                   self.trigger.score) * 100))),
                x1 * w / fw, y2 * h / fh + 6, 22, self.colors["neuron"], "tl", 3)
        if self._level > 0 and self._person_conf > 0:
            l1 = self.S.get("backstage_line1",
                            conf=int(round(self._person_conf * 100)))
            l2 = self.S.get(f"backstage_{self.effect}")
            cv2.rectangle(img, (0, h - 66), (w, h), (20, 16, 14), -1)
            overlay.draw_text(img, l1, w // 2, h - 36, 22, (255, 255, 255), "bc", 0,
                              weight=600)
            overlay.draw_text(img, l2, w // 2, h - 8, 20, (215, 215, 215), "bc", 0,
                              weight=500)

    # -- the whole screen --------------------------------------------------
    def _compose(self, clean: np.ndarray, model: np.ndarray,
                 using_fallback: bool) -> np.ndarray:
        # One canvas for the whole session: allocating 1600x900x3 per frame
        # costs ~10 ms, and the panels overwrite themselves anyway.
        canvas = self._canvas
        canvas[0:PANEL_Y] = self.colors["panel_bg"]
        canvas[PANEL_Y:PANEL_Y + PANEL_H, 0:PANEL_W] = clean
        canvas[PANEL_Y:PANEL_Y + PANEL_H, PANEL_W:PANEL_W * 2] = model
        cv2.line(canvas, (PANEL_W, PANEL_Y), (PANEL_W, PANEL_Y + PANEL_H),
                 (250, 250, 250), 2)

        title = self.colors["text"]
        overlay.draw_text(canvas, self.S.get("panel_camera"), PANEL_W // 2,
                          BRAND_H + 8, 32, title, "tc", 0, weight=600)
        overlay.draw_text(canvas, f"{self.S.get('panel_model')} · "
                                  f"{self.segmenter.name}",
                          PANEL_W + PANEL_W // 2, BRAND_H + 8, 32,
                          self.colors["accent"], "tc", 0, weight=600)
        overlay.draw_text(canvas, self.S.get(f"view_{self.view}"),
                          PANEL_W + 12, PANEL_Y + 10, 20, (235, 235, 235), "tl", 3,
                          weight=500)
        if using_fallback:
            overlay.draw_text(canvas, self.S.get("mode_fallback"), 12, PANEL_Y + 10,
                              22, (235, 235, 235), "tl", 3, weight=600)

        self._render_bar(canvas)

        # Logos go on last: _render_bar repaints the whole bottom strip.
        self.branding.draw(canvas, {
            "top-left": (24, BRAND_H // 2, "lm"),
            "top-right": (CANVAS_W - 24, BRAND_H // 2, "rm"),
            "bottom-left": (24, CANVAS_H - 16, "lb"),
            "bottom-right": (CANVAS_W - 24, CANVAS_H - 12, "rb"),
        })
        self.branding.draw_text_at(canvas, CANVAS_W - 24, BRAND_H // 2, 40)
        return canvas

    def _render_bar(self, canvas: np.ndarray) -> None:
        """Metrics bar, three columns: pixels per class | signal chart | status."""
        y0 = BAR_Y
        cv2.rectangle(canvas, (0, y0), (CANVAS_W, CANVAS_H), self.colors["bar_bg"], -1)
        head = (235, 235, 235)
        dim = self.colors["dim"]

        # --- column A: pixels per class (what the right-hand panel shows)
        ax = 28
        overlay.draw_text(canvas, self.S.get("classes_title"), ax, y0 + 16, 24,
                          head, "tl", 0, weight=700)
        overlay.draw_text(canvas, self.S.get("classes_unit"), ax + 372, y0 + 21, 17,
                          dim, "tr", 0, weight=400)
        labels = self._shown_labels
        counts = (np.bincount(labels.ravel(), minlength=256) / labels.size
                  if labels is not None else np.zeros(256))
        rows = list(self.classes)
        if self._level >= 0.5 and self.effect == "relabel" \
                and self.relabel_to not in rows:
            rows.append(self.relabel_to)
        bar_x, bar_w = ax + 172, 120
        row_h = 31 if len(rows) <= 7 else int(31 * 7 / len(rows))
        for r, c in enumerate(rows):
            yy = y0 + 54 + r * row_h
            share = float(counts[c])
            col = self.class_colors.get(c, (200, 200, 200))
            on = share > 0.0005
            cv2.rectangle(canvas, (ax, yy + 4), (ax + 14, yy + 18), col, -1)
            overlay.draw_text(canvas, self.S.cls(c), ax + 24, yy + 11, 20,
                              head if on else dim, "lm", 0, weight=500)
            cv2.rectangle(canvas, (bar_x, yy + 5), (bar_x + bar_w, yy + 17),
                          (52, 46, 42), -1)
            if on:
                # square-root scale: a phone is 0.5% of the frame, a person 30%
                fill = int(bar_w * min(1.0, np.sqrt(share)))
                cv2.rectangle(canvas, (bar_x, yy + 5), (bar_x + max(2, fill), yy + 17),
                              col, -1)
            overlay.draw_text(canvas, f"{share * 100:.1f}%", ax + 372, yy + 11, 20,
                              head if on else dim, "rm", 0, weight=500)

        # --- column B: the network outputs over time
        cx, cw, ch = 470, 600, 196
        cy = y0 + 56
        overlay.draw_text(canvas, self.S.get("chart_title",
                                             s=int(self.chart.seconds)),
                          cx, y0 + 16, 24, head, "tl", 0, weight=700)
        self.chart.draw(canvas, cx, cy, cw, ch,
                        {"person": self.colors["person_trace"],
                         "neuron": self.colors["neuron"],
                         "effect": self.colors["effect"]},
                        self.trigger.threshold)
        lx, ly = cx, cy + ch + 22
        for key, col in (("chart_person", self.colors["person_trace"]),
                         ("chart_neuron", self.colors["neuron"])):
            cv2.line(canvas, (lx, ly), (lx + 22, ly), col, 4, cv2.LINE_AA)
            w, _ = overlay.draw_text(canvas, self.S.get(key), lx + 30, ly, 19,
                                     head, "lm", 0, weight=500)
            lx += 30 + w + 26
        cv2.line(canvas, (lx, ly), (lx + 22, ly), (235, 235, 235), 1, cv2.LINE_AA)
        w, _ = overlay.draw_text(canvas, f"{self.S.get('chart_threshold')} "
                                         f"{self.trigger.threshold:.2f}",
                                 lx + 30, ly, 19, dim, "lm", 0, weight=500)

        # --- column C: status
        sx = 1120
        line = y0 + 18
        fired = self.trigger.fired

        def kv(key: str, value: str, color=head, size=24) -> None:
            nonlocal line
            overlay.draw_text(canvas, self.S.get(key), sx, line, 17, dim, "tl", 0,
                              weight=600)
            overlay.draw_text(canvas, value, sx, line + 20, size, color, "tl", 0,
                              weight=600)
            line += 50

        trig = (f"{self.S.get('trigger_on') if fired else self.S.get('trigger_off')}"
                f"   {self.trigger.score:.2f}")
        if self._force:
            trig += f"   [{self.S.get('manual')}]"
        kv("status_trigger", trig, self.colors["good"] if fired or self._force else head)
        if self.effect == "vanish":
            eff = self.S.get("effect_vanish")
        else:
            eff = self.S.get("effect_relabel", cls=self.S.cls(self.relabel_to))
        kv("status_effect", eff, self.colors["effect"] if self._level > 0 else head)
        kv("status_target", self.S.get(f"target_{self.target_mode}"))
        cov = self.plate.coverage
        plate = f"{cov * 100:.0f}%"
        if cov < 0.7:
            plate += f"  ·  {self.S.get('plate_hint')}"
        kv("status_plate", plate, head if cov >= 0.7 else self.colors["warn"],
           size=20 if cov < 0.7 else 24)
        kv("status_speed", self.S.get("speed_line", seg=self._seg.ms if self._seg else 0,
                                      trg=self.trigger.ms, fps=self._fps), size=20)

        # --- footer
        overlay.draw_text(canvas, self.S.get("privacy"), 28, CANVAS_H - 12, 18,
                          dim, "bl", 0, weight=400)
        overlay.draw_text(canvas, self.S.get("keys_help"), CANVAS_W - 230,
                          CANVAS_H - 12, 18, dim, "br", 0, weight=400)
        if self.cfg["ui"].get("show_fps") or self.args.show_fps:
            n = len(self._seg.instances) if self._seg else 0
            bits = (f"{self._fps:4.1f} fps   cls{self.trigger.target_class} "
                    f"{self.trigger.score:.2f} {'FIRE' if fired else '--'}   "
                    f"inst {n}   level {self._level:.2f}")
            overlay.draw_text(canvas, bits, 12, PANEL_Y + PANEL_H - 8, 20,
                              self.colors["good"] if fired else (200, 200, 200),
                              "bl", 3)

    def _compose_waiting(self) -> np.ndarray:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), self.colors["panel_bg"], np.uint8)
        overlay.draw_text(canvas, self.S.get("camera_lost"), CANVAS_W // 2,
                          CANVAS_H // 2 - 20, 48, (255, 255, 255), "cm", 0, weight=600)
        overlay.draw_text(canvas, self.S.get("camera_retry"), CANVAS_W // 2,
                          CANVAS_H // 2 + 40, 28, (190, 190, 190), "cm", 0, weight=400)
        return canvas

    # -- input -------------------------------------------------------------
    def _handle_keys(self) -> None:
        k = cv2.waitKey(1) & 0xFF
        if k == 255:
            return
        k = chr(k).lower() if k < 128 else ""
        if k == "\x1b":
            self._running = False
        elif k == "f":
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
        elif k == "m":
            self.view = VIEWS[(VIEWS.index(self.view) + 1) % len(VIEWS)]
            log.info("view -> %s", self.view)
        elif k == "e":
            self.effect = EFFECTS[(EFFECTS.index(self.effect) + 1) % len(EFFECTS)]
            log.info("effect -> %s", self.effect)
        elif k == "t":
            self.target_mode = TARGETS[(TARGETS.index(self.target_mode) + 1)
                                       % len(TARGETS)]
            log.info("target -> %s", self.target_mode)
        elif k == "d":
            self.backstage = not self.backstage
        elif k == "b":
            self.plate.reset()
            log.info("background plate reset")
        elif k == "v":
            self._force = not self._force
            log.warning("manual override %s", "ON" if self._force else "OFF")

    def _apply_fullscreen(self) -> None:
        try:
            cv2.setWindowProperty(
                WND, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        except Exception as exc:
            log.debug("fullscreen toggle failed: %s", exc)

    def shutdown(self) -> None:
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()


def _lighter(bgr: tuple[int, int, int], k: float = 0.35) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * k) for c in bgr)


# ------------------------------------------------------------------ main ---

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_patch_matches_model(cfg: dict) -> None:
    """A poster only moves the model it was optimised against. Fail loudly."""
    target = str(cfg["patch"]["target_model"]).strip()
    loaded = str(cfg["trigger"]["name"]).strip()
    if target != loaded:
        raise PatchModelMismatch(
            f"\n  Poster A was optimised against '{target}' but the trigger is "
            f"loading '{loaded}'.\n"
            "  A patch only drives the model it was optimised against; this pair\n"
            "  would make the poster do nothing at all. Fix trigger.name /\n"
            "  patch.target_model in config.yaml, or re-train the patch with\n"
            "  tools/train_patch.py.\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OPEN DAY 2026 segmentation demo")
    ap.add_argument("--lang", choices=("it", "en"), default=None)
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index; never guessed, pick it explicitly")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--fallback", action="store_true",
                    help="no camera: loop assets/fallback through the pipeline")
    ap.add_argument("--view", choices=VIEWS, default=None)
    ap.add_argument("--effect", choices=EFFECTS, default=None)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    cfg = load_config(args.config)
    if args.windowed:
        cfg["ui"]["fullscreen"] = False

    try:
        check_patch_matches_model(cfg)
    except PatchModelMismatch as exc:
        print(f"REFUSING TO START:{exc}", file=sys.stderr)
        return 2

    try:
        show = Show(cfg, args)
    except FileNotFoundError as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return 2
    return show.run()


if __name__ == "__main__":
    sys.exit(main())
