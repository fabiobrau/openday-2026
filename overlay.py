"""All the drawing: text, class maps, the signal chart, the Matrix view, logos.

Text goes through Pillow (OpenCV's Hershey fonts are far too thin to read from
3 m) but every string is rendered once and cached as an RGBA sprite, so the
per-frame cost is a numpy alpha blit, not a PIL round-trip.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("openday.overlay")

_FONT_PATH = "assets/fonts/SpaceGrotesk.ttf"


def set_font(path: str) -> None:
    global _FONT_PATH
    if os.path.exists(path):
        _FONT_PATH = path
    else:
        log.warning("font %s missing; falling back to Pillow default", path)


@lru_cache(maxsize=32)
def _font(size: int, weight: int = 600):
    try:
        f = ImageFont.truetype(_FONT_PATH, size)
        try:
            # Variable font: set the weight axis, leave any other at default.
            axes = f.get_variation_axes()
            vals = [float(np.clip(weight, a["minimum"], a["maximum"]))
                    if b"eight" in a.get("name", b"") else float(a["default"])
                    for a in axes]
            f.set_variation_by_axes(vals)
        except Exception:
            pass  # static font: nothing to vary
        return f
    except Exception as exc:
        log.warning("font load failed (%s); using default", exc)
        return ImageFont.load_default()


@lru_cache(maxsize=512)
def _text_sprite(text: str, size: int, color: tuple, outline: int,
                 outline_color: tuple, weight: int) -> np.ndarray:
    """Render text once into a BGRA sprite. Cached: the timer only ticks 10x/s."""
    font = _font(size, weight)
    pad = outline + max(2, size // 8)
    dummy = Image.new("RGBA", (1, 1))
    box = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font,
                                         stroke_width=outline)
    w = max(1, box[2] - box[0] + 2 * pad)
    h = max(1, box[3] - box[1] + 2 * pad)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text(
        (pad - box[0], pad - box[1]), text, font=font,
        fill=(color[2], color[1], color[0], 255), stroke_width=outline,
        stroke_fill=(outline_color[2], outline_color[1], outline_color[0], 255))
    rgba = np.array(img)
    bgra = rgba[:, :, [2, 1, 0, 3]].copy()
    return bgra


def alpha_blit(dst: np.ndarray, sprite: np.ndarray, x: int, y: int,
               anchor: str = "tl", scale: float = 1.0) -> tuple[int, int]:
    """Blit a BGRA sprite onto a BGR frame. Returns the drawn (w, h)."""
    if sprite is None or sprite.size == 0:
        return 0, 0
    if scale != 1.0:
        sh, sw = sprite.shape[:2]
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        sprite = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_AREA)
    h, w = sprite.shape[:2]

    if "c" in anchor:
        x -= w // 2
    if "r" in anchor:
        x -= w
    if "m" in anchor:
        y -= h // 2
    if "b" in anchor:
        y -= h

    H, W = dst.shape[:2]
    sx1, sy1 = max(0, -x), max(0, -y)
    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(W, x + w), min(H, y + h)
    if dx2 <= dx1 or dy2 <= dy1:
        return w, h

    src = sprite[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)]
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    region = dst[dy1:dy2, dx1:dx2].astype(np.float32)
    dst[dy1:dy2, dx1:dx2] = (region * (1 - a) + src[:, :, :3].astype(np.float32) * a
                             ).astype(np.uint8)
    return w, h


def draw_text(dst, text, x, y, size=48, color=(255, 255, 255), anchor="tl",
              outline=4, outline_color=(30, 20, 15), weight=600):
    sprite = _text_sprite(str(text), int(size), tuple(color), int(outline),
                          tuple(outline_color), int(weight))
    return alpha_blit(dst, sprite, int(x), int(y), anchor)


def text_size(text, size=48, outline=4, weight=600) -> tuple[int, int]:
    s = _text_sprite(str(text), int(size), (255, 255, 255), int(outline),
                     (0, 0, 0), int(weight))
    return s.shape[1], s.shape[0]


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------

def rounded_rect(img, p1, p2, color, thickness=10, radius=28):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    r = int(max(0, min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    for cx, cy, a in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                      (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color,
                    thickness, cv2.LINE_AA)


def corner_box(img, box, color, thickness=12, frac=0.28, radius=24):
    """The detection box: fat cartoon corner brackets, not a thin rectangle."""
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    lx, ly = int(w * frac), int(h * frac)
    r = min(radius, lx, ly)
    segs = [
        ((x1 + r, y1), (x1 + lx, y1)), ((x1, y1 + r), (x1, y1 + ly)),
        ((x2 - lx, y1), (x2 - r, y1)), ((x2, y1 + r), (x2, y1 + ly)),
        ((x1 + r, y2), (x1 + lx, y2)), ((x1, y2 - ly), (x1, y2 - r)),
        ((x2 - lx, y2), (x2 - r, y2)), ((x2, y2 - ly), (x2, y2 - r)),
    ]
    for a, b in segs:
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    for cx, cy, ang in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                        (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color,
                    thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------
# the model's view as numbers
# --------------------------------------------------------------------------

class MatrixView:
    """Green falling-digit rain for the robot panel.

    Every glyph is a digit 0-9 that IS the brightness of that patch of the
    frame, so the effect is also the lesson: the robot does not see a picture,
    it sees a grid of numbers. The falling drops use the film's own code --
    mirrored half-width katakana plus a few digits and symbols -- when a font
    with katakana is on the machine (macOS ships Hiragino), and fall back to
    plain digits otherwise. Only the glyphs are lit: each stroke takes the
    brightness of the pixels under it, and a soft phosphor glow around them
    keeps the user and the poster recognisable from 3 m. An optional `ghost`
    blends a faint copy of the real frame in behind the code as well.

    Rendered on a fixed grid at panel resolution (~9 ms), then scaled back to
    the frame size so the boxes and labels drawn afterwards keep their
    coordinates and stay in full colour on top.
    """

    CW, CH = 8, 12                     # glyph cell, px at the 800x450 panel
    GLITCH_S = 1.2                     # how long the "lost you" glitch lasts
    # The film's rain: half-width katakana (drawn mirrored), digits, symbols.
    RAIN = ("ｦｱｳｴｵｶｷｹｺｻｼｽｾｿﾀﾂﾃﾅﾆﾇﾈﾊﾋﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾜ"
            "0123456789Z:.\"=*+-<>¦|")
    FONTS = ("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",   # bold reads from afar
             "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
             "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
             "/Library/Fonts/Arial Unicode.ttf")

    def __init__(self, w: int = 800, h: int = 450, seed: int = 0,
                 font: str | None = None, ghost: float = 0.0,
                 stroke_detail: bool = True, glow: float = 1.2) -> None:
        self.cols, self.rows = w // self.CW, h // self.CH
        self.W, self.H = self.cols * self.CW, self.rows * self.CH
        self._rng = np.random.default_rng(seed)
        self._atlas = self._build_atlas(font)
        r = self._rng
        self._head = r.uniform(-self.rows, self.rows, self.cols)
        self._speed = r.uniform(0.3, 1.0, self.cols)
        self._trail = r.uniform(8, 25, self.cols)
        self._noise = self._rain_glyphs((self.rows, self.cols))
        self._rowix = np.arange(self.rows, dtype=np.float32)[:, None]
        self._ghost_tint = np.array([0.3, 1.0, 0.3], np.float32)
        self.ghost = ghost                   # faint copy of the frame behind the code
        self.stroke_detail = stroke_detail   # light strokes by per-pixel brightness
        self.glow = glow                     # phosphor halo around each glyph
        self._glitch_box: tuple[int, int, int, int] | None = None
        self._glitch_until = 0.0

    def _build_atlas(self, font: str | None) -> np.ndarray:
        """Glyphs 0-9 are the readable brightness digits; the rest is rain."""
        path = next((p for p in (font, *self.FONTS) if p and os.path.exists(p)),
                    None)
        if path is None:
            log.warning("no katakana font found; matrix rain uses digits only")
            atlas = np.zeros((10, self.CH, self.CW), np.float32)
            for i in range(10):
                cell = np.zeros((self.CH, self.CW), np.uint8)
                cv2.putText(cell, str(i), (1, self.CH - 2),
                            cv2.FONT_HERSHEY_PLAIN, 0.75, 255, 1, cv2.LINE_AA)
                atlas[i] = cell / 255.0
            return atlas

        pil = ImageFont.truetype(path, self.CH - 1)
        chars = [(c, False) for c in "0123456789"] + [(c, True) for c in self.RAIN]
        atlas = np.zeros((len(chars), self.CH, self.CW), np.float32)
        for i, (c, mirror) in enumerate(chars):
            im = Image.new("L", (self.CW * 2, self.CH * 2), 0)
            ImageDraw.Draw(im).text((self.CW, self.CH), c, 255, font=pil,
                                    anchor="mm")
            g = np.asarray(im)[self.CH // 2:self.CH // 2 + self.CH,
                               self.CW // 2:self.CW // 2 + self.CW]
            atlas[i] = (g[:, ::-1] if mirror else g) / 255.0
        return atlas

    def _rain_glyphs(self, shape) -> np.ndarray:
        n = len(self._atlas) - 10
        if n <= 0:                                   # digit-only fallback
            return self._rng.integers(0, 10, shape)
        return 10 + self._rng.integers(0, n, shape)

    def glitch(self, box_frac: tuple[float, float, float, float],
               now: float) -> None:
        """Scramble the cells where the user just vanished, box in 0..1 units."""
        x1, y1, x2, y2 = box_frac
        self._glitch_box = (int(np.clip(x1, 0, 1) * self.cols),
                            int(np.clip(y1, 0, 1) * self.rows),
                            int(np.ceil(np.clip(x2, 0, 1) * self.cols)),
                            int(np.ceil(np.clip(y2, 0, 1) * self.rows)))
        self._glitch_until = now + self.GLITCH_S

    def render(self, bgr: np.ndarray, now: float) -> np.ndarray:
        """Return the rain version of `bgr`, same size as the input."""
        oh, ow = bgr.shape[:2]
        img = cv2.resize(bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(grey, (self.cols, self.rows),
                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        edges = cv2.resize(cv2.Canny(grey, 60, 140), (self.cols, self.rows),
                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255

        # a few cells flicker to a new random digit each frame
        r = self._rng
        flip = r.random(self._noise.shape) < 0.03
        self._noise[flip] = self._rain_glyphs(int(flip.sum()))

        # rain: bright head, fading trail, one drop per column
        self._head += self._speed
        wrap = self._head - self._trail > self.rows
        self._head[wrap] = -r.uniform(0, self.rows, int(wrap.sum()))
        d = self._head[None, :] - self._rowix
        rain = np.where((d >= 0) & (d < self._trail), 1 - d / self._trail, 0.0)
        heads = (d >= 0) & (d < 1)

        digit = np.minimum((small * 10).astype(np.intp), 9)
        glyph = np.where(rain > 0.6, self._noise, digit)
        inten = 0.08 + 1.1 * small ** 1.3 + 0.9 * edges
        inten = inten * (0.7 + 0.5 * rain) + 0.35 * rain

        red = None
        left = self._glitch_until - now
        if self._glitch_box is not None and left > 0:
            gx1, gy1, gx2, gy2 = self._glitch_box
            k = left / self.GLITCH_S                  # 1 -> 0 as it fades
            region = (slice(gy1, gy2), slice(gx1, gx2))
            glyph[region] = self._rain_glyphs(glyph[region].shape)
            inten[region] = np.maximum(inten[region], 0.9 * k)
            red = np.zeros((self.rows, self.cols), np.float32)
            red[region] = k

        tiles = self._atlas[glyph]                    # rows, cols, CH, CW
        mono = (tiles * inten[..., None, None]).transpose(0, 2, 1, 3) \
            .reshape(self.H, self.W)
        if self.stroke_detail:
            px = grey.astype(np.float32) / 255
            mono *= 0.35 + 1.0 * px ** 0.9
        if self.glow > 0:
            mono += self.glow * cv2.GaussianBlur(mono, (0, 0), 1.6)
        out = np.empty((self.H, self.W, 3), np.float32)
        out[..., 1] = mono
        out[..., 0] = out[..., 2] = mono * 0.25
        hm = np.repeat(np.repeat(heads, self.CH, 0), self.CW, 1)
        out[hm] = mono[hm, None]                      # white drop heads
        if red is not None:
            rm = np.repeat(np.repeat(red, self.CH, 0), self.CW, 1)
            out[..., 2] = np.maximum(out[..., 2], mono * rm)
            out[..., 1] *= 1 - 0.8 * rm
        if self.ghost > 0:
            out = 0.7 * out + self.ghost * (img.astype(np.float32) / 255) \
                * self._ghost_tint
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return cv2.resize(out, (ow, oh), interpolation=cv2.INTER_NEAREST)


class Branding:
    """Event name and institutional logos, each placed in its own corner.

    No plate behind them: supply artwork that reads on a dark background (the
    white variant of a dark logo). A missing file is skipped with a warning
    rather than leaving a hole, so the demo runs whether or not the artwork has
    arrived.
    """

    def __init__(self, entries: list[dict], text: str = "",
                 default_height: int = 56) -> None:
        self.text = text
        self.logos: list[tuple[str, np.ndarray]] = []
        for e in entries:
            path = e.get("path", "")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                log.warning("logo missing: %s", path)
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            h = int(e.get("height", default_height))
            w = max(1, int(img.shape[1] * h / img.shape[0]))
            self.logos.append((str(e.get("pos", "top-left")),
                               cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)))

    def draw(self, img, anchors: dict[str, tuple[int, int, str]]) -> None:
        """`anchors` maps a position name to (x, y, blit anchor)."""
        for pos, logo in self.logos:
            spot = anchors.get(pos)
            if spot is None:
                continue
            x, y, anchor = spot
            alpha_blit(img, logo, x, y, anchor=anchor)

    def draw_text_at(self, img, x: int, y: int, size: int,
                     color=(245, 245, 245), anchor: str = "rm") -> None:
        if self.text:
            draw_text(img, self.text, x, y, size, color, anchor, 4)


# --------------------------------------------------------------------------
# segmentation drawing
# --------------------------------------------------------------------------

def color_lut(colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """1x256x3 BGR lookup for cv2.LUT: label value -> colour (unlisted: black)."""
    lut = np.zeros((1, 256, 3), np.uint8)
    for c, bgr in colors.items():
        lut[0, int(c)] = bgr
    return lut


def paint_classes(base: np.ndarray, labels: np.ndarray, lut: np.ndarray,
                  view: str, alpha: float = 0.5, none: int = 255) -> np.ndarray:
    """Render the class map onto `base` (both panel-sized). Returns a new image.

    overlay -- the camera, every labelled pixel tinted with its class colour
    mask    -- flat class colours on a dark ground: the textbook output
    matrix  -- `base` is already the Matrix rain; recolour the glyphs that fall
               on a class, so the numbers themselves carry the label

    All OpenCV calls (LUT, copyTo): numpy fancy indexing at 800x450 cost ~10 ms
    a call, and this runs up to twice a frame.
    """
    col = cv2.LUT(cv2.merge([labels, labels, labels]), lut)
    hit = (labels != none).astype(np.uint8)
    if view == "mask":
        out = np.full_like(base, 24)
        cv2.copyTo(col, hit, out)
        return out
    out = base.copy()
    if view == "matrix":
        lum = cv2.max(cv2.max(base[..., 0], base[..., 1]), base[..., 2])
        gain = cv2.addWeighted(lum, 1.1, lum, 0.0, 64)          # 0.25 + 1.1*lum
        tint = cv2.multiply(col, cv2.merge([gain, gain, gain]), scale=1 / 255)
        cv2.copyTo(tint, hit, out)
        return out
    mix = cv2.addWeighted(base, 1.0 - alpha, col, alpha, 0)
    cv2.copyTo(mix, hit, out)
    return out


def outline(img: np.ndarray, mask: np.ndarray, color, thickness: int = 2) -> None:
    """Draw the boundary of a bool mask (any size: scaled to `img`)."""
    m = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return
    sx, sy = img.shape[1] / m.shape[1], img.shape[0] / m.shape[0]
    cs = [np.round(c.astype(np.float32) * (sx, sy)).astype(np.int32) for c in cs]
    cv2.polylines(img, cs, True, color, thickness, cv2.LINE_AA)


class SignalChart:
    """A scrolling oscilloscope of the network outputs over the last N seconds.

    Two confidence traces on a 0-1 axis: the person class from the segmenter,
    and the class poster A drives in the trigger network (labelled "Adversarial
    Patch"). The threshold is a dashed line, and the stretches where the effect
    was on are shaded -- so the room can see the patch trace spike, the effect
    follow, and the person trace *not* drop.
    """

    def __init__(self, seconds: float = 10.0) -> None:
        self.seconds = seconds
        self._t: list[float] = []
        self._v: list[tuple[float, float, float]] = []

    def push(self, t: float, person: float, neuron: float, effect: float) -> None:
        self._t.append(t)
        self._v.append((person, neuron, effect))
        while self._t and t - self._t[0] > self.seconds + 0.5:
            self._t.pop(0)
            self._v.pop(0)

    def draw(self, img, x, y, w, h, colors: dict, threshold: float) -> None:
        cv2.rectangle(img, (x, y), (x + w, y + h), (34, 30, 28), -1)
        for q in (0.25, 0.5, 0.75):                            # grid
            gy = int(y + h - q * h)
            cv2.line(img, (x, gy), (x + w, gy), (56, 50, 46), 1, cv2.LINE_AA)
        for q in (0.0, 0.5, 1.0):
            draw_text(img, f"{q:.1f}", x - 8, int(y + h - q * h), 18,
                      (150, 150, 150), "rm", 0, weight=400)
        if self._t:
            now = self._t[-1]
            t = np.array(self._t)
            v = np.array(self._v, np.float32)
            px = (x + w - (now - t) / self.seconds * w).astype(np.int32)
            keep = px >= x
            px, v = px[keep], v[keep]
            # shade where the effect was on
            on = v[:, 2] > 0.02
            if on.any():
                band = img[y:y + h, x:x + w]
                shade = band.copy()
                for k in np.flatnonzero(on):
                    x0 = px[k] - x
                    x1 = (px[k + 1] - x) if k + 1 < len(px) else w
                    cv2.rectangle(shade, (int(x0), 0), (int(x1), h),
                                  colors["effect"], -1)
                cv2.addWeighted(shade, 0.22, band, 0.78, 0, band)
            for col, key in ((1, "neuron"), (0, "person")):
                py = (y + h - np.clip(v[:, col], 0, 1) * h).astype(np.int32)
                pts = np.stack([px, py], 1).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(img, [pts], False, colors[key], 3, cv2.LINE_AA)
        ty = int(y + h - threshold * h)
        for xx in range(x, x + w, 14):
            cv2.line(img, (xx, ty), (min(xx + 7, x + w), ty), (235, 235, 235), 1,
                     cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 84, 80), 1)
