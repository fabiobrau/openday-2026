"""Every user-facing string in the demo, in Italian and English.

Nothing that appears on screen is written inline in the drawing code: the open
day is Italian-speaking, and the operator flips the whole interface with
`--lang`.

The audience is a group of 14-19 year olds crowded round the monitor: plain,
correct vocabulary (segmentation, class, confidence, threshold, neuron), no
baby talk, and every line short enough to read from two metres.

Strings with {placeholders} are templates; fill them with str.format.
"""
from __future__ import annotations

LANGUAGES = ("it", "en")
DEFAULT_LANG = "it"

_STRINGS: dict[str, dict[str, str]] = {
    "it": {
        # --- panel headings -------------------------------------------------
        "panel_camera": "TELECAMERA",
        "panel_model": "SEGMENTAZIONE SEMANTICA",
        "view_overlay": "vista: sovrapposizione",
        "view_mask": "vista: mappa delle classi",
        "view_matrix": "vista: numeri",
        # --- metrics: classes -----------------------------------------------
        "classes_title": "PIXEL PER CLASSE",
        "classes_unit": "dell'immagine",
        "instances": "{n} oggetti segmentati",
        # --- metrics: chart -------------------------------------------------
        "chart_title": "CONFIDENZA PER CLASSE · ULTIMI {s} s",
        "chart_person": "persona",
        "chart_neuron": "Adversarial Patch",
        "chart_threshold": "soglia",
        "chart_effect": "effetto attivo",
        # --- metrics: status ------------------------------------------------
        "status_trigger": "TRIGGER",
        "trigger_on": "ATTIVO",
        "trigger_off": "inattivo",
        "status_effect": "EFFETTO",
        "effect_vanish": "sparizione",
        "effect_relabel": "cambio di classe → {cls}",
        "status_target": "CLASSE BERSAGLIO",
        "status_speed": "LATENZA",
        "speed_line": "seg {seg:.0f} ms · trigger {trg:.0f} ms · {fps:.0f} fps",
        "manual": "MANUALE",
        # --- footer ---------------------------------------------------------
        "privacy": "Nessuna immagine viene salvata: tutto avviene in memoria, "
                   "in tempo reale.",
        "keys_help": "M vista   E effetto   F schermo intero   ESC esci",
        "mode_fallback": "VIDEO REGISTRATO",
        # --- trouble --------------------------------------------------------
        "camera_lost": "Telecamera non disponibile",
        "camera_retry": "Riprovo a collegarmi...",
    },
    "en": {
        "panel_camera": "CAMERA",
        "panel_model": "SEMANTIC SEGMENTATION",
        "view_overlay": "view: overlay",
        "view_mask": "view: class map",
        "view_matrix": "view: numbers",
        "classes_title": "PIXELS PER CLASS",
        "classes_unit": "of the image",
        "instances": "{n} objects segmented",
        "chart_title": "CONFIDENCE PER CLASS · LAST {s} s",
        "chart_person": "person",
        "chart_neuron": "Adversarial Patch",
        "chart_threshold": "threshold",
        "chart_effect": "effect on",
        "status_trigger": "TRIGGER",
        "trigger_on": "ON",
        "trigger_off": "off",
        "status_effect": "EFFECT",
        "effect_vanish": "vanish",
        "effect_relabel": "class swap → {cls}",
        "status_target": "TARGET CLASS",
        "status_speed": "LATENCY",
        "speed_line": "seg {seg:.0f} ms · trigger {trg:.0f} ms · {fps:.0f} fps",
        "manual": "MANUAL",
        "privacy": "No image is ever saved: everything happens in memory, "
                   "in real time.",
        "keys_help": "M view   E effect   F fullscreen   ESC quit",
        "mode_fallback": "RECORDED VIDEO",
        "camera_lost": "Camera unavailable",
        "camera_retry": "Trying to reconnect...",
    },
}

# COCO class names for the classes the demo shows, plus the trigger class.
# Anything not listed falls back to the model's own English name.
_CLASSES: dict[str, dict[int, str]] = {
    "it": {
        0: "persona", 24: "zaino", 26: "borsa", 39: "bottiglia", 41: "tazza",
        56: "sedia", 63: "portatile", 67: "cellulare", 73: "libro",
        78: "asciugacapelli",
    },
    "en": {
        0: "person", 24: "backpack", 26: "handbag", 39: "bottle", 41: "cup",
        56: "chair", 63: "laptop", 67: "cell phone", 73: "book",
        78: "hair drier",
    },
}


class Strings:
    """Tiny lookup object so drawing code reads `S.get('panel_model')`."""

    def __init__(self, lang: str = DEFAULT_LANG,
                 model_names: dict[int, str] | None = None):
        if lang not in _STRINGS:
            lang = DEFAULT_LANG
        self.lang = lang
        self._d = _STRINGS[lang]
        self._model_names = model_names or {}

    def get(self, key: str, **kw) -> str:
        # A missing key must never crash the demo mid-show.
        s = self._d.get(key, _STRINGS[DEFAULT_LANG].get(key, key))
        if kw:
            try:
                return s.format(**kw)
            except (KeyError, IndexError, ValueError):
                return s
        return s

    __call__ = get

    def cls(self, c: int) -> str:
        return _CLASSES[self.lang].get(c) or self._model_names.get(c, str(c))
