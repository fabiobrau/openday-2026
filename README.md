# Open Day 2026 — what does a segmentation network see?

A live demo for secondary-school students. A **semantic segmentation** network
labels every pixel of the camera feed — person, backpack, phone, bottle — while
a group crowds round the monitor. One of them raises a printed poster and
**vanishes from the network's output**, or **turns into another class**, while
their friends stay exactly as they were.

Then the honest part: the bottom of the screen plots what the networks really
output, and it shows the segmenter never stopped seeing them.

Forked from [`sharpernight-2026`](https://github.com/fabiobrau/sharpernight-2026)
(the *INVISIBLE!* demo for 6-12 year olds). **Same printed posters**; the
detector box became a segmentation, and the cartoon parts — sounds, robot
mascot, invisibility timer and scoreboard — are gone.

Runs on an Apple Silicon Mac, **offline**, and **writes nothing to disk**.

```bash
python app.py --lang it
```

<!-- screenshot: run tools/render_test.py and drop it_3_vanish_A.png here -->

---

## 1. Quick start

Tested with Python 3.12–3.13; the SharperNight rig ran macOS 15 on Apple Silicon.

```bash
git clone <this repo> && cd openday-2026

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # ~2 GB, mostly torch

.venv/bin/python tools/fetch_assets.py           # weights (~20 MB) + test scene
.venv/bin/python tools/render_test.py --out /tmp/screens   # sanity check, no camera

.venv/bin/python app.py --lang it
```

**Camera permission:** the first launch asks for it. Until you click Allow the
app runs the fallback loop (if built) and reconnects on its own afterwards. Do
it once before the doors open.

**Do not leave the weights to the day itself.** Without `fetch_assets.py`,
Ultralytics tries to download at first use, and the venue wifi will fail you.

**At setup, let the camera see the empty scene for a few seconds** — that is
how the vanish effect learns what is behind people (§5).

### Keys

| Key | Action |
|---|---|
| `M` | right-hand view: overlay → class map → numbers (Matrix) |
| `E` | effect: vanish ⇄ change class |
| `B` | forget the learned background (after moving the camera) — operator only |
| `T` | target: whoever holds the poster ⇄ every person — operator only |
| `V` | manual override: apply the effect without a poster (operator only) |
| `F` | fullscreen |
| `ESC` | quit |

The on-screen key hint shows only `M`, `E`, `F`, `ESC`; the others are for the
operator and are not advertised.

Flags: `--camera 1` (the index is **never** guessed), `--show-fps` (adds an
operator readout, `cls78 0.97 FIRE   bg 87%   target holder` — use it to find
the working distance and to check the background is learned),
`--windowed`, `--lang en`, `--view mask|matrix`, `--effect relabel`,
`--fallback`.

---

## 2. The posters — the actual point

| **A — the magic poster** | **B — the decoy** |
|---|---|
| <img src="docs/img/poster_a.png" width="300"> | <img src="docs/img/poster_b.png" width="300"> |
| Optimised to drive one output neuron of a detector. The holder vanishes. | An ordinary photo of a very similar dog. Nothing happens. |

These are the posters already printed for SharperNight: same files, same
sizes, same staging (`print/README.md`). Hand both out at identical size and
let the group guess which one works. B is the control that turns "you just hid
behind cardboard" into "it is one specific pattern, built for one specific
network". It measured `0.000` on the trigger class everywhere, at every size.

To reprint: `.venv/bin/python tools/make_print.py --paper A4` → two pages,
A then B. Print at 100 %; the square must measure **18 cm** on A4.

## 3. What is on screen

```
+---------------------------+-------------------------------+
|  CAMERA                   |  SEMANTIC SEGMENTATION        |
|  (clean, mirrored)        |  (class per pixel + labels)   |
+---------------+-----------+--------------+----------------+
| PIXELS PER    | CONFIDENCE PER CLASS, 10 s | TRIGGER        |
| CLASS         |  person ─────              | EFFECT         |
|  person 31%   |  Adversarial Patch ─────   | TARGET CLASS:  |
|  phone 0.4%   |  threshold - - -           |   person       |
|  ...          |  (shaded = effect on)      | LATENCY        |
+---------------+----------------------------+----------------+
```

**Right panel, three views (`M`):**

- **overlay** — the camera with every labelled pixel tinted by its class,
  outlined, and labelled `class confidence%`.
- **class map** — flat colours on black: the textbook output of a
  segmentation network, and the easiest way to explain "one class per pixel".
- **numbers** — the Matrix rain from SharperNight: the frame as a grid of
  brightness digits, now coloured by class. "The network never sees a
  picture, only numbers."

**The classes** are deliberately few — what a group of teenagers actually
brings: person, backpack, handbag, bottle, chair, laptop, cell phone
(`segmentation.classes` in `config.yaml`; any COCO id works, give it a colour
under `ui.class_colors`).

**The metrics**, and the question each one invites:

| metric | what it is | question for the room |
|---|---|---|
| pixels per class | share of the frame painted with each class, as shown | "why is *person* 30 % and *phone* 0.4 %?" (square-root bar scale) |
| confidence: person | the segmenter's confidence on the targeted (or largest) person | "you vanished — why didn't this line drop?" |
| confidence: Adversarial Patch | the class-78 (`hair drier`) score the poster drives in the trigger network | "what does the poster actually change?" |
| threshold | 0.20, dashed | "what if we set it to 0.9? to 0.01?" |
| target class | the class the effect acts on: person | "could a patch target any class?" |
| latency | ms per network, frames per second | "why two networks? why is one slower?" |

The background plate that makes vanishing possible is deliberately not on
screen (it confused more than it explained); its coverage is in the
`--show-fps` operator readout.

## 4. How it works — read this first

### Two networks

```
camera ─┬─► yolov8n-seg @640  ─► instance masks ─► class-per-pixel map ─┐
        │                                                               ├─► screen
        └─► yolov8n @1280 ─► class 78 "hair drier" ─► trigger ─► pick holder ─┘
```

Poster A was optimised (in `sharpernight-2026`) to make **one output neuron of
`yolov8n`** — class 78, `hair drier` — fire as hard as possible. It does
nothing to the segmenter: measured 0.00 on `yolov8n-seg` and `yolo11n-seg`, at
every size (`docs/OPENDAY_NOTES.md`). So the segmenter draws the picture, and a
second small detector only reads that neuron.

### This simulates an adversarial attack. It is not one.

When class 78 fires, the **app** finds the person holding the poster and edits
the segmentation output: erases them, or repaints them as another class. The
segmentation network is not fooled for a single frame; its person confidence
stays where it was, and the chart shows exactly that.

So: the **effect** is staged, the **mechanism** is real. The poster is not a
trick picture or a hard-coded `if`: it is a gradient-optimised, model-specific
pattern that genuinely controls a network's output — the same machinery
(EOT, printability, TV loss) a real attack uses. What it does not do is blind
anybody.

With high-schoolers this is the best part of the demo, not a disclaimer, and
the confidence chart carries it on its own: when the student vanishes, the
Adversarial Patch trace jumps above the threshold and the person trace does
**not** drop. Questions it opens:

- *Why does the poster work on one network and not the one next to it?*
  (model-specificity; the patch is a key cut for one lock)
- *Could a poster make the segmenter itself miss you?* (yes in principle — that
  is a real evasion attack; SharperNight measured ~30 % success against a
  detector, see `docs/PATCH_NOTES.md`)
- *Where would that be dangerous?* (cars, surveillance, medical imaging)

The startup log always states it:

```
WARNING openday: TRIGGER: the effect fires when class 78 of yolov8n rises
above 0.20 -- the segmenter is not being fooled.
```

### The guard

`config.yaml` records `patch.target_model`; if it disagrees with
`trigger.name` the app **refuses to start** rather than silently doing nothing.

## 5. The two effects (`E`)

**Vanish.** The holder's pixels are replaced by the background, and their mask
leaves the class map and the pixel counts. The background is a plate learned
continuously from every pixel nobody is standing on; anything never seen
uncovered is in-painted from its neighbours (a soft smear). The `--show-fps`
readout shows how much of the scene is learned (`bg NN%`). **At setup, keep
the scene empty for a few seconds** until it is well above 70 %; press `B`
after moving the camera. The
poster stays floating in the air on purpose.

**Change class.** The mask stays put but is painted, labelled and counted as
class 78, `hair drier` — the class the poster really does light up, just on the
other network. Change it with `effect.relabel_to`.

Both ease in and out over 0.35 s, and in a group only **the person holding the
poster** is edited: the class-78 box is the poster, and the person whose box
contains most of it is the holder (`effects.pick_holder`). `T` switches to
editing everyone.

## 6. Operating point

Unchanged from SharperNight, because the trigger is the same network at the
same input size. The poster must be big enough **in the trigger's input**:

| `trigger.imgsz` | fires from | A4 (18 cm) works to | A3 (26 cm) |
|---|---|---|---|
| 960 | ~140 px | 1.4 m | 2.1 m |
| **1280 (default)** | **~100–160 px** | **~2.0 m** | **~2.9 m** |

Distances assume a ~60° webcam. Find yours with `--show-fps` (raise A, walk
back until `FIRE` stops, step forward, **tape the floor**). The full staging
guide is in [`print/README.md`](print/README.md).

For a group around a monitor: the camera on top of the screen, the holder on
the tape mark, friends either side or behind. Everyone in frame is segmented;
only the holder is edited.

## 7. Privacy

Nothing is written to disk. No `imwrite` in the capture path, no logs of
frames, no state that outlives the process. The footer says so on screen.

## 8. Repo layout

```
app.py            main loop, camera, layout, keys
segmenter.py      YOLOv8n-seg wrapper: instance masks -> class-per-pixel map
trigger.py        reads class 78 of yolov8n; latch, every-N-frames
effects.py        background plate, vanish/relabel, pick_holder
overlay.py        text, class painting, confidence chart, Matrix view, logos
strings.py        ALL on-screen text, IT + EN, class names
config.yaml
tools/            fetch_assets.py   render_test.py   soak_test.py
                  validate_demo.py  validate_transfer.py   scene.py
                  make_print.py     make_fallback.py   make_assets.py
                  train_patch.py    validate_patch.py   framing.py   <- how poster A
                                                                     was made (SharperNight)
assets/           patches/ decoys/ fonts/ logos/ models/ testimg/ fallback/
print/            README.md (+ generated posters_*.pdf)
docs/             OPENDAY_NOTES.md (this fork)  PATCH_NOTES.md (the patch)
```

## 9. Troubleshooting

Run with `--show-fps` first: the readout tells you whether the trigger fires.

| Symptom | Fix |
|---|---|
| `cls78 0.00` with poster A up | **Distance.** ~100 px wide in frame at `imgsz: 1280`. Step closer until `FIRE`, tape the floor. |
| Trigger fires but a friend vanishes | The holder must hold the poster against their own body; a poster held out into a neighbour's box gets attributed to them. Or press `T` to edit everyone. |
| Vanished person leaves a smeary ghost | The background behind them was never seen. Clear the scene for 3 s, or press `B` and do it again. |
| Background "learned" a person | Someone stood still for minutes below the segmenter's threshold. Press `B` with the scene empty. |
| **Poster B fires the trigger** | Should be impossible — it measured `0.000` everywhere. Stop and investigate rather than raising the threshold. |
| App refuses to start, "optimised against X" | `trigger.name` and `patch.target_model` disagree in `config.yaml`. |
| Slow | Trigger every 2nd frame is the default; try `trigger.imgsz: 960` (shorter range) or `segmentation.imgsz: 480`. |
| Wrong camera | `--camera 1`. Indices are never guessed. |
| `CERTIFICATE_VERIFY_FAILED` from `fetch_assets.py` | Handled: certifi, then `curl`. If both fail you are offline. |

## 10. Status

Measured, not assumed. Every row names the command that reproduces it.

| Criterion | Status |
|---|---|
| Poster A transfers to the segmenter | **No** — 0.00 on `yolov8n-seg` and `yolo11n-seg` at every size; hence the separate trigger (`docs/OPENDAY_NOTES.md`) |
| Trigger fires on A, never on B or nothing | **Yes** — A fires 100 % from 140 px up (67 % at 100 px); B and an empty hand 0 %, max score 0.00 (`tools/validate_demo.py`) |
| Effect lands on the holder, not a friend | **Yes** — 100 % of firing frames, 3-person group, every holder position, 0° and tilted (`tools/validate_demo.py`) |
| Segmenter still sees the holder with A up | **Yes** — 100 %: the effect is the app's, not the network's (`tools/validate_demo.py`) |
| Runs unattended, no frame raises | **Yes** — 1200-frame soak through every view, effect and target, dropped frames included, 0 errors (`tools/soak_test.py`) |
| …without memory growth | **On CPU, yes** — +42 MB high-water over 1200 frames, levelling off (656.0 → 656.1 MB over the last 300). MPS has its own allocator: run one long soak on the rig. |
| ≥ 20 fps on Apple Silicon | **Not yet measured on MPS.** Everything here was measured in a CPU-only container. Run `tools/soak_test.py` on the rig; drawing alone is 20–45 ms/frame on a 2-core cloud CPU, so expect ~3× less on an M-series. |
| Nothing written to disk | **Yes** — no `imwrite` in the capture path |
| One command to start | **Yes** — `python app.py --lang it` |

`tools/validate_demo.py`, staged three-person group (CPU, `trigger.imgsz: 1280`):

```
  poster     side    fires   holder  seg kept  trigger score
  A magic     100      67%     100%      100%    0.32 (max 0.58)
  A magic     140     100%     100%      100%    0.98 (max 0.99)
  A magic     200     100%     100%      100%    0.97 (max 0.99)
  A magic     260     100%     100%      100%    0.54 (max 0.84)
  B decoy   100-260     0%        -   61-100%    0.00 (max 0.00)
  none          -       0%        -      100%    0.00 (max 0.00)
```

(The decoy's "seg kept" drops to 61 % at 260 px — most likely plain occlusion
breaking up the holder's mask, though not investigated. Poster A at the same
size stays at 100 %.)
