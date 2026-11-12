# From SharperNight to Open Day: why there are two networks

This repo is a fork of `sharpernight-2026` (the *INVISIBLE!* science-fair demo
for 6-12 year olds), reworked for secondary-school students: the detector box
became a live **semantic segmentation**, the cartoon parts (sounds, robot
mascot, invisibility timer and scoreboard) are gone, and the bottom of the
screen now shows what the networks actually output.

The printed posters did **not** change. Poster A is still the patch trained in
`sharpernight-2026` against `yolov8n` to light up class 78 (`hair drier`) —
see [`PATCH_NOTES.md`](PATCH_NOTES.md) for how it was made and every dead end on
the way. That one fact shapes the whole design.

## The patch does not transfer to the segmenter

The obvious design is one network: run `yolov8n-seg`, read class 78 from its
own detection head, done. It does not work. A patch is optimised against one
set of weights, and `yolov8n-seg` is a different set — same architecture
family, different training run.

Measured on 12 COCO photos of a large person (Ultralytics' `coco128` sample),
re-cropped to demo framing (`tools/framing.py`), poster pasted at chest height.
Values are mean class-78 confidence / share of images ≥ 0.20. Reproduce with
`tools/validate_transfer.py --models yolov8n,yolov8n-seg,yolo11n-seg`.

| imgsz | poster side | poster | `yolov8n` (trained on) | `yolov8n-seg` | `yolo11n-seg` |
|---|---|---|---|---|---|
| 1280 | 100 px | **A** | 0.33 / 67% | 0.00 / 0% | 0.00 / 0% |
| 1280 | 160 px | **A** | **0.99 / 100%** | 0.00 / 0% | 0.00 / 0% |
| 1280 | 240 px | **A** | **0.83 / 100%** | 0.00 / 0% | 0.00 / 0% |
| 1280 | any | B decoy | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| 640 | 240 px | **A** | 0.83 / 100% | 0.00 / 0% | 0.00 / 0% |
| 640 | ≤ 160 px | **A** | ≤ 0.01 / 0% | 0.00 / 0% | 0.00 / 0% |

Two conclusions:

1. **Two networks.** The segmenter (`yolov8n-seg`, `imgsz: 640`) draws the
   picture; a second small detector (`yolov8n`, `imgsz: 1280`) only reads
   class 78. That is what `trigger.py` is. The config guard
   (`patch.target_model` must equal `trigger.name`) still applies.
2. **The segmenter is not fooled, and that is the lesson.** Its person
   confidence with poster A up was the same as without it (0.73–0.77 in every
   cell above). The effect — vanish or relabel — is the app editing the
   segmentation output after reading a signal. Same honesty rule as
   SharperNight: *the mechanism is real, the effect is staged.* The confidence
   chart makes it visible on its own: when the student vanishes, the magenta
   Adversarial Patch trace jumps and the blue person trace does **not** drop.

The same model-specificity is a teaching point in itself: "the poster is a key
cut for one lock — the segmentation network next to it does not even notice it."

## What changed, file by file

| SharperNight | Open Day | why |
|---|---|---|
| `detector.py` box + `StableTracker` | `segmenter.py` | instance masks flattened to a per-pixel class map |
| trigger code inside `app.py` | `trigger.py` | its own model now; latch in seconds, every N frames |
| box pops, puff particles | `effects.py` | vanish (learned background plate) or relabel |
| largest person only | `pick_holder` | in a group, only whoever holds the poster is edited |
| mascot, speech bubble, sounds | — | removed |
| invisibility timer, top-5 board | — | removed |
| confidence health bar | pixels-per-class bars, confidence chart (person vs Adversarial Patch), status | metrics a 16-year-old can read and question |
| expert mode (`yolov10n`) | — | removed; the transfer table above makes the point |
| virtual ArUco board, `compositor.py` | — | removed; the posters are already printed |
| `--trigger detector` (real evasion) | — | removed; it was about the detector box |
| Matrix rain | kept (`M`) | glyphs now coloured by class |
| Fredoka | Space Grotesk | less playground |

## Who gets edited: `pick_holder`

The trigger's class-78 box *is* the poster. Held at the chest, it sits inside
its holder's person box; a friend standing behind overlaps it less. Each person
instance is scored by the share of the poster box inside its own box, ties
broken by distance to the poster centre. While the effect fades out (poster
lowered), the same person is followed by IoU so the edit does not jump to
someone else mid-fade. `effect.target: all` (key `T`) edits every person
instead.

`tools/validate_demo.py` measures this on a staged three-person group (each of
the three as holder, three layouts, flat and tilted): the right person was
picked in 100 % of firing frames. The README has the full table.

## The vanish effect: a background plate

To erase a person you need to know what is behind them. `BackgroundPlate`
learns it continuously from every pixel that is *not* within 15 px of any
person mask (`cv2.accumulateWeighted` with a mask, ~1 ms at panel size). A
pixel never seen uncovered is in-painted from its neighbours at quarter
resolution — a soft smear rather than a hole.

Consequence for the day: **let the camera see the empty scene for a few
seconds at setup** (the `--show-fps` readout shows `bg NN%`; wait for well
above 70%). It used to be a row on screen; it confused the audience more than
it explained, so it is operator-only now. Press `B` to start over if the camera is moved. If a crowd stands
in front of the camera all afternoon, the plate keeps learning from whatever
patches of the room become visible.

The poster itself is not part of the person mask, so in vanish mode it is left
floating in the air. That is deliberate: the one object the network-reading
trick depends on stays in view.

## Performance

Not yet measured on the Apple Silicon rig — everything above and in the README
was measured on CPU. Budget, from the SharperNight numbers: the trigger at
1280 was ~27 ms on MPS; a nano segmenter at 640 should add ~15–20 ms. With
`trigger.every_n_frames: 2` (default) the trigger cost halves and the 0.5 s
latch hides the gap. Measure with `tools/soak_test.py` before the day, and if
it is short, drop `trigger.imgsz` to 960 (poster then needs ~140 px, i.e.
stand within ~1.4 m with A4).
