# Printing and staging guide

```bash
.venv/bin/python tools/make_print.py --paper A4    # or --paper A3
```

| File | What it is |
|---|---|
| `posters_A4.pdf` | 2 pages, **A** = magic, **B** = ordinary dog |

**The posters printed for SharperNight 2026 are the same files** — reuse them.
Regenerate only for a reprint.

A4 is the default choice: 18 cm square, easy to hold, and good out to 2 m with
`trigger.imgsz: 1280`. Go to A3 only if the holder must stand further back.

The poster pages carry no visible title — only a tiny grey `poster A/B` below
the cut line, for you. **The audience must not be able to read which is which.**

---

## Paper and mounting

- **Matte paper**, mounted on foam board. Matte still helps -- a specular
  highlight blows out part of the pattern -- but this is no longer make-or-break.
  The trigger survived washed-out colours, a dim printer and bad JPEG in testing.
- Print at **100% scale / no fit-to-page**. Both posters must end up exactly
  the same size, or the comparison is not a comparison.

## Holding and standing

- Hold at **chest height**, flat and square to the camera.
- Hold it **against your own body**. In a group, the app edits whoever's
  silhouette contains the poster; held out at arm's length into a friend's
  silhouette, the friend gets edited instead.
- **Distance is the one thing that matters**, and it follows from the paper
  size and `trigger.imgsz`. The poster must be roughly **100 px wide in frame**
  at `imgsz: 1280` (the default), or **140 px** at 960. On a ~60 deg webcam:

  | paper | printed square | works out to (imgsz 1280) | (imgsz 960) |
  |---|---|---|---|
  | A3 | 26 cm | 2.9 m | 2.1 m |
  | **A4 (default)** | **18 cm** | **2.0 m** | 1.4 m |
  | A5 | 13 cm | 1.4 m | 1.0 m |

  Regenerate any of them with `--paper A4` / `--paper A3`.
  Run with `--show-fps`, walk back until the readout stops saying `FIRE`, then
  put **tape on the floor** a step closer. Two minutes, and the flakiest part of
  the demo becomes a mark on the ground.
- Tilt is forgiving up to ~30 deg; beyond that the margin gets thin.
- Motion blur is the weakest case measured, so: hold it still.

## Light

- Even, **frontal** light. No window behind the group — backlit people are
  silhouettes and the segmenter struggles before the poster is even raised.
- Avoid a spotlight directly on the poster: the specular hotspot destroys the
  pattern just like glossy paper does.

## Setting up (five minutes)

1. `python app.py --lang it --show-fps`
2. **Step out of frame for a few seconds.** The status column shows
   "background learned"; wait until it is well above 70 %. (Press `B` to
   restart this if you move the camera later.)
3. Stand on the intended spot and raise poster **A**. The readout must go
   green: `cls78 0.9x FIRE`, and you vanish from the right-hand panel.
4. Step back until it stops firing, then forward one step. **Tape the floor.**
5. Sanity-check with poster **B**: it must stay at `cls78 0.00` and nothing
   must change. If B ever fires, stop and tell whoever built this.
6. Try it with two friends either side: only the holder must be edited.
7. Turn `--show-fps` off for the actual demo.

## Spares

Print **two copies of poster A**. It is the one a hundred students will handle,
and a creased, finger-marked magic poster is a demo that quietly stops working
halfway through the afternoon.
