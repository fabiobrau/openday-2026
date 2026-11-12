# Third-party assets

| Asset | Source | Licence |
|---|---|---|
| `assets/fonts/SpaceGrotesk.ttf` | [Google Fonts – Space Grotesk](https://github.com/google/fonts/tree/main/ofl/spacegrotesk) (variable, wght 300-700) | SIL Open Font License 1.1 (`assets/fonts/SpaceGrotesk-OFL.txt`) |
| `assets/decoys/dog_photo.jpg` | "Puppy dog face" by **Ian Phillips** (Flickr `PLA_25`), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Puppy_dog_face.jpg). Cropped to the head. | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
| `assets/patches/exp*.png` | [Bimo99B9/NaturalisticAdversarialPatches](https://github.com/Bimo99B9/NaturalisticAdversarialPatches) | MIT. Kept only as the visual *anchor* for training and for comparison — see `docs/PATCH_NOTES.md`. |
| `assets/models/*.pt` | [Ultralytics](https://github.com/ultralytics/assets) releases | AGPL-3.0 |
| `assets/testimg/coco_*.jpg` (not committed; `tools/fetch_assets.py`) | COCO 2017 via Ultralytics' `coco128` sample. Used only to stage the synthetic test scene. | [COCO terms](https://cocodataset.org/#termsofuse): images under their Flickr licences |

Generated here, no third-party rights: `assets/decoys/noise.png`,
`assets/patches/magic_dog.png`, the PDFs in `print/`.

**Note on Ultralytics' AGPL-3.0:** this demo is a non-distributed local
installation. If you ever redistribute it or serve it over a network, read the
licence terms first.
