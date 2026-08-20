# dreamroom

Furniture replacement pipeline. Current scope: steps 0-4.

- **Step 0** — resize the input so its longest side is 1280 px.
- **Step 1** — select an object: draw polylines in an OpenCV window, segment
  with a local [SimpleClick](https://github.com/uncbiag/SimpleClick) model
  (ViT-Huge, CoCo+LVIS checkpoint), confirm the mask.
- **Step 2** — draw a reference line on an object of known length and enter
  its length in meters to get a px-per-meter scale.
- **Step 3** — send the working image to the MoGe-2 API and receive a point
  map, textured GLB, metadata, depth, and normal previews.
- **Step 4** — resize the confirmed mask to point-map resolution, calibrate
  the MoGe scale from Step 2, fit a floor plane, and generate a floor-aligned
  3D bounding box with 2D/3D debug visualizations.

## Project layout

```text
dreamroom/
├── dreamroom/              # the package
│   ├── config.py           # Settings (paths, thresholds, sizes)
│   ├── image_ops.py        # step 0: load / resize / save
│   ├── segmenter.py        # local SimpleClick port (from simpleclick-modal)
│   ├── geometry3d.py       # step 4: floor plane and 3D box fitting
│   ├── moge_client.py      # step 3: MoGe-2 API client and response parser
│   ├── pipeline/            # ordered stages, context, timing, and outputs
│   │   ├── __init__.py      # FurniturePipeline facade
│   │   ├── models.py        # shared pipeline context/result models
│   │   ├── outputs.py       # output artifact persistence
│   │   └── stages/          # one module per pipeline stage
│   ├── viz3d.py             # step 4: 2D overlay and calibrated GLB export
│   └── ui/
│       ├── window.py       # shared OpenCV window base class
│       ├── strokes.py      # step 1: polylines -> segment -> confirm
│       └── reference.py    # step 2: reference line -> length in meters
├── scripts/
│   ├── setup_simpleclick.sh      # clone repo + install deps + download weights
│   ├── run_pipeline.py           # CLI entry point
│   └── smoke_test_segmenter.py   # non-interactive segmenter test
├── third_party/SimpleClick # cloned by setup (gitignored)
├── weights/                # checkpoint (gitignored)
└── outputs/                # per-run results (gitignored)
```

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/setup_simpleclick.sh   # clones SimpleClick, installs mmcv, downloads ~2.7 GB weights
```

The checkpoint download can also be run directly:

```bash
.venv/bin/gdown 1GXk6q5fwKo2twkY5ZZGjVKCgJv7XeLAW -O weights/cocolvis_vit_huge.pth
```

## Run

```bash
.venv/bin/python scripts/run_pipeline.py --image path/to/room.jpg
```

Options: `--output-dir`, `--max-side`, `--threshold`, `--max-display-width`,
`--no-flip` (about 2x faster segmentation on CPU, slightly lower quality),
`--moge-endpoint`, `--moge-timeout`, `--debug`, and `--skip-moge` (run only
steps 0-2).

### Step 1 controls

| Input | Action |
| --- | --- |
| left-drag | positive stroke (red, on the object) |
| right-drag | negative stroke (blue, on the background) |
| `u` / `c` | undo last stroke / clear all strokes |
| Enter / Space | close the annotation window and run segmentation, then open the confirmation view |
| `y` / Enter | confirm the mask in the confirmation view |
| `n` / `r` | redraw the strokes in a new annotation view |
| Esc / `q` | abort |

### Step 2 controls

| Input | Action |
| --- | --- |
| console input | enter the known reference length in meters before the window opens |
| left-drag | draw the reference line (yellow) |
| Enter | confirm the drawn line |
| `u` / `c` | redraw the line |
| Esc / `q` | abort |

### Step 3: MoGe-2

The client sends a multipart `POST /predict` request. Production mode is the
default and requests `include_mesh=false` and `include_debug=false` to reduce
latency. Use `--debug` to request both flags as `true`, persist the full MoGe
asset set, and generate `debug_3d.glb`. The endpoint can be overridden with
`--moge-endpoint` or `DREAMROOM_MOGE_ENDPOINT`.

MoGe-2 currently clamps the API input to a maximum side of 800 px, while the
working image defaults to 1280 px. The confirmed mask is therefore resized
down to the returned point-map resolution with nearest-neighbor interpolation
before 3D fitting. Point-map coordinates follow the GLB camera convention:
`+X` right, `+Y` up, and `-Z` forward; normalized intrinsics are stored in the
returned metadata.

### Step 4: mask to 3D box

- Reference-line endpoints are sampled in the point map and used to convert
  MoGe units to meters.
- Valid masked points are extracted with light erosion and radial outlier
  clipping.
- Non-object points in the lower half of the image are used for constrained
  RANSAC floor-plane fitting. If the floor cannot be recovered, a camera-up
  fallback plane is used and marked in the output.
- Object points are transformed into the floor frame. A minimum-area 2D
  footprint and robust height percentile produce the oriented box.
- In `--debug` mode, `debug_3d.glb` scales the original MoGe scene with the
  Step 2 calibration factor before adding the fitted box and floor plane,
  keeping all displayed geometry in calibrated metric coordinates.

## Outputs

Each run writes `outputs/<image-name>-<timestamp>/`:

- `image.png` — the resized (max-side 1280) working image.
- `mask.png` — confirmed object mask (`255` = object).
- `overlay.png` — mask preview over the image.
- `selection.json` — click points used and mask area.
- `reference.json` — reference line endpoints, pixel length, meters, px/m.
- `meta.json` — sizes, resize scale (`original_coord = resized_coord * resize_scale`), settings.
- `point_map.npy` — MoGe point map at the API response resolution (debug mode).
- `output.glb` — original MoGe textured scene in native MoGe coordinates (debug mode).
- `moge_metadata.json` — point-map size, camera convention, and normalized intrinsics (debug mode).
- `depth.png` / `normal.png` — MoGe debug previews (debug mode).
- `box3d.json` — calibrated box center, axes, extents, corners, floor plane, and scale correction.
- `debug_2d.png` — fitted floor plane and box reprojected onto the working image.
- `debug_3d.glb` — calibrated MoGe scene with the fitted box and floor plane overlays (debug mode).
- `stats.json` — per-step latency in seconds, output-save time, and total runtime.

The CLI also prints the same latency summary after the output directory is
written. Steps 3 and 4 are reported as `skipped` when `--skip-moge` is used.

## Notes

- The checkpoint is loaded with `torch.load(..., mmap=True)` to keep peak RAM
  near the model size (~2.7 GB). On machines with 8 GB RAM, close heavy apps
  before running; a standard load peaks above 5 GB and can be OOM-killed.
- Measured on a CPU-only 8 GB machine: model load ~1-2 min (one-time per
  process), first segmentation ~38 s, subsequent segmentations on the same
  image ~18 s (image features are cached). Use `--no-flip` to roughly halve
  segmentation time. CUDA is used automatically when available.
- SimpleClick compiles a small Cython extension on first model load; this is
  normal and happens once (cached in `~/.pyxbld`).
- Paths can be overridden with `DREAMROOM_SIMPLECLICK_ROOT` and
  `DREAMROOM_CHECKPOINT`.
- MoGe can be skipped with `--skip-moge`, which is useful for validating the
  interactive Steps 0-2 flow without the API.

## Test

```bash
.venv/bin/python scripts/smoke_test_segmenter.py
.venv/bin/python -m pytest -q tests
```
