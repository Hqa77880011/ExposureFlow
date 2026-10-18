# ExposureFlow

Implementation of **ExposureFlow: Exposure-Integrated and Clock-Offset-Aware Continuous-Time Optical Flow from Events and RGB Frames**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, torch 2.7.1+cu118, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Obtain BlinkVision, MVSEC or DSEC-Flow from their dataset distributions. Keep complete source sequences in one split. For MVSEC use the paper's indoor_flying1–2 training, indoor_flying3 validation and indoor_flying4/outdoor_day1 testing protocol. DSEC images require geometric registration to the rectified event view and explicit exclusion of unresolved parallax/occlusion. Registration uses released calibration and permitted depth; it is performed before creating input windows.

Export one numeric NPZ per window:

| Key | Shape | Definition |
| --- | --- | --- |
| `frames` | `2,H,W,3` | RGB in `[0,1]` or `[0,255]`, both on the event image plane |
| `exposure` | `2,2` | Exposure start/end in seconds on the frame clock |
| `events` | `N,4` | Rectified `x,y,event-clock seconds,polarity`; polarity `-1/+1` |
| `event_pixel_id` | `N` | Original sensor pixel ID; required for subpixel rectification |
| `reference` | scalar | Frame-clock origin of every flow target; defaults to first exposure midpoint |
| `flow` | `Q,H,W,2` | Optional reference-to-query displacement in pixels |
| `flow_time` | `Q` | Corresponding absolute frame-clock query seconds |
| `valid` | `Q,H,W` | Optional ground-truth validity mask |
| `offset_gt` | scalar | Optional controlled offset in seconds |
| `exposure_valid` | `2,H,W` | Optional known exposure-rendering validity mask |

Replicate grayscale channels for MVSEC. Export native per-pixel event coordinates after rectification and retain each original pixel's ID so trigger predecessors are never inferred by merging neighboring rectified pixels. Integer co-located event coordinates can omit the ID array. Include a preceding-event margin around the window so adjacent trigger pairs exist; timestamps can extend slightly beyond the exposure bounds. The clock convention is `t_rgb = t_event + offset`. Numeric times must use seconds consistently.

A CSV manifest contains `path,sequence,split`, where split is `train`, `val` or `test`:

```bash
exposureflow prepare --manifest data/windows.csv --output data/prepared
```

Preparation verifies sequence isolation, converts timestamps to the normalized exposure span, derives adjacent same-pixel trigger pairs, and preserves the physical span for converting offsets back to seconds. It does not use test data for learned preprocessing or reconstruct missing flow labels. For the trajectory protocol export labels at normalized times 0.25, 0.50, 0.75 and 1.00, with the same reference as the input archive. The latest labelled time is the endpoint used for endpoint metrics.

## Training and inference

```bash
exposureflow train --data data/prepared --output outputs/full --epochs 30 --width 64 --updates 6 --quadrature 8 --seed 7
```

## Evaluation

```bash
exposureflow evaluate --data data/prepared --model outputs/full/model.pt --output outputs/evaluation --quadrature 16
```

## Comparisons and ablations

```bash
exposureflow train --data data/prepared --output outputs/frame-only --variant frame_only
exposureflow train --data data/prepared --output outputs/midpoint --variant midpoint
exposureflow train --data data/prepared --output outputs/zero --variant fixed_zero
exposureflow train --data data/prepared --output outputs/binned --variant binned_events
exposureflow train --data data/prepared --output outputs/linear --variant linear
exposureflow train --data data/prepared --output outputs/two-fields --variant two_trajectories
exposureflow train --data data/prepared --output outputs/no-scale --variant no_scale
```

Keep seed, splits and query times identical. `stress.py` transforms timing/exposure using the exported numeric window schema.

## Outputs

Training writes `model.pt`, `history.json` and `learning.png`. Evaluation writes per-window controls/flows and endpoint plots; available labels enable `metrics.json` and `risk_coverage.png`. Missing flow/timing labels leave those metrics unavailable.
