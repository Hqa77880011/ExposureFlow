# ExposureFlow

Implementation of **ExposureFlow: Exposure-Integrated and Clock-Offset-Aware Continuous-Time Optical Flow from Events and RGB Frames**.

ExposureFlow predicts cubic Bernstein displacement fields and a reference radiance image. The same motion fields explain timestamped event contrast and finite-exposure frame formation. A bounded clock estimate shifts event times after feature encoding; the output also includes a positive flow-error scale.

## Installation

Use Python 3.11+ in an activated virtual environment:

```bash
python -m pip install .
```

CPU is the default. Install your CUDA-compatible PyTorch build first and pass `--device cuda` for GPU execution.

## Data preparation

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

## Training

```bash
exposureflow train --data data/prepared --output outputs/full --epochs 30 --width 64 --updates 6 --quadrature 8
```

For dataset-specific fine-tuning, pass `--initialize outputs/pretraining/model.pt` with the same width/update architecture. A new optimizer is created while model parameters start from that checkpoint.

Timestamp Fourier codes enter local cell attention before bilinear spatial scattering. Each eight-pixel cell retains at most four events per temporal/polarity stratum, for up to 32 tokens per cell and 120,000 per window. The event observation loss separately samples adjacent trigger pairs uniformly, so its estimator does not inherit token-sampling bias.

The recurrent decoder predicts controls `c[4,2,H,W]`; queried flow is `sum_j (B_j(t)-B_j(reference))*c_j`. Inverse warping uses four fixed-point steps and rejects nonconverged or out-of-image correspondences. Gauss–Legendre quadrature approximates the exposure integral. A no-event term samples interior intervals between consecutive same-pixel events.

The loss includes heteroscedastic flow error when labels exist, event contrast, rendered RGB/gradient agreement, curvature/spatial regularization, an offset prior or support-gated offset supervision, and scale calibration. Validation observation/flow objective selects the checkpoint. The offset bound is ±6 ms. `--support-threshold 0.1` gates synthetic offset supervision using finite-difference event sensitivity; it is not a learned clock confidence.

## Baselines and ablations

```bash
exposureflow train --data data/prepared --output outputs/frame-only --variant frame_only
exposureflow train --data data/prepared --output outputs/midpoint --variant midpoint
exposureflow train --data data/prepared --output outputs/zero --variant fixed_zero
exposureflow train --data data/prepared --output outputs/binned --variant binned_events
exposureflow train --data data/prepared --output outputs/linear --variant linear
exposureflow train --data data/prepared --output outputs/two-fields --variant two_trajectories
exposureflow train --data data/prepared --output outputs/no-scale --variant no_scale
```

The frame-only control removes event inputs and event loss. Midpoint uses one exposure sample. Binned events replace times with ten bin centers. Linear motion keeps only first-order displacement. The two-field control assigns independent control fields to the event and RGB likelihoods; the full model averages the same two control heads into a shared field, keeping parameter count equal. No-scale fixes the Laplace scale at one pixel. `oracle_offset` requires `offset_gt` in every training/validation/evaluation window and supplies it only to the observation operator.

## Controlled timing and exposure

```bash
python stress.py --input data/raw-window.npz --output data/offset3.npz --offset-ms 3
python stress.py --input data/raw-window.npz --output data/exposure10-offset3.npz --exposure-ms 10 --offset-ms 3
```

Timing injection subtracts the requested shift from event-clock times, preserves count/polarity and records the required positive correction. Exposure synthesis additionally requires `sharp[H,W,3]` in `[0,1]`, a known cubic `control[4,2,H,W]`, absolute `t0,t1`, and `reference`. It integrates the known radiance/trajectory using 32 quadrature samples around each original exposure midpoint. These are generator inputs, not network inputs. Generate 1/5/10-ms exposure conditions and offsets −5/−3/−1/0/1/3/5 ms into separate manifests before preparation.

## Evaluation and outputs

```bash
exposureflow evaluate --data data/prepared --model outputs/full/model.pt --output outputs/evaluation --quadrature 16
```

Training writes `model.pt`, `history.json`, and `learning.png`. Evaluation writes per-window controls and reference-relative flows at four query times, endpoint-flow plots for the first three windows, plus `metrics.json` and `risk_coverage.png` when flow labels exist.

Metrics include endpoint EPE, percentage above `max(3 pixels,5% target magnitude)`, 1-pixel error, per-window trajectory EPE, Laplace NLL, ten equal-mass-bin scale calibration error in pixels, EPE AURC, offset MAE/bias in milliseconds, event residual, exposure loss, offset support, parameter count, elapsed window time and peak allocated GPU memory. NLL uses both displacement components. Scale calibration compares mean L1 error with the Laplace expectation `2*s`; AURC ranks pixels by increasing predicted scale. Missing timing or flow labels leave their metrics absent.

Elapsed time includes data loading, model execution, diagnostic rendering and metric computations for every exported window. It is an execution measurement rather than the paper's 50-warm-up/500-window timing protocol. Repeat paired seeds and scene splits before comparing methods. Group stress results by ground-truth motion tertiles and registered exposure/offset settings.

```bash
python compare.py --runs outputs/full-evaluation/metrics.json outputs/midpoint-evaluation/metrics.json --output outputs/comparison
```

The comparison script plots only metrics actually present in the selected evaluation outputs. Use identical sequences, query times and validity masks for paired method comparisons.

## Implementation choices

The backbone uses a compact shared RGB encoder, local RGB correlation, timestamp cell attention and convolutional recurrent refinement. It retains the paper's temporal observation model with a smaller feature hierarchy. Contrast thresholds are a smooth training-learned bounded spatial field shared across sequences; they are not optimized on test windows. Occlusion handling uses geometric validity and fixed-point consistency, plus supplied validity masks; there is no separate occlusion predictor. Flow queries reuse the encoded controls. These choices affect capacity, sensor-threshold adaptation and runtime while keeping the observation losses tied to the same geometric trajectory.
