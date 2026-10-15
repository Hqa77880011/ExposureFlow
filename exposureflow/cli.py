import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .data import prepare, read
from .model import ExposureFlow, event_residual, query, render, sample


def losses(model, data, args):
    oracle = data.get("offset_gt")
    state = model(data["frames"], data["events"], data["duration"], oracle)
    reference = data["reference"]; total = state["radiance"].sum() * 0; diagnostics = {}
    exposure_loss = total
    for k in range(2):
        prediction, valid = render(state, data["exposure"][k], reference, args.variant, args.quadrature)
        if "exposure_valid" in data:
            valid = valid & data["exposure_valid"][k].bool()
        difference = ((prediction - data["frames"][k]) ** 2 + 1e-6).sqrt().mean(0)
        exposure_loss = exposure_loss + (difference * valid).sum() / valid.sum().clamp_min(1) + .1 * (1 - valid.float().mean())
        for dimension in [-1, -2]:
            difference = (torch.diff(prediction, dim=dimension) - torch.diff(data["frames"][k], dim=dimension)).abs().mean(0)
            gradient_valid = valid[:, 1:] & valid[:, :-1] if dimension == -1 else valid[1:] & valid[:-1]
            gradient_error = (difference * gradient_valid).sum() / gradient_valid.sum().clamp_min(1)
            exposure_loss = exposure_loss + .2 * gradient_error
    pairs = data["pairs"]
    if args.variant == "binned_events":
        pairs = pairs.clone(); pairs[:, [2, 4]] = (torch.floor(pairs[:, [2, 4]] * 10) + .5) / 10
    residual, valid = event_residual(state, pairs, reference, args.variant)
    event_loss = ((residual[valid] ** 2 + 1e-6).sqrt().mean() if valid.any() else total)
    if len(pairs):
        no_event = pairs.clone(); gap = pairs[:, 2] - pairs[:, 4]
        no_event[:, 2] = pairs[:, 4] + .75 * gap; no_event[:, 4] = pairs[:, 4] + .25 * gap; no_event[:, 3] = 0
        contrast, valid0 = event_residual(state, no_event, reference, args.variant)
        threshold = sample(state["contrast"][None], pairs[:, :2])[:, 0]
        if valid0.any():
            event_loss = event_loss + .1 * F.relu(contrast.abs() - threshold)[valid0].mean()
        with torch.no_grad():
            plus, _ = event_residual(state, pairs, reference, args.variant, state["normalized_delta"] + .001)
            minus, _ = event_residual(state, pairs, reference, args.variant, state["normalized_delta"] - .001)
            sensitivity = ((plus - minus) / (.002 * data["duration"])) ** 2
            support = (sensitivity / (sensitivity.median() + 1e-8)).clamp(0, 10).mean()
    else:
        support = total.detach()
    flow_loss, calibration = total, total
    if "flow" in data:
        for index, t in enumerate(data["flow_time"]):
            predicted, scale = query(state, t, reference, args.variant)
            valid_flow = data.get("valid", torch.ones_like(data["flow"][..., 0], dtype=torch.bool))[index].bool()
            error = (predicted - data["flow"][index]).abs().sum(-1)
            if valid_flow.any():
                flow_loss = flow_loss + (error / scale + 2 * scale.log())[valid_flow].mean()
                empirical = (predicted - data["flow"][index]).abs()[valid_flow].detach().median() / math.log(2)
                calibration = calibration + (scale[valid_flow].mean() - empirical).abs()
    control = state["control"]
    curvature = (control[2:] - 2 * control[1:-1] + control[:-2]).abs().mean()
    spatial = total
    for dim in [-1, -2]:
        edge = torch.diff(state["radiance"], dim=dim).abs().mean(0)
        spatial = spatial + (torch.diff(control, dim=dim).abs().mean((0, 1)) * torch.exp(-10 * edge)).mean()
    contrast_tv = torch.diff(state["contrast"], dim=-1).abs().mean() + torch.diff(state["contrast"], dim=-2).abs().mean()
    offset_loss = state["delta"].abs() / .001
    if oracle is not None:
        offset_loss = (state["delta"] - oracle).abs() / .001 * (support.detach() >= args.support_threshold)
    if args.variant == "frame_only":
        event_loss = event_loss * 0
    total = flow_loss + .15 * event_loss + .2 * exposure_loss + .01 * (curvature + .1 * spatial) + .005 * offset_loss + .02 * calibration + .001 * contrast_tv
    diagnostics.update(exposure_error=float(exposure_loss.detach()), event_residual=float(event_loss.detach()), offset_support=float(support), offset_seconds=float(state["delta"].detach()))
    return total, state, diagnostics


def train(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    rows = json.loads((Path(args.data) / "manifest.json").read_text()); training = [r for r in rows if r["split"] == "train"]; validation = [r for r in rows if r["split"] == "val"]
    if not training or not validation:
        raise ValueError("Separate training and validation sequences required")
    model = ExposureFlow(args.width, args.updates, args.variant).to(args.device)
    if args.initialize:
        source = torch.load(args.initialize, map_location=args.device, weights_only=True)
        model.load_state_dict(source["state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); best, history = float("inf"), []
    for epoch in range(args.epochs):
        model.train(); values = []
        for i in rng.permutation(len(training)):
            data = read(args.data, training[i], args.seed + epoch + int(i), args.device)
            loss, _, _ = losses(model, data, args)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step(); values.append(float(loss.detach()))
        model.eval(); val = []
        with torch.no_grad():
            for i, row in enumerate(validation):
                loss, _, _ = losses(model, read(args.data, row, args.seed + i, args.device), args); val.append(float(loss))
        score = float(np.mean(val)); history.append(dict(epoch=epoch, train=float(np.mean(values)), validation=score))
        if score < best:
            best = score; torch.save({"state": model.state_dict(), "config": vars(args)}, output / "model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2)); plt.plot([r["train"] for r in history], label="Train"); plt.plot([r["validation"] for r in history], label="Validation"); plt.legend(); plt.savefig(output / "learning.png", dpi=180)


def evaluate(args):
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True); config = argparse.Namespace(**checkpoint["config"]); config.quadrature = args.quadrature
    model = ExposureFlow(config.width, config.updates, config.variant).to(args.device); model.load_state_dict(checkpoint["state"]); model.eval()
    rows = [r for r in json.loads((Path(args.data) / "manifest.json").read_text()) if r["split"] == args.split]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); records, errors, scales, endpoint_error, magnitudes, signed_offsets, euclidean_errors = [], [], [], [], [], [], []
    timing = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter(); data = read(args.data, row, config.seed + i, args.device)
            _, state, diagnostic = losses(model, data, config)
            predictions = []
            for t in [.25, .5, .75, 1.]:
                flow, _ = query(state, state["delta"].new_tensor(t), data["reference"], config.variant); predictions.append(flow.cpu().numpy())
            if "flow" in data:
                window_errors = []
                for j, t in enumerate(data["flow_time"]):
                    flow, scale = query(state, t, data["reference"], config.variant)
                    valid = data.get("valid", torch.ones_like(data["flow"][..., 0], dtype=torch.bool))[j].bool()
                    difference = flow - data["flow"][j]; epe = difference.norm(dim=-1)[valid]
                    errors.extend(difference.abs().sum(-1)[valid].cpu().numpy()); scales.extend(scale[valid].cpu().numpy()); window_errors.extend(epe.cpu().numpy())
                    euclidean_errors.extend(epe.cpu().numpy())
                    if j == len(data["flow_time"]) - 1:
                        endpoint_error.extend(epe.cpu().numpy()); magnitudes.extend(data["flow"][j].norm(dim=-1)[valid].cpu().numpy())
                diagnostic["trajectory_epe"] = float(np.mean(window_errors)) if window_errors else None
            if "offset_gt" in data:
                signed_offsets.append(float(state["delta"] - data["offset_gt"]))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            timing.append(time.perf_counter() - started); records.append(dict(window=row["file"], **diagnostic))
            np.savez_compressed(output / row["file"], control=state["control"].cpu().numpy(), flow=np.array(predictions), query_time=np.array([.25, .5, .75, 1.]), offset=float(state["delta"]), reference=float(data["reference"]))
            if i < 3:
                endpoint = predictions[-1]; figure, axis = plt.subplots(); axis.imshow(np.linalg.norm(endpoint, axis=-1), cmap="viridis")
                step = max(1, min(endpoint.shape[:2]) // 20); yy, xx = np.mgrid[0:endpoint.shape[0]:step, 0:endpoint.shape[1]:step]
                axis.quiver(xx, yy, endpoint[::step, ::step, 0], endpoint[::step, ::step, 1], color="white"); figure.savefig(output / f"flow_{i:06d}.png", dpi=180); plt.close(figure)
    metrics = {"windows": records, "variant": config.variant, "mean_window_seconds": float(np.mean(timing)) if timing else None, "parameters": sum(p.numel() for p in model.parameters())}
    if errors:
        error, scale = np.array(errors), np.array(scales); endpoint, magnitude = np.array(endpoint_error), np.array(magnitudes)
        metrics.update(endpoint_epe=float(endpoint.mean()), outlier_percent=float(100 * np.mean(endpoint > np.maximum(3, .05 * magnitude))), one_pixel_error_percent=float(100 * np.mean(endpoint > 1)), laplace_nll=float(np.mean(error / scale + 2 * np.log(2 * scale))))
        bins = [ids for ids in np.array_split(np.argsort(scale), min(10, len(scale))) if len(ids)]
        metrics["scale_ece_pixels"] = float(sum(len(ids) / len(scale) * abs(error[ids].mean() - 2 * scale[ids].mean()) for ids in bins))
        order = np.argsort(scale); risk = np.cumsum(np.array(euclidean_errors)[order]) / np.arange(1, len(error) + 1); metrics["aurc_epe_pixels"] = float(risk.mean())
        plt.plot(np.arange(1, len(error) + 1) / len(error), risk); plt.xlabel("Coverage"); plt.ylabel("Flow endpoint error"); plt.savefig(output / "risk_coverage.png", dpi=180)
    if signed_offsets:
        metrics.update(offset_mae_ms=float(np.mean(np.abs(signed_offsets)) * 1000), offset_bias_ms=float(np.mean(signed_offsets) * 1000))
    metrics["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--manifest", required=True); p.add_argument("--output", required=True); p.set_defaults(func=prepare)
    p = sub.add_parser("train"); p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["full", "frame_only", "midpoint", "fixed_zero", "oracle_offset", "binned_events", "linear", "two_trajectories", "no_scale"], default="full")
    for key, value in [("width", 64), ("updates", 6), ("epochs", 30), ("quadrature", 8), ("seed", 7)]:
        p.add_argument("--" + key, type=int, default=value)
    p.add_argument("--lr", type=float, default=.0002); p.add_argument("--support-threshold", type=float, default=.1); p.add_argument("--device", default="cpu"); p.add_argument("--initialize"); p.set_defaults(func=train)
    p = sub.add_parser("evaluate"); p.add_argument("--data", required=True); p.add_argument("--model", required=True); p.add_argument("--output", required=True); p.add_argument("--split", default="test", choices=["val", "test"]); p.add_argument("--quadrature", type=int, default=16); p.add_argument("--device", default="cpu"); p.set_defaults(func=evaluate)
    args = parser.parse_args()
    if hasattr(args, "width") and (args.width < 4 or args.width % 4 or min(args.updates, args.epochs, args.quadrature, args.lr) <= 0):
        parser.error("Width must be a positive multiple of four and optimization sizes positive")
    args.func(args)
