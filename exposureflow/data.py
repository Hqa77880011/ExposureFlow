import csv
import json
from pathlib import Path

import numpy as np
import torch


def prepare(args):
    manifest = Path(args.manifest); rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    groups, records = {}, []
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows):
        if row["split"] not in {"train", "val", "test"} or groups.setdefault(row["sequence"], row["split"]) != row["split"]:
            raise ValueError("Sequence-disjoint train/val/test splits required")
        data = dict(np.load(manifest.parent / row["path"], allow_pickle=False))
        frames, events, exposure = data["frames"], data["events"].astype(np.float64), data["exposure"]
        if frames.ndim != 4 or frames.shape[0] != 2 or frames.shape[-1] != 3 or min(frames.shape[1:3]) < 2 or exposure.shape != (2, 2) or not np.isfinite(exposure).all() or np.any(exposure[:, 1] <= exposure[:, 0]):
            raise ValueError("Two RGB frames and two positive exposure intervals required")
        if events.ndim != 2 or events.shape[1] != 4 or not np.isfinite(events).all() or not np.isfinite(frames).all():
            raise ValueError("Finite events[x,y,time,polarity] and RGB frames required")
        if not np.isin(events[:, 3], [-1, 1]).all():
            raise ValueError("Event polarity uses -1/+1")
        if len(events) and (np.any(events[:, :2] < 0) or np.any(events[:, 0] >= frames.shape[2]) or np.any(events[:, 1] >= frames.shape[1])):
            raise ValueError("Rectify event coordinates to the RGB image domain before preparation")
        ordering = np.argsort(events[:, 2], kind="stable")
        if "event_pixel_id" in data:
            pixel_ids = data["event_pixel_id"][ordering]
        else:
            if not np.allclose(events[:, :2], np.rint(events[:, :2])):
                raise ValueError("Subpixel rectified events require original event_pixel_id for correct trigger predecessors")
            pixel_ids = (events[:, 1].astype(np.int64) * frames.shape[2] + events[:, 0].astype(np.int64))[ordering]
        events = events[ordering]
        start, end = float(exposure.min()), float(exposure.max()); duration = end - start
        data["reference"] = np.array((float(data.get("reference", exposure[0].mean())) - start) / duration, dtype=np.float32)
        data["duration"] = np.array(duration, dtype=np.float32)
        data["exposure"] = ((exposure - start) / duration).astype(np.float32)
        events[:, 2] = (events[:, 2] - start) / duration
        previous, pairs = {}, []
        for event, pixel_id in zip(events, pixel_ids):
            key = int(pixel_id)
            if key in previous:
                pairs.append([*event, previous[key]])
            previous[key] = float(event[2])
        data["events"] = events.astype(np.float32); data["pairs"] = np.array(pairs, dtype=np.float32).reshape(-1, 5)
        data["frames"] = (frames.astype(np.float32) / (255 if frames.max() > 1 else 1)).transpose(0, 3, 1, 2)
        if "flow_time" in data:
            data["flow_time"] = ((data["flow_time"] - start) / duration).astype(np.float32)
            ordering = np.argsort(data["flow_time"]); data["flow_time"] = data["flow_time"][ordering]; data["flow"] = data["flow"][ordering]
            if "valid" in data:
                data["valid"] = data["valid"][ordering]
            expected = (len(ordering), *frames.shape[1:3], 2)
            if data["flow"].shape != expected:
                raise ValueError("Flow targets must align with query times and rectified image dimensions")
            data["valid"] = data.get("valid", np.ones(expected[:-1], dtype=bool)).astype(bool) & np.isfinite(data["flow"]).all(-1)
            data["flow"] = np.where(data["valid"][..., None], data["flow"], 0).astype(np.float32)
        name = f"window_{number:06d}.npz"; np.savez_compressed(output / name, **data)
        records.append(dict(file=name, sequence=row["sequence"], split=row["split"]))
    (output / "manifest.json").write_text(json.dumps(records, indent=2))


def read(root, row, seed, device, maximum=120000):
    data = dict(np.load(Path(root) / row["file"], allow_pickle=False))
    events = data["events"]; rng = np.random.default_rng(seed)
    selected, weights = [], []
    if len(events):
        cell = events[:, :2].astype(int) // 8
        time_bin = np.minimum(3, np.maximum(0, (events[:, 2] * 4).astype(int)))
        keys = np.column_stack([cell, time_bin, events[:, 3] > 0])
        _, groups = np.unique(keys, axis=0, return_inverse=True)
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group); count = min(4, len(ids))
            chosen = rng.choice(ids, count, replace=False); selected.extend(chosen); weights.extend([len(ids) / count] * count)
        if len(selected) > maximum:
            positions = rng.choice(len(selected), maximum, replace=False)
            probability = maximum / len(selected); selected = np.array(selected)[positions]; weights = np.array(weights)[positions] / probability
    data["events"] = events[np.asarray(selected, dtype=int)]
    pairs = data["pairs"]
    if len(pairs) > maximum:
        pairs = pairs[rng.choice(len(pairs), maximum, replace=False)]
    data["pairs"] = pairs
    keys = {"frames", "events", "pairs", "duration", "reference", "exposure", "flow", "flow_time", "valid", "offset_gt", "exposure_valid"}
    return {key: torch.as_tensor(value, device=device) for key, value in data.items() if key in keys}
