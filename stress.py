import argparse
from pathlib import Path

import numpy as np
import torch

from exposureflow.model import render


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output", required=True); parser.add_argument("--offset-ms", type=float, default=0); parser.add_argument("--exposure-ms", type=float); args = parser.parse_args()
    data = dict(np.load(args.input, allow_pickle=False)); data["events"] = data["events"].astype(float)
    shift = args.offset_ms / 1000; data["events"][:, 2] -= shift; data["offset_gt"] = np.array(float(data.get("offset_gt", 0)) + shift, dtype=np.float32)
    if args.exposure_ms is not None:
        if not all(key in data for key in ["sharp", "control", "t0", "t1", "reference"]):
            raise ValueError("Exposure synthesis requires sharp radiance, cubic control, t0/t1 and reference time")
        duration = float(data["t1"] - data["t0"]); reference = torch.tensor((float(data["reference"]) - float(data["t0"])) / duration)
        state = {"control": torch.tensor(data["control"], dtype=torch.float32), "radiance": torch.tensor(data["sharp"], dtype=torch.float32).permute(2, 0, 1)}
        centers = data["exposure"].mean(1); exposure = np.column_stack([centers - args.exposure_ms / 2000, centers + args.exposure_ms / 2000]); images = []; masks = []
        with torch.no_grad():
            for interval in exposure:
                image, valid = render(state, torch.tensor((interval - float(data["t0"])) / duration), reference, "full", 32)
                images.append(image.permute(1, 2, 0).numpy()); masks.append(valid.numpy())
        data["frames"] = np.array(images); data["exposure"] = exposure; data["exposure_valid"] = np.array(masks)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output, **data)


if __name__ == "__main__":
    main()
