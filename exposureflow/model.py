import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def bernstein(t):
    return torch.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3], -1)


def sample(field, points):
    height, width = field.shape[-2:]
    grid = points / points.new_tensor([max(width - 1, 1), max(height - 1, 1)]) * 2 - 1
    return F.grid_sample(field[None], grid[None, None], align_corners=True, padding_mode="border")[0, :, 0].T


def displacement(control, points, times, reference, linear=False):
    coefficients = sample(control.flatten(0, 1), points).reshape(-1, 4, 2)
    if linear:
        return (times - reference)[..., None] * (coefficients[:, 3] - coefficients[:, 0])
    return ((bernstein(times) - bernstein(reference))[..., None] * coefficients).sum(1)


def inverse(control, points, times, reference, iterations, linear=False):
    coordinate = points
    for _ in range(iterations):
        coordinate = points - displacement(control, coordinate, times, reference, linear)
    height, width = control.shape[-2:]
    valid = (coordinate[:, 0] >= 0) & (coordinate[:, 0] <= width - 1) & (coordinate[:, 1] >= 0) & (coordinate[:, 1] <= height - 1)
    residual = coordinate + displacement(control, coordinate, times, reference, linear) - points
    return coordinate, valid & (residual.norm(dim=-1) < 1)


class CellEvents(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(15, width), nn.SiLU(), nn.Linear(width, width))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.time_projection = nn.Linear(12, width, bias=False)

    def forward(self, events, height, width, cell=8):
        output = events.new_zeros(height * width, self.embedding[-1].out_features)
        mass = events.new_zeros(height * width, 1)
        if not len(events):
            return output.T.reshape(-1, height, width)
        time_code = torch.cat([torch.sin(events[:, 2:3] * 2 ** j * math.pi) for j in range(6)] + [torch.cos(events[:, 2:3] * 2 ** j * math.pi) for j in range(6)], -1)
        token = self.embedding(torch.cat([events[:, :2] / events.new_tensor([width, height]), events[:, 3:4], time_code], -1))
        token = token + self.time_projection(time_code)
        cells = (events[:, 1].long() // cell) * math.ceil(width / cell) + events[:, 0].long() // cell
        refined = torch.zeros_like(token)
        groups = {int(identity): torch.where(cells == identity)[0] for identity in cells.unique()}
        columns = math.ceil(width / cell)
        for identity, ids in groups.items():
            query = token[ids][None]
            row, column = divmod(identity, columns)
            neighbors = [groups[y * columns + x] for y in range(max(0, row - 1), row + 2) for x in range(max(0, column - 1), min(columns, column + 2)) if y * columns + x in groups]
            context = token[torch.cat(neighbors)][None]
            mixed, _ = self.attention(query, context, context, need_weights=False)
            refined[ids] = token[ids] + mixed[0]
        x0, y0 = events[:, 0].floor().long(), events[:, 1].floor().long()
        fx, fy = events[:, 0] - x0, events[:, 1] - y0
        for dx, dy, weight in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
            xx, yy = x0 + dx, y0 + dy
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            position = yy[valid] * width + xx[valid]
            output.index_add_(0, position, refined[valid] * weight[valid, None]); mass.index_add_(0, position, weight[valid, None])
        return (output / mass.clamp_min(1e-6)).T.reshape(-1, height, width)


class ExposureFlow(nn.Module):
    def __init__(self, width=64, updates=6, variant="full", offset_bound=.006):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(3, width, 5, 2, 2), nn.SiLU(), nn.Conv2d(width, width, 3, 2, 1), nn.SiLU())
        self.events = CellEvents(width)
        self.context = nn.Sequential(nn.Conv2d(3 * width + 25, width, 3, padding=1), nn.SiLU())
        self.gru = nn.Conv2d(2 * width, 3 * width, 3, padding=1)
        self.control = nn.Conv2d(width, 16, 3, padding=1)
        self.radiance = nn.Conv2d(width, 3, 3, padding=1)
        self.scale = nn.Conv2d(width, 4, 3, padding=1)
        self.offset = nn.Linear(3 * width, 1)
        self.contrast = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.updates, self.variant, self.offset_bound = updates, variant, offset_bound

    def forward(self, frames, events, duration, oracle=None):
        height, width = frames.shape[-2:]
        encoded = self.rgb(frames)
        if self.variant == "frame_only":
            events = events[:0]
        if self.variant == "binned_events":
            events = events.clone(); events[:, 2] = (torch.floor(events[:, 2] * 10) + .5) / 10
        event_map = self.events(events, height, width)
        event_map = F.interpolate(event_map[None], encoded.shape[-2:], mode="bilinear", align_corners=True)
        first, second = encoded[:1], encoded[1:]
        other = F.unfold(second, 5, padding=2).reshape(1, first.shape[1], 25, *first.shape[-2:])
        correlation = (first[:, :, None] * other).mean(1)
        context = self.context(torch.cat([first, second, event_map, correlation], 1))
        hidden = torch.zeros_like(context)
        for _ in range(self.updates):
            reset, update, candidate = self.gru(torch.cat([hidden, context], 1)).chunk(3, 1)
            candidate = torch.tanh(candidate + reset.sigmoid() * hidden)
            hidden = (1 - update.sigmoid()) * hidden + update.sigmoid() * candidate
        controls = F.interpolate(self.control(hidden), (height, width), mode="bilinear", align_corners=True)[0].reshape(2, 4, 2, height, width)
        event_control, rgb_control = controls[0], controls[1]
        if self.variant != "two_trajectories":
            event_control = rgb_control = controls.mean(0)
        pooled = torch.cat([encoded.mean((2, 3)).flatten(), event_map.mean((2, 3)).flatten()])
        delta = self.offset_bound * torch.tanh(self.offset(pooled)[0])
        if self.variant == "fixed_zero":
            delta = delta * 0
        if self.variant == "oracle_offset":
            if oracle is None:
                raise ValueError("oracle_offset requires declared offset_gt")
            delta = delta * 0 + oracle
        radiance = F.interpolate(self.radiance(hidden), (height, width), mode="bilinear", align_corners=True)[0].sigmoid().clamp_min(1e-4)
        scale = F.interpolate(self.scale(hidden), (height, width), mode="bilinear", align_corners=True)[0]
        contrast = .05 + .45 * F.interpolate(self.contrast, (height, width), mode="bilinear", align_corners=True)[0, 0].sigmoid()
        return {"event_control": event_control, "control": rgb_control, "radiance": radiance, "scale": scale, "contrast": contrast, "delta": delta, "normalized_delta": delta / duration}


def pixel_grid(height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2).float()


def query(state, time, reference, variant):
    height, width = state["control"].shape[-2:]; points = pixel_grid(height, width, state["control"].device)
    times = time.expand(len(points))
    flow = displacement(state["control"], points, times, reference, variant == "linear").reshape(height, width, 2)
    raw = (bernstein(time)[:, None, None] * state["scale"]).sum(0)
    scale = F.softplus(raw) + .01
    if variant == "no_scale":
        scale = scale * 0 + 1
    return flow, scale


def render(state, exposure, reference, variant, quadrature=8, iterations=4):
    nodes, weights = np.polynomial.legendre.leggauss(1 if variant == "midpoint" else quadrature)
    height, width = state["radiance"].shape[-2:]; points = pixel_grid(height, width, state["radiance"].device)
    color = state["radiance"].new_zeros(height * width, 3); validity = torch.ones(len(points), device=points.device, dtype=torch.bool)
    for node, weight in zip(nodes, weights):
        t = exposure[0] + (float(node) + 1) * .5 * (exposure[1] - exposure[0])
        coordinate, valid = inverse(state["control"], points, t.expand(len(points)), reference, iterations, variant == "linear")
        color = color + float(weight) * .5 * sample(state["radiance"], coordinate); validity &= valid
    return color.T.reshape(3, height, width), validity.reshape(height, width)


def event_residual(state, pairs, reference, variant, shift=None):
    if not len(pairs):
        return pairs.new_empty(0), pairs.new_empty(0, dtype=torch.bool)
    shift = state["normalized_delta"] if shift is None else shift
    first, valid1 = inverse(state["event_control"], pairs[:, :2], pairs[:, 2] + shift, reference, 4, variant == "linear")
    previous, valid0 = inverse(state["event_control"], pairs[:, :2], pairs[:, 4] + shift, reference, 4, variant == "linear")
    luminance = (state["radiance"] * state["radiance"].new_tensor([.299, .587, .114])[:, None, None]).sum(0, keepdim=True).log()
    contrast = sample(state["contrast"][None], pairs[:, :2])[:, 0]
    residual = sample(luminance, first)[:, 0] - sample(luminance, previous)[:, 0] - pairs[:, 3] * contrast
    return residual, valid0 & valid1
