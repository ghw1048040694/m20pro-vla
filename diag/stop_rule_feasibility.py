#!/usr/bin/env python3
"""Offline feasibility study: can the teacher's stop decision be made observable?

Motivation
----------
`collect_m20_mujoco_vla.py` sets the expert stop label from a *privileged* quantity:

    if distance <= curriculum_success_radius:   # 0.70 m, distance uses target_xy
        expert = (0.0, 0.0, 0.0, 1.0)

`target_xy` is exactly what the policy is forbidden to use. M20-STOP-DIAG-02 found
that on the teacher's own stop frame the policy camera shows **zero** target pixels
in 19/29 episodes, i.e. the label is not decodable from the observation. Before
spending any simulator time on a fix, this script asks the offline question:

    Under a *candidate* stop rule driven only by observable evidence
    (front-camera pixels / lidar fan), how much evidence is actually present
    at the trigger, and where in the trajectory does the trigger land?

It answers that from the stored `front_rgb` / `lidar` / `proprio` streams, which
is free -- no MuJoCo, no policy, no GPU.

Prerequisite: the per-frame base pose. `proprio` is `concat(qpos, qvel)` with
`qpos[:3]` zeroed, but `qvel[:2]` is intact, so the trajectory is reconstructed by
backward integration anchored at the episode's `final_xy`:

    xy[t] = final_xy - dt * sum_{k>=t} v[k],   dt = 0.02 s

`dt` was calibrated against `initial_xy` and reproduces it to <=3 mm.

Usage:
    python diag/stop_rule_feasibility.py [--dataset m20_hidden_search_v2]
                                         [--out feasibility.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import statistics as st

import numpy as np

from m20pro_vla.data.visibility import target_color_mask

FRAME_DT_S = 0.02          # PHYSICS_STEPS(8) * model timestep (0.0025), calibrated vs initial_xy
FRAME_H, FRAME_W = 96, 160
FWD_RAYS = slice(32, 41)   # +-20 deg about the heading
DEFAULT_DATASET = "m20_hidden_search_v2"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / ".runtime" / "datasets"


def reconstruct_xy(proprio: np.ndarray, final_xy: np.ndarray) -> np.ndarray:
    """Backward-integrate base velocity into world xy, anchored at `final_xy`."""
    vel = proprio[:, 23:25].astype(float)
    tail = np.cumsum(vel[::-1], axis=0)[::-1]      # tail[t] = sum_{k>=t} v[k]
    return final_xy.astype(float) - tail * FRAME_DT_S


def yaw_from_quat(proprio: np.ndarray) -> np.ndarray:
    w, x, y, z = (proprio[:, i].astype(float) for i in range(3, 7))
    return np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def mask_centroid_row(rgb: np.ndarray, label: str) -> float:
    mask = target_color_mask(rgb, label)
    rows = np.flatnonzero(mask.any(axis=1))
    return float(rows.mean()) if rows.size else float("nan")


def pixel_trace(front: np.ndarray, label: str) -> np.ndarray:
    return np.array([int(target_color_mask(f, label).sum()) for f in front], dtype=np.int64)


def centroid_trace(front: np.ndarray, label: str) -> np.ndarray:
    return np.array([mask_centroid_row(f, label) for f in front], dtype=float)


def pct(values, q):
    values = [v for v in values if v == v]
    return float(np.percentile(values, q)) if values else float("nan")


def med(values):
    values = [v for v in values if v == v]
    return float(st.median(values)) if values else float("nan")


def load_episode(path: str):
    meta = json.load(open(path.replace(".npz", ".json")))
    z = np.load(path)
    proprio = z["proprio"].astype(float)
    action = z["action"].astype(float)
    front = z["front_rgb"]
    lidar = z["lidar"].astype(float)
    xy = reconstruct_xy(proprio, np.asarray(meta["final_xy"], float))
    target = np.asarray(meta["target_xy_privileged_label_only"], float)
    return dict(
        meta=meta, name=os.path.basename(path).replace(".npz", ""),
        xy=xy, dist=np.linalg.norm(xy - target, axis=1), yaw=yaw_from_quat(proprio),
        px=pixel_trace(front, meta["target_label"]),
        centroid=centroid_trace(front, meta["target_label"]),
        fan=lidar[:, FWD_RAYS].min(axis=1), action=action,
        stop=np.flatnonzero(action[:, 3] >= 0.5),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--stop-radius", type=float, default=0.70,
                        help="the radius the *dataset* was collected with; 0.70 for "
                             "m20_hidden_search_v2, 0.95 for the r095 re-collection")
    parser.add_argument("--out", default="/tmp/stop_rule_feasibility.json")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    stop_radius = float(args.stop_radius)

    root = pathlib.Path(os.path.expanduser(args.root)) / args.dataset
    paths = sorted(glob.glob(str(root / "episode_*.npz")))
    if not paths:
        print(f"ERROR: no episodes under {root}")
        return 2

    episodes = [load_episode(p) for p in paths]
    n = len(episodes)
    print(f"dataset {args.dataset}: {n} episodes, dt={FRAME_DT_S}s\n")

    # ---- 0. sanity: reconstruction vs stored privileged distance -----------
    errs = [abs(float(ep["dist"].min()) - float(ep["meta"]["min_target_distance"]))
            for ep in episodes if ep["meta"].get("min_target_distance") is not None]
    print("[0] reconstruction check: max |recon_minD - meta_minD| = %.4f m over %d eps"
          % (max(errs), len(errs)))

    # ---- 1. does the target ever become large in the camera? ---------------
    ever80 = sum(1 for ep in episodes if (ep["px"] >= 80).any())
    ever400 = sum(1 for ep in episodes if (ep["px"] >= 400).any())
    print("\n[1] visual availability (front camera)")
    print("    episodes with any frame >=  80 px: %d/%d" % (ever80, n))
    print("    episodes with any frame >= 400 px: %d/%d" % (ever400, n))
    peak = [int(ep["px"].max()) for ep in episodes]
    print("    per-episode peak px: min %d  median %d  max %d"
          % (min(peak), int(med(peak)), max(peak)))

    # ---- 2. where is the peak relative to arrival? -------------------------
    print("\n[2] peak-of-sight vs arrival (privileged d <= %.2f m)" % stop_radius)
    rows = []
    for ep in episodes:
        arr = np.flatnonzero(ep["dist"] <= stop_radius)
        if arr.size == 0:
            continue
        t_arr = int(arr[0])
        t_pk = int(np.argmax(ep["px"][:t_arr + 1])) if t_arr >= 0 else -1
        rows.append((ep["name"][8:], t_arr, t_pk, t_arr - t_pk,
                     float(ep["dist"][t_pk]), int(ep["px"][t_arr]),
                     float(ep["centroid"][t_arr])))
    print("    %-6s %6s %6s %7s %9s %8s %9s" %
          ("ep", "arriv", "px_pk", "lag", "d@px_pk", "px@arriv", "crow@arr"))
    for r in rows[:12]:
        print("    %-6s %6d %6d %7d %9.3f %8d %9.1f" % r)
    print("    ... %d episodes reached %.2f m" % (len(rows), stop_radius))
    print("    lag (arrival - peak)  : median %.0f frames  p90 %.0f"
          % (med([r[3] for r in rows]), pct([r[3] for r in rows], 90)))
    print("    distance at px peak   : median %.3f m  p10 %.3f  p90 %.3f"
          % (med([r[4] for r in rows]), pct([r[4] for r in rows], 10),
             pct([r[4] for r in rows], 90)))
    print("    px at arrival         : median %d  ==0 in %d/%d"
          % (med([r[5] for r in rows]), sum(1 for r in rows if r[5] == 0), len(rows)))

    # ---- 3. counterfactual: stop at a larger privileged distance -----------
    print("\n[3] counterfactual 'stop when d <= D' -- evidence at the trigger")
    print("    %6s %8s %8s %9s %9s" % ("D(m)", "px_med", "px_p10", "fan_med", "ep_hit"))
    dist_curve = []
    for d_stop in (0.70, 0.80, 0.90, 1.00, 1.20, 1.50):
        px_at, fan_at, hit = [], [], 0
        for ep in episodes:
            idx = np.flatnonzero(ep["dist"] <= d_stop)
            if idx.size == 0:
                px_at.append(np.nan); fan_at.append(np.nan); continue
            i = int(idx[0]); hit += 1
            px_at.append(float(ep["px"][i])); fan_at.append(float(ep["fan"][i]))
        dist_curve.append(dict(d_stop=d_stop, px_med=med(px_at), px_p10=pct(px_at, 10),
                               fan_med=med(fan_at), episodes_hit=hit))
        print("    %6.2f %8.0f %8.0f %9.3f %6d/%d"
              % (d_stop, med(px_at), pct(px_at, 10), med(fan_at), hit, n))

    # ---- 4. counterfactual: stop on an observable pixel event --------------
    print("\n[4] counterfactual observable rules -- px at the trigger")
    rules = {}
    for thresh in (80, 200, 400, 700):
        px_at, d_at, hit = [], [], 0
        for ep in episodes:
            idx = np.flatnonzero(ep["px"] >= thresh)
            if idx.size == 0:
                continue
            i = int(idx[0]); hit += 1
            px_at.append(float(ep["px"][i])); d_at.append(float(ep["dist"][i]))
        rules[f"first_px_ge_{thresh}"] = dict(hit=hit, px_med=med(px_at), d_med=med(d_at))
        print("    first frame px >= %-4d : fires %2d/%d  d_med %.3f m  px_med %.0f"
              % (thresh, hit, n, med(d_at), med(px_at)))

    # peak-then-retreat detector
    px_at, d_at, hit = [], [], 0
    for ep in episodes:
        px = ep["px"]
        if px.max() < 200:
            continue
        peak_i = int(np.argmax(px))
        after = np.flatnonzero(px[peak_i:] < 0.6 * px.max())
        if after.size == 0:
            continue
        i = peak_i + int(after[0]); hit += 1
        px_at.append(float(px[i])); d_at.append(float(ep["dist"][i]))
    rules["peak_then_retreat_0.6"] = dict(hit=hit, px_med=med(px_at), d_med=med(d_at))
    print("    px peak then drop <0.6x : fires %2d/%d  d_med %.3f m  px_med %.0f"
          % (hit, n, med(d_at), med(px_at)))

    # ---- 5. why does the target leave the frame? --------------------------
    print("\n[5] vertical image position vs distance (bottom row = %d)" % (FRAME_H - 1))
    buckets = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 2.5)]
    for lo, hi in buckets:
        rows_here = []
        for ep in episodes:
            sel = (ep["dist"] >= lo) & (ep["dist"] < hi) & (ep["px"] >= 20)
            rows_here.extend(ep["centroid"][sel].tolist())
        if rows_here:
            print("    d %.1f-%.1f m : n=%5d  centroid row median %5.1f  p90 %5.1f"
                  % (lo, hi, len(rows_here), med(rows_here), pct(rows_here, 90)))

    payload = dict(dataset=args.dataset, episodes=n, frame_dt_s=FRAME_DT_S,
                   stop_radius_m=stop_radius,
                   recon_max_err_m=max(errs), ever_80px=ever80, ever_400px=ever400,
                   distance_curve=dist_curve, observable_rules=rules)
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
