#!/usr/bin/env python3
import argparse, os, sys, glob, math
from typing import Dict, List, Tuple
import yaml
import matplotlib.pyplot as plt
import csv

TARGET_STEPS = 1000  # pad to this many steps by repeating last value

def find_yaml_files(paths: List[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            # look for typical filenames, but accept any *.yaml
            files.extend(glob.glob(os.path.join(p, "**", "object_recon_metrics.yaml"), recursive=True))
            files.extend(glob.glob(os.path.join(p, "**", "*.yaml"), recursive=True))
        elif os.path.isfile(p) and p.lower().endswith((".yaml", ".yml")):
            files.append(p)
    # de-dup while preserving order
    seen = set(); out = []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out

def load_series_from_yaml(path: str) -> Tuple[str, List[int], List[float]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    policy = "unknown"
    try:
        policy = str(data.get("experiment", {}).get("policy_name", "unknown"))
        if policy == "unknown":
            # helpful fallback: parent folder name
            policy = os.path.basename(os.path.dirname(path)) or "unknown"
    except Exception:
        pass

    steps = []
    crs   = []
    for row in (data.get("steps") or []):
        try:
            s = int(row.get("step"))
            cr = float(row.get("completeness_ratio", 0.0))
            steps.append(s)
            crs.append(cr)
        except Exception:
            continue

    # sort by step and collapse duplicates (keep the latest)
    zipped = sorted(zip(steps, crs), key=lambda x: x[0])
    dedup: Dict[int, float] = {}
    for s, v in zipped:
        dedup[s] = v
    steps_sorted = sorted(dedup.keys())
    cr_sorted    = [dedup[s] for s in steps_sorted]

    return policy, steps_sorted, cr_sorted

def pad_to_target(steps: List[int], values: List[float], target_steps: int) -> Tuple[List[int], List[float]]:
    if not steps:
        return list(range(target_steps)), [0.0]*target_steps
    # build dense array up to max(existing, target)
    max_step = max(steps)
    last_value = values[-1]
    out_steps = list(range(min(target_steps, max(max_step+1, target_steps))))
    # sparse -> dense (carry forward the most recent value)
    dense = []
    cur_idx = 0
    for s in out_steps:
        while cur_idx+1 < len(steps) and steps[cur_idx+1] <= s:
            cur_idx += 1
        if steps[cur_idx] <= s:
            dense.append(values[cur_idx])
        else:
            dense.append(values[0])  # before first recorded step (rare) -> hold first
    # if we still need to extend to target length, repeat the last value
    if len(out_steps) < target_steps:
        pad_len = target_steps - len(out_steps)
        out_steps = list(range(target_steps))
        dense += [last_value] * pad_len
    elif len(out_steps) > target_steps:
        dense = dense[:target_steps]
        out_steps = out_steps[:target_steps]

    return out_steps, dense

def main():
    ap = argparse.ArgumentParser(description="Plot completeness_ratio over steps for multiple policies.")
    ap.add_argument("paths", nargs="+", help="YAML files or directories to scan")
    ap.add_argument("--out", default="completeness_over_steps.png", help="Output plot filename (PNG)")
    ap.add_argument("--csv", default="completeness_merged.csv", help="Output merged CSV")
    ap.add_argument("--title", default="Object Reconstruction: Completeness Ratio vs Steps", help="Plot title")
    ap.add_argument("--target_steps", type=int, default=TARGET_STEPS, help="Pad curves to this length")
    args = ap.parse_args()

    files = find_yaml_files(args.paths)
    print(f"Found {len(files)} YAML files to process.")
    if not files:
        print("No YAML files found.", file=sys.stderr)
        sys.exit(1)

    # collect per-policy series (aggregate multiple runs of the same policy if present)
    per_policy: Dict[str, List[Tuple[List[int], List[float], str]]] = {}

    for f in files:
        try:
            policy, steps, cr = load_series_from_yaml(f)
            if not steps:
                continue
            steps_p, cr_p = pad_to_target(steps, cr, args.target_steps)
            per_policy.setdefault(policy, []).append((steps_p, cr_p, f))
            print(f"[OK] {policy:20s}  {os.path.relpath(f)}  (points: {len(steps)} -> {len(steps_p)})")
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}", file=sys.stderr)

    if not per_policy:
        print("No valid series extracted.", file=sys.stderr)
        sys.exit(1)

    # prepare merged CSV
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as csvf:
        writer = None
        for policy, runs in per_policy.items():
            for run_idx, (steps_p, cr_p, src) in enumerate(runs):
                # write one row per step per policy/run
                for s, v in zip(steps_p, cr_p):
                    row = {
                        "policy": policy,
                        "run": run_idx,
                        "step": s,
                        "completeness_ratio": v,
                        "source": src,
                    }
                    if writer is None:
                        writer = csv.DictWriter(csvf, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)

    # plot
    plt.figure(figsize=(10, 5), dpi=140)
    for policy, runs in per_policy.items():
        # if multiple runs per policy, average them
        # stack per step
        import numpy as np
        stack = np.stack([r[1] for r in runs], axis=0)  # (R, T)
        mean_curve = stack.mean(axis=0)
        # optionally: std band
        std_curve = stack.std(axis=0)

        steps_axis = runs[0][0]
        # main line
        plt.plot(steps_axis, mean_curve, label=policy)
        # shaded std (comment out if you prefer just lines)
        plt.fill_between(steps_axis, mean_curve-std_curve, mean_curve+std_curve, alpha=0.15)

    plt.xlabel("Steps")
    plt.ylabel("Completeness ratio (%)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Policy", ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"\nSaved plot to: {args.out}")
    print(f"Saved merged CSV to: {args.csv}")

if __name__ == "__main__":
    main()