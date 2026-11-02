"""Build log-uniform (in centroid distance) no_intersection train/eval datasets from several raw CSVs.

Pass 1 computes the centroid distance of every valid row. Rows are then drawn without replacement so
that the 100 log-spaced centroid-distance bins are as equally populated as the pool allows (bins that
run short give up their deficit to the others). Pass 2 extracts the chosen rows; the result is fully
shuffled, because downstream truncation keeps the first N rows of a file.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.file_management import EXPECTED_COLUMNS  # noqa: E402

CHUNK = 500_000


def centroid_distance(coords: np.ndarray) -> np.ndarray:
    t1 = coords[:, :12].reshape(-1, 4, 3).mean(axis=1)
    t2 = coords[:, 12:24].reshape(-1, 4, 3).mean(axis=1)
    return np.linalg.norm(t1 - t2, axis=1)


def read_chunks(path):
    return pd.read_csv(path, chunksize=CHUNK, dtype=np.float64)


def valid_mask(chunk: pd.DataFrame) -> np.ndarray:
    values = chunk.values
    ok = ~np.isnan(values).any(axis=1)
    ok &= (chunk["HasIntersection"].values == 0) & (chunk["IntersectionVolume"].values == 0)
    return ok


def water_fill(counts: np.ndarray, total: int) -> np.ndarray:
    """Per-bin quotas as equal as possible, capped by availability, summing to `total`."""
    quota = np.zeros_like(counts)
    remaining = total
    open_bins = counts > 0
    while remaining > 0 and open_bins.any():
        share = remaining // open_bins.sum()
        if share == 0:
            idx = np.flatnonzero(open_bins)[:remaining]
            quota[idx] += 1
            break
        take = np.minimum(counts - quota, share) * open_bins
        quota += take
        remaining -= take.sum()
        open_bins = (counts - quota) > 0
    return quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--train-size", type=int, default=5_000_000)
    ap.add_argument("--eval-size", type=int, default=100_000)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    total = args.train_size + args.eval_size

    # Pass 1: centroid distance per row (NaN = invalid row).
    cds = []
    for path in args.inputs:
        parts, dropped = [], 0
        for chunk in read_chunks(path):
            assert list(chunk.columns) == EXPECTED_COLUMNS, f"unexpected columns in {path}"
            ok = valid_mask(chunk)
            dropped += int((~ok).sum())
            cd = np.full(len(chunk), np.nan, dtype=np.float64)
            cd[ok] = centroid_distance(chunk.values[ok, :24])
            parts.append(cd)
        cd = np.concatenate(parts)
        cds.append(cd)
        print(f"{os.path.basename(path)}: {len(cd):,} rows, {dropped:,} invalid dropped, "
              f"cd range [{np.nanmin(cd):.4f}, {np.nanmax(cd):.4f}]", flush=True)

    pool = np.concatenate(cds)
    valid = ~np.isnan(pool)
    if valid.sum() < total:
        raise SystemExit(f"Only {valid.sum():,} valid rows, need {total:,}")

    log_cd = np.log10(np.clip(pool, 1e-9, None))
    edges = np.linspace(np.nanmin(log_cd), np.nanmax(log_cd), args.bins + 1)
    bin_id = np.clip(np.digitize(log_cd, edges) - 1, 0, args.bins - 1)
    bin_id[~valid] = -1

    counts = np.bincount(bin_id[valid], minlength=args.bins)
    quota = water_fill(counts, total)
    print(f"bin pool sizes min/median/max: {counts.min():,}/{int(np.median(counts)):,}/{counts.max():,}")
    print(f"bin quotas     min/median/max: {quota.min():,}/{int(np.median(quota)):,}/{quota.max():,}")

    chosen = np.zeros(len(pool), dtype=bool)
    for b in range(args.bins):
        if quota[b] == 0:
            continue
        members = np.flatnonzero(bin_id == b)
        chosen[rng.choice(members, size=quota[b], replace=False)] = True
    assert chosen.sum() == total

    # Pass 2: extract the chosen rows.
    selected, offset = [], 0
    for path in args.inputs:
        pos = 0
        for chunk in read_chunks(path):
            mask = chosen[offset + pos: offset + pos + len(chunk)]
            if mask.any():
                selected.append(chunk.values[mask])
            pos += len(chunk)
        offset += pos
    data = np.concatenate(selected)
    assert len(data) == total
    data = data[rng.permutation(len(data))]

    frame = pd.DataFrame(data, columns=EXPECTED_COLUMNS)
    frame["HasIntersection"] = frame["HasIntersection"].astype(int)
    for out, part in ((args.eval_out, frame.iloc[:args.eval_size]), (args.train_out, frame.iloc[args.eval_size:])):
        part.to_csv(out, index=False, float_format="%.17g")
        cd = centroid_distance(part.values[:, :24])
        hist = np.histogram(np.log10(cd), bins=edges)[0]
        print(f"wrote {len(part):,} rows -> {out}; per-bin count min/max {hist.min()}/{hist.max()}", flush=True)


if __name__ == "__main__":
    main()
