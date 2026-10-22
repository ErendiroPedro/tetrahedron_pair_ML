from typing import Dict, Iterable, Tuple


def allocate_counts(
    train_total: int,
    val_total: int,
    distributions: Dict[str, float],
    available_types: Iterable[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Split train/val totals across intersection types by percentage.

    Types are processed from largest share to smallest; the last configured one absorbs the
    rounding remainder so the totals are exact. Types without a configured share get 0.
    """
    total_pct = sum(distributions.values())
    if abs(total_pct - 100.0) > 0.001:
        raise ValueError(f"Intersection distributions must sum to 100%, got {total_pct}%")

    available = list(available_types)
    train_targets = {t: 0 for t in available}
    val_targets = {t: 0 for t in available}

    configured = sorted((t for t in available if t in distributions), key=lambda t: distributions[t], reverse=True)
    allocated_train = allocated_val = 0
    for i, t in enumerate(configured):
        if i == len(configured) - 1:
            train_targets[t] = train_total - allocated_train
            val_targets[t] = val_total - allocated_val
        else:
            share = distributions[t] / 100.0
            train_targets[t] = int(train_total * share)
            val_targets[t] = int(val_total * share)
            allocated_train += train_targets[t]
            allocated_val += val_targets[t]

    if sum(train_targets.values()) != train_total or sum(val_targets.values()) != val_total:
        missing = sorted(set(distributions) - set(available))
        raise ValueError(
            f"Sample allocation error: got {sum(train_targets.values())}+{sum(val_targets.values())}, "
            f"expected {train_total}+{val_total}"
            + (f" (no data found for configured types: {missing})" if missing else "")
        )
    return train_targets, val_targets
