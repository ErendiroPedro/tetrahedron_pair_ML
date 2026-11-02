from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.data_processing.transformations import TRANSFORMATIONS


def parse_transformations(value) -> Tuple[str, ...]:
    """Accepts None, a single name, or a list of names; returns validated names."""
    if value is None:
        return ()
    names = [value] if isinstance(value, str) else list(value)
    names = tuple(n for n in names if n and n.strip())
    unknown = [n for n in names if n not in TRANSFORMATIONS]
    if unknown:
        raise ValueError(f"Unknown transformations {unknown}; available: {sorted(TRANSFORMATIONS)}")
    return names


@dataclass(frozen=True)
class ProcessorConfig:
    raw_data_path: str
    processed_data_path: str
    num_train_samples: int
    num_val_samples: int
    intersection_distributions: Dict[str, float] = field(default_factory=dict)
    volume_scale_factor: float = 1.0
    n_volume_bins: int = 100
    volume_range: Optional[Tuple[float, float]] = None
    transformations: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict) -> "ProcessorConfig":
        if raw is None:
            raise ValueError("processor_config cannot be None. Please provide a valid configuration.")

        binning = raw.get("volume_binning") or {}
        volume_range = binning.get("volume_range")

        return cls(
            raw_data_path=raw["dataset_paths"]["raw_data"],
            processed_data_path=raw["dataset_paths"]["processed_data"],
            num_train_samples=raw["num_train_samples"],
            num_val_samples=raw["num_val_samples"],
            intersection_distributions=dict(raw.get("intersection_distributions") or {}),
            volume_scale_factor=float(raw.get("volume_scale_factor", 1.0)),
            n_volume_bins=binning.get("n_bins", 100),
            volume_range=(float(volume_range[0]), float(volume_range[1])) if volume_range else None,
            transformations=parse_transformations(raw.get("transformations")),
        )
