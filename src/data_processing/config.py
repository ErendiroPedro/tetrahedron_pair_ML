from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.data_processing.transformations import TRANSFORMATIONS

SAMPLING_STRATEGIES = ("log_uniform_volume",)


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
    sampling_strategy: str = "log_uniform_volume"
    n_volume_bins: int = 10
    volume_range: Optional[Tuple[float, float]] = None
    point_wise_pct: float = 0
    tetrahedron_wise_pct: float = 0
    transformations: Tuple[str, ...] = ()

    @property
    def has_augmentation(self) -> bool:
        return self.point_wise_pct > 0 or self.tetrahedron_wise_pct > 0

    @classmethod
    def from_dict(cls, raw: dict) -> "ProcessorConfig":
        if raw is None:
            raise ValueError("processor_config cannot be None. Please provide a valid configuration.")

        binning = raw.get("volume_binning") or {}
        volume_range = binning.get("volume_range")
        augmentations = raw.get("augmentations") or {}
        sampling_strategy = raw.get("sampling_strategy", "log_uniform_volume")
        if sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"Unsupported sampling_strategy '{sampling_strategy}'; available: {SAMPLING_STRATEGIES}")

        return cls(
            raw_data_path=raw["dataset_paths"]["raw_data"],
            processed_data_path=raw["dataset_paths"]["processed_data"],
            num_train_samples=raw["num_train_samples"],
            num_val_samples=raw["num_val_samples"],
            intersection_distributions=dict(raw.get("intersection_distributions") or {}),
            volume_scale_factor=float(raw.get("volume_scale_factor", 1.0)),
            sampling_strategy=sampling_strategy,
            n_volume_bins=binning.get("n_bins", 10),
            volume_range=(float(volume_range[0]), float(volume_range[1])) if volume_range else None,
            point_wise_pct=augmentations.get("point_wise_permutation_augmentation_pct", 0),
            tetrahedron_wise_pct=augmentations.get("tetrahedron_wise_permutation_augmentation_pct", 0),
            transformations=parse_transformations(raw.get("transformations")),
        )
