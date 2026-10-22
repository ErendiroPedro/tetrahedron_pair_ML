from typing import NamedTuple, Optional


class SampleResult(NamedTuple):
    """Temp files (None when empty) and row counts produced by a sampler for one intersection type."""

    train_file: Optional[str]
    val_file: Optional[str]
    train_count: int
    val_count: int
