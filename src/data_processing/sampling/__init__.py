from src.data_processing.config import ProcessorConfig
from src.data_processing.sampling.base import SampleResult
from src.data_processing.sampling.log_uniform import LogUniformVolumeSampler
from src.data_processing.sampling.regular import RegularSampler

__all__ = ["SampleResult", "create_sampler"]


def create_sampler(intersection_type: str, config: ProcessorConfig):
    """Polyhedron intersections are volume-binned; every other type is sampled uniformly."""
    if intersection_type == "polyhedron_intersection":
        return LogUniformVolumeSampler(config)
    return RegularSampler(config)
