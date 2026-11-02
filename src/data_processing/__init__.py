import numpy as np
import torch

# Import-time numeric defaults for the whole pipeline.
torch.set_default_dtype(torch.float64)
np.seterr(under="ignore")

from src.data_processing.config import ProcessorConfig
from src.data_processing.processor import DataProcessor
from src.data_processing.transformations import transform_data

__all__ = ["DataProcessor", "ProcessorConfig", "transform_data"]
