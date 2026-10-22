import numpy as np
import torch

# Import-time globals inherited from the old CDataProcessor module.
torch.set_default_dtype(torch.float64)
np.seterr(under="ignore")

from src.data_processing.config import ProcessorConfig
from src.data_processing.processor import CDataProcessor
from src.data_processing.transformations import transform_data

__all__ = ["CDataProcessor", "ProcessorConfig", "transform_data"]
