from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime

class ArtifactType(Enum):
    MODEL = "model"
    METRICS = "metrics"
    VISUALIZATION = "visualization"
    REPORT = "report"
    CONFIG = "config"
    LOG = "log"
    DATA = "data"

class StorageLevel(Enum):
    MINIMAL = "minimal"      # Essential files only
    STANDARD = "standard"    # Balanced save
    COMPLETE = "complete"    # Everything including debug info

@dataclass
class ArtifactMetadata:
    name: str
    type: ArtifactType
    format: str
    size_bytes: int
    checksum: str
    created_at: datetime.datetime
    tags: List[str]
    description: Optional[str] = None
    relationships: Optional[Dict[str, str]] = None

@dataclass
class ExperimentContext:
    experiment_id: str
    timestamp: datetime.datetime
    git_commit: Optional[str]
    system_info: Dict[str, Any]
    data_fingerprint: Optional[str]
    config_hash: str

# ============================================================================
# 2. ARTIFACT HANDLER SYSTEM
# ============================================================================

class ArtifactHandler(ABC):
    """Base class for handling specific artifact types"""
    
    @abstractmethod
    def can_handle(self, artifact: Any) -> bool:
        pass
    
    @abstractmethod
    def save(self, artifact: Any, path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_artifact_type(self) -> ArtifactType:
        pass
