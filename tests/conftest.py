import copy

import numpy as np
import pandas as pd
import pytest

from src.data_processing import DataProcessor
from src.utils.dataset_files import EXPECTED_COLUMNS, INTERSECTION_TYPES

TASK = "IntersectionStatus_IntersectionVolume"
TEST_ROWS_PER_DATASET = 150


def write_dataset(path, rng, intersection_type, rows, volume_range=(1e-5, 1e-2)):
    """Random tetrahedron pairs whose labels look like the real data (one intersection type per file)."""
    frame = pd.DataFrame(rng.random((rows, 24)), columns=EXPECTED_COLUMNS[:24])
    if intersection_type == "no_intersection":
        frame["IntersectionVolume"], frame["HasIntersection"] = 0.0, 0
    else:
        volume = np.exp(rng.uniform(np.log(volume_range[0]), np.log(volume_range[1]), rows))
        frame["IntersectionVolume"] = np.where(rng.random(rows) < 0.3, 0.0, volume)  # some pairs only touch
        frame["HasIntersection"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_config(root):
    processed = str(root / "processed")
    layers = {"per_vertex_layers": [4, 4], "per_tetrahedron_layers": [4, 4], "per_two_tetrahedra_layers": [4, 4],
              "shared_layers": [4], "classification_head": [4, 1], "regression_head": [4, 1],
              "vertices_aggregation_function": "max", "tetrahedra_aggregation_function": "max"}
    return {
        "artifacts_config": {"save_artifacts_to": str(root / "artifacts")},
        "processor_config": {
            "skip_processing": False,
            "num_train_samples": 200,
            "num_val_samples": 100,
            "volume_scale_factor": 1000.0,
            "volume_binning": {"n_bins": 5, "volume_range": [1e-4, 1e-2]},
            "dataset_paths": {"raw_data": str(root / "raw"), "processed_data": processed},
            "intersection_distributions": {"no_intersection": 40, "point_intersection": 10, "segment_intersection": 10,
                                           "polygon_intersection": 20, "polyhedron_intersection": 20},
            "transformations": ["principal_axis"],
        },
        "model_config": {
            "skip_building": False,
            "processed_data_path": processed,
            "common_parameters": {"activation_function": "relu", "dropout_rate": 0.0, "task": TASK, "volume_scale_factor": 1000.0},
            "architecture": {
                "use_model": "tetrahedronpairnet",
                "tetrahedronpairnet": layers,
                "mlp": {"shared_layers": [8], "classification_head": [8, 1], "regression_head": [8, 1]},
            },
        },
        "trainer_config": {"skip_training": False, "device": "cpu", "processed_data_path": processed,
                           "batch_size": 64, "epochs": 2, "learning_rate": 1e-3, "task": TASK},
        "evaluator_config": {
            "skip_evaluation": False, "device": "cpu", "task": TASK, "test_data_path": str(root / "test_data"),
            "regression_consistency_thresholds": {"atol": 1e-3, "rtol": 0.1},
            "volume_range": [0, 0.01], "evaluation_n_bins": 4, "volume_scale_factor": 1000.0,
        },
    }


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """Synthetic raw data, test data and a processed train/val split, built once; returns a valid base config."""
    root = tmp_path_factory.mktemp("workspace")
    rng = np.random.default_rng(0)
    for intersection_type in INTERSECTION_TYPES:
        raw_rows = 3000 if intersection_type == "polyhedron_intersection" else 400
        write_dataset(root / "raw" / intersection_type / "part.csv", rng, intersection_type, raw_rows)
        write_dataset(root / "test_data" / intersection_type / "test.csv", rng, intersection_type, TEST_ROWS_PER_DATASET)
    config = build_config(root)
    DataProcessor(config["processor_config"]).process()
    return config


@pytest.fixture
def config(workspace, tmp_path):
    """A valid, independent copy of the base config whose artifacts go to this test's temporary folder."""
    config = copy.deepcopy(workspace)
    config["artifacts_config"]["save_artifacts_to"] = str(tmp_path / "artifacts")
    return config
