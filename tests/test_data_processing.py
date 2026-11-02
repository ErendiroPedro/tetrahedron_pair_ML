import numpy as np
import pandas as pd
import pytest

from src.data_processing import DataProcessor, transform_data
from src.data_processing.config import parse_transformations
from src.data_processing.sampling import allocate_counts, sample_log_uniform
from src.utils.dataset_files import EXPECTED_COLUMNS


def run_processor(config, output_dir):
    config["processor_config"]["dataset_paths"]["processed_data"] = str(output_dir)
    DataProcessor(config["processor_config"]).process()
    return pd.read_csv(output_dir / "train/train_data.csv"), pd.read_csv(output_dir / "val/val_data.csv")


def test_allocate_counts_gives_exact_totals_and_nothing_to_unconfigured_types():
    train, val = allocate_counts(101, 33, {"a": 50, "b": 30, "c": 20}, ["a", "b", "c", "d"])
    assert sum(train.values()) == 101 and sum(val.values()) == 33
    assert train["d"] == 0 and val["d"] == 0


def test_allocate_counts_rejects_shares_that_do_not_sum_to_100():
    with pytest.raises(ValueError, match="sum to 100"):
        allocate_counts(100, 10, {"a": 50, "b": 40}, ["a", "b"])


def test_processor_writes_exact_counts_scaled_volumes_and_the_expected_mix(config, tmp_path):
    train, val = run_processor(config, tmp_path / "processed")

    assert (len(train), len(val)) == (200, 100)
    assert list(train.columns) == EXPECTED_COLUMNS
    assert ((train.HasIntersection == 0).sum(), (val.HasIntersection == 0).sum()) == (80, 40)  # 40% no_intersection
    assert (train.loc[train.HasIntersection == 0, "IntersectionVolume"] == 0).all()
    assert train.IntersectionVolume.max() <= 1e-2 * 1000 * (1 + 1e-9), "volumes are scaled by volume_scale_factor"


def test_processor_is_deterministic(config, tmp_path):
    run_processor(config, tmp_path / "first")
    run_processor(config, tmp_path / "second")
    for split in ("train", "val"):
        assert (tmp_path / "first" / split / f"{split}_data.csv").read_bytes() == \
               (tmp_path / "second" / split / f"{split}_data.csv").read_bytes()


def test_processor_applies_the_principal_axis_transformation(config, tmp_path):
    train, _ = run_processor(config, tmp_path / "processed")
    t1_centroid = train.iloc[:, :12].to_numpy().reshape(-1, 4, 3).mean(axis=1)
    assert np.abs(t1_centroid).max() < 1e-12


def test_log_uniform_sampling_fills_volume_bins_equally_without_overlap(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((5000, 24)), columns=EXPECTED_COLUMNS[:24])
    frame["IntersectionVolume"] = np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), len(frame)))
    frame["HasIntersection"] = 1
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)

    train_path, val_path = sample_log_uniform([str(source)], 100, 50, str(tmp_path / "sample"), (1e-4, 1e-2), n_bins=4)
    train, val = pd.read_csv(train_path), pd.read_csv(val_path)

    assert (len(train), len(val)) == (100, 50)
    assert set(map(tuple, train.iloc[:, :24].round(12).values)).isdisjoint(map(tuple, val.iloc[:, :24].round(12).values))
    edges = np.logspace(-4, -2, 5)
    per_bin = np.histogram(pd.concat([train, val]).IntersectionVolume, bins=edges)[0]
    assert set(per_bin) <= {37, 38}, f"150 rows over 4 bins should be 37/38 each, got {per_bin}"


def test_transform_data_unitary_keeps_t1_columns_and_labels_only():
    frame = pd.DataFrame(np.random.default_rng(0).random((20, 26)), columns=EXPECTED_COLUMNS)
    out = transform_data(frame.copy(), ("unitary_tetrahedron",))
    assert list(out.columns) == [f"T1_v{v}_{axis}" for v in range(1, 5) for axis in "xyz"] + ["IntersectionVolume", "HasIntersection"]
    assert out[["IntersectionVolume", "HasIntersection"]].equals(frame[["IntersectionVolume", "HasIntersection"]])


def test_transform_data_drops_rows_with_invalid_coordinates():
    frame = pd.DataFrame(np.random.default_rng(0).random((20, 26)), columns=EXPECTED_COLUMNS).astype(object)
    frame.loc[3, "T1_v1_x"] = "oops"
    assert len(transform_data(frame, ("principal_axis",))) == 19


def test_transform_data_without_transformations_returns_the_data_unchanged():
    frame = pd.DataFrame(np.random.default_rng(0).random((5, 26)), columns=EXPECTED_COLUMNS)
    assert transform_data(frame, ()) is frame


@pytest.mark.parametrize("value, expected", [
    (None, ()), ("principal_axis", ("principal_axis",)),
    (["unitary_tetrahedron", "principal_axis"], ("unitary_tetrahedron", "principal_axis")), ([], ()),
])
def test_parse_transformations(value, expected):
    assert parse_transformations(value) == expected


def test_parse_transformations_rejects_unknown_names():
    with pytest.raises(ValueError, match="rotate"):
        parse_transformations(["rotate"])
