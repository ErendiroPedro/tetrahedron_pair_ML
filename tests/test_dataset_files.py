import pytest

from src.utils.dataset_files import EXPECTED_COLUMNS, count_samples, discover_csv_files, processed_split_path


@pytest.mark.parametrize("content, rows", [
    ("h\n1\n2\n", 2),
    ("h\n1\n2", 2),  # last row without a trailing newline still counts
    ("h\n", 0),
    ("h", 0),
    ("", 0),
])
def test_count_samples_excludes_the_header(tmp_path, content, rows):
    path = tmp_path / "data.csv"
    path.write_text(content)
    assert count_samples(str(path)) == rows


def test_count_samples_handles_files_larger_than_one_read_chunk(tmp_path):
    path = tmp_path / "big.csv"
    rows = 30_000
    path.write_text("header\n" + "\n".join("x" * 100 for _ in range(rows)))  # ~3 MB, no trailing newline
    assert count_samples(str(path)) == rows


def test_discover_csv_files_returns_sorted_csv_files_only(tmp_path):
    for name in ("b.csv", "a.csv", "notes.txt"):
        (tmp_path / name).write_text("x")
    assert discover_csv_files(str(tmp_path)) == [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]


def test_discover_csv_files_fails_for_a_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_csv_files(str(tmp_path / "missing"))


def test_processed_split_path():
    assert processed_split_path("data/processed", "val") == "data/processed/val/val_data.csv"


def test_expected_columns_are_the_24_coordinates_then_the_two_labels():
    assert len(EXPECTED_COLUMNS) == 26
    assert EXPECTED_COLUMNS[0] == "T1_v1_x" and EXPECTED_COLUMNS[23] == "T2_v4_z"
    assert EXPECTED_COLUMNS[24:] == ["IntersectionVolume", "HasIntersection"]
