import pandas as pd

SAMPLE_SIZE = 5000


def report_distribution(file_path: str, dataset_name: str) -> None:
    """Print label balance and intersection-volume statistics for the head of a processed file."""
    try:
        sample = pd.read_csv(file_path, nrows=SAMPLE_SIZE)
        if sample.empty:
            print(f"{dataset_name} dataset is empty!")
            return

        print(f"{dataset_name} distribution (first {len(sample)} samples):")
        for label, count in sample["HasIntersection"].value_counts().items():
            name = "No intersection" if label == 0 else "Has intersection"
            print(f"  {name}: {count} samples ({count / len(sample) * 100:.1f}%)")

        volumes = sample.loc[sample["HasIntersection"] == 1, "IntersectionVolume"]
        if len(volumes) > 0:
            print("  Intersection volumes:")
            print(f"    min={volumes.min():.8e}, max={volumes.max():.8e}")
            print(f"    mean={volumes.mean():.8e}, std={volumes.std():.8e}")
            print(f"    Samples with volume > 0: {(volumes > 0).sum()}")

    except Exception as e:
        print(f"Could not check {dataset_name} distribution: {e}")
