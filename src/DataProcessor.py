import os
import pandas as pd
import numpy as np
import src.GeometryUtils as gu

class DataProcessor:
    def __init__(self, processor_config):
        self.config = processor_config

    def process(self):
        if self.config["skip_processing"]:
            print("-- Skipped Processing Data --")
            return
            
        print("-- Processing Data --")
        train_data, val_data = self._load_data()
        # train_data = self._augment_data(train_data)
        self._save_data(train_data, val_data)

    def _load_data(self):
        """Main method to load and sample raw data."""
        raw_data_path = self.config["dataset_paths"]["raw_data"]
        intersection_distributions = self.config["intersection_distributions"]
        num_train_samples = self.config["num_train_samples"]
        num_val_samples = self.config["num_val_samples"]

        train_data_list, val_data_list = [], []

        for intersection_type, percentage in intersection_distributions.items():
            if percentage == 0:
                continue

            # Load and sample data for the intersection type
            raw_data = self._load_data_for_intersection_type(raw_data_path, intersection_type)

            if intersection_type == "polyhedron_intersection":
                train_data, val_data = self._uniform_sample_by_volume(
                    raw_data, percentage, num_train_samples, num_val_samples
                )
            else:
                train_data, val_data = self._sample_data(
                    raw_data, percentage, num_train_samples, num_val_samples
                )

            train_data_list.append(train_data)
            val_data_list.append(val_data)

        # Combine all intersection types
        train_data = self._combine_and_shuffle_data(train_data_list)
        val_data = self._combine_and_shuffle_data(val_data_list)

        return train_data, val_data

    def _load_data_for_intersection_type(self, raw_data_path, intersection_type):
        """Loads raw data for a specific intersection type."""
        folder_path = os.path.join(raw_data_path, intersection_type)
        raw_data_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]
        raw_data_list = [pd.read_csv(file) for file in raw_data_files]
        raw_data = pd.concat(raw_data_list, ignore_index=True)
        return raw_data.sample(frac=1, random_state=42).reset_index(drop=True)

    def _uniform_sample_by_volume(self, raw_data, percentage, num_train_samples, num_val_samples):
        """Uniformly samples data based on intersection volume."""
        volume_range = self.config["volume_range"]
        nbins = self.config["number_of_bins"]

        # Calculate total samples for train and validation
        total_train_samples = int((percentage / 100) * num_train_samples)
        total_val_samples = int((percentage / 100) * num_val_samples)

        # Define bin edges
        bin_edges = np.linspace(volume_range[0], volume_range[1], nbins + 1)

        train_samples_per_bin = total_train_samples // nbins
        val_samples_per_bin = total_val_samples // nbins

        train_data_list, val_data_list = [], []

        for i in range(nbins):
            # Define bin range
            bin_min, bin_max = bin_edges[i], bin_edges[i + 1]

            # Filter data within the bin range
            bin_data = raw_data[(raw_data["IntersectionVolume"] >= bin_min) &
                            (raw_data["IntersectionVolume"] < bin_max)]

            # Sample training data
            bin_train_data = bin_data.sample(
                n=min(train_samples_per_bin, len(bin_data)),
                replace=len(bin_data) < train_samples_per_bin,
                random_state=42 + i  # Different random state for each bin
            )

            # Remove training samples from the pool before sampling validation
            remaining_data = bin_data[~bin_data.index.isin(bin_train_data.index)]

            # Then sample validation data from remaining data
            bin_val_data = remaining_data.sample(
                n=min(val_samples_per_bin, len(remaining_data)),
                replace=len(remaining_data) < val_samples_per_bin,
                random_state=42 + nbins + i  # Different random state for validation
            )

            train_data_list.append(bin_train_data)
            val_data_list.append(bin_val_data)

        # Combine sampled data from all bins
        train_data = pd.concat(train_data_list, ignore_index=True).sample(frac=1, random_state=42)
        val_data = pd.concat(val_data_list, ignore_index=True).sample(frac=1, random_state=43)

        return train_data, val_data
    
    def _sample_data(self, raw_data, percentage, num_train_samples, num_val_samples):
        """Samples data based on the given percentage."""
        num_train = int((percentage / 100) * num_train_samples)
        num_val = int((percentage / 100) * num_val_samples)
        train_data = raw_data.iloc[:num_train]
        val_data = raw_data.iloc[num_train:num_train + num_val]
        return train_data, val_data

    def _combine_and_shuffle_data(self, data_list):
        """Combines and shuffles a list of dataframes."""
        combined_data = pd.concat(data_list, ignore_index=True)
        return combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    def _augment_data(self, train_data):
        """Applies augmentations to training data."""
        if self.config["augmentations"]["sort"]:
            sort_type = self.config["augmentations"]["sort"]
            if sort_type == "X":
                train_data = gu.sort_by_X_coordinate(train_data)
            elif sort_type == "SFC":
                train_data = gu.sort_by_space_filling_curve(train_data)
            else:
                raise ValueError("Invalid sort augmentation specified.")

        if self.config["augmentations"]["larger_tetrahedron_first"]:
            train_data = gu.larger_tetrahedron_first(train_data)

        # Placeholder for additional augmentations
        if self.config["augmentations"]["vertex_permutation_augmentation_pct"] > 0:
            pass
        if self.config["augmentations"]["tetrahedron_permutation_augmentation_pct"] > 0:
            pass
        if self.config["augmentations"]["rigid_transformation_augmentation_pct"] > 0:
            pass
        if self.config["augmentations"]["affine_linear_transformation_augmentation_pct"] > 0:
            pass

        return train_data

    def _save_data(self, train_data, val_data):
        """Saves processed training and validation data in a structured folder layout."""
        processed_data_path = self.config["dataset_paths"]["processed_data"]

        # Define subdirectories for train and validation
        train_data_path = os.path.join(processed_data_path, "train")
        val_data_path = os.path.join(processed_data_path, "val")

        # Create directories if they don't exist
        os.makedirs(train_data_path, exist_ok=True)
        os.makedirs(val_data_path, exist_ok=True)

        # Save training and validation data
        train_data_file = os.path.join(train_data_path, "train_data.csv")
        val_data_file = os.path.join(val_data_path, "val_data.csv")
        train_data.to_csv(train_data_file, index=False)
        val_data.to_csv(val_data_file, index=False)

        # Optionally log the save operation
        print(f"Training data saved to: {train_data_file}")
        print(f"Validation data saved to: {val_data_file}")