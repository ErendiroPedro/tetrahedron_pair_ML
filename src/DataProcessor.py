import os
import pandas as pd
import numpy as np
import src.GeometryUtils as gu
import torch

class DataProcessor:
    def __init__(self, processor_config):
        self.config = processor_config
        self.train_data = None
        self.val_data = None

    def process(self): 
           
        print("-- Processing Data --")

        self._load_data()

        self.train_data = self.augment_data(self.train_data, self.config)
        self.val_data = self.augment_data(self.val_data, self.config)
        
        self.train_data = self.transform_data(self.train_data, self.config)
        self.val_data = self.transform_data(self.val_data, self.config)

        self._save_data()

        print("---- Data Processed ----")

    def _load_data(self):
        """Main method to load and sample raw data with augmentations."""
        raw_data_path = self.config["dataset_paths"]["raw_data"]
        intersection_distributions = self.config["intersection_distributions"]
        num_train_samples = self.config["num_train_samples"]
        num_val_samples = self.config["num_val_samples"]
        augmentations_config = self.config["augmentations"]
        point_pct = augmentations_config["point_wise_permutation_augmentation_pct"]
        tetra_pct = augmentations_config["tetrahedron_wise_permutation_augmentation_pct"]

        # Validate augmentation percentages
        total_aug_pct = point_pct + tetra_pct
        if total_aug_pct > 50:
            raise ValueError("Total augmentation percentage cannot exceed 50%")
        if total_aug_pct < 0:
            raise ValueError("Augmentation percentages cannot be negative")

        # Calculate augmented samples as percentage of ORIGINAL DATA
        original_num_train = int(num_train_samples * (100 - total_aug_pct) / 100)
        original_num_val = int(num_val_samples * (100 - total_aug_pct) / 100)

        # Calculate exact augmented counts
        augmented_train = num_train_samples - original_num_train
        augmented_val = num_val_samples - original_num_val

        # Load original data with reduced counts
        self.config["num_train_samples"] = original_num_train
        self.config["num_val_samples"] = original_num_val

        train_data_list, val_data_list = [], []

        for intersection_type, percentage in intersection_distributions.items():
            if percentage == 0:
                continue

            # Load and sample data
            raw_data = self._load_data_for_intersection_type(raw_data_path, intersection_type)

            if intersection_type == "polyhedron_intersection":
                train_data, val_data = self._uniform_sample_by_volume(
                    raw_data, percentage, original_num_train, original_num_val
                )
            else:
                train_data, val_data = self._sample_data(
                    raw_data, percentage, original_num_train, original_num_val
                )

            train_data_list.append(train_data)
            val_data_list.append(val_data)

        # Combine sampled data
        self.train_data = self._combine(train_data_list)
        self.val_data = self._combine(val_data_list)

        # Apply augmentations using percentages of ORIGINAL DATA size
        if augmented_train > 0:
            point_augmented_train = int(original_num_train * point_pct / 100)
            tetra_augmented_train = int(original_num_train * tetra_pct / 100)
            
            # Handle rounding errors
            total = point_augmented_train + tetra_augmented_train
            if total != augmented_train:
                remainder = augmented_train - total
                if point_pct >= tetra_pct:
                    point_augmented_train += remainder
                else:
                    tetra_augmented_train += remainder

            self._apply_augmentations('train', point_augmented_train, tetra_augmented_train)

        if augmented_val > 0:
            # Repeat same logic for validation
            point_augmented_val = int(original_num_val * point_pct / 100)
            tetra_augmented_val = int(original_num_val * tetra_pct / 100)
            
            total = point_augmented_val + tetra_augmented_val
            if total != augmented_val:
                remainder = augmented_val - total
                if point_pct >= tetra_pct:
                    point_augmented_val += remainder
                else:
                    tetra_augmented_val += remainder

            self._apply_augmentations('val', point_augmented_val, tetra_augmented_val)

        # Restore original config values
        self.config["num_train_samples"] = num_train_samples
        self.config["num_val_samples"] = num_val_samples

    def _apply_augmentations(self, split, point_augmented_count, tetra_augmented_count):
        if split == 'train':
            original_data = self.train_data
        else:
            original_data = self.val_data

        if original_data is None or original_data.shape[0] == 0:
            return

        original_count = original_data.shape[0]
        total_augmented = point_augmented_count + tetra_augmented_count

        # Check if we have enough original samples
        if total_augmented > original_count:
            raise ValueError(f"Not enough original {split} samples to generate augmentations. Needed {total_augmented}, have {original_count}.")

        # Shuffle the original data
        indices = np.random.permutation(original_count)

        # Split indices for each augmentation
        point_indices = indices[:point_augmented_count]
        tetra_indices = indices[point_augmented_count : point_augmented_count + tetra_augmented_count]

        # Extract samples for augmentation (excluding the last two columns which are labels)
        point_samples = original_data.iloc[point_indices, :-2]
        tetra_samples = original_data.iloc[tetra_indices, :-2]

        # Extract corresponding labels
        point_labels = original_data.iloc[point_indices, -2:]
        tetra_labels = original_data.iloc[tetra_indices, -2:]

        # Convert feature samples to tensors for augmentation
        point_tensor = torch.tensor(point_samples.values, dtype=torch.float32)
        tetra_tensor = torch.tensor(tetra_samples.values, dtype=torch.float32)

        # Apply augmentations to features
        augmented_point_tensor = gu.permute_points_within_tetrahedrons(point_tensor)
        augmented_tetra_tensor = gu.swap_tetrahedrons(tetra_tensor)

        # Convert augmented tensors back to DataFrames
        augmented_point = pd.DataFrame(augmented_point_tensor.numpy(), columns=point_samples.columns)
        augmented_tetra = pd.DataFrame(augmented_tetra_tensor.numpy(), columns=tetra_samples.columns)

        # Reattach labels to augmented data
        augmented_point = pd.concat([augmented_point, point_labels.reset_index(drop=True)], axis=1)
        augmented_tetra = pd.concat([augmented_tetra, tetra_labels.reset_index(drop=True)], axis=1)

        # Combine original and augmented data
        combined_data = pd.concat([original_data, augmented_point, augmented_tetra], axis=0)

        # Shuffle to mix augmented and original samples
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Update the appropriate dataset
        if split == 'train':
            self.train_data = combined_data
        else:
            self.val_data = combined_data

    def _load_data_for_intersection_type(self, raw_data_path, intersection_type):
        """Loads raw data for a specific intersection type."""
        folder_path = os.path.join(raw_data_path, intersection_type)
        raw_data_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]
        raw_data_list = [pd.read_csv(file) for file in raw_data_files]
        raw_data = pd.concat(raw_data_list, ignore_index=True)
        return raw_data.sample(frac=1, random_state=42).reset_index(drop=True)

    def _uniform_sample_by_volume(self, raw_data, percentage, num_train_samples, num_val_samples):
        """Uniformly samples data based on intersection volume with fallback to random sampling."""
        volume_range = self.config["volume_range"]
        nbins = self.config["number_of_bins"]

        # Calculate target samples for this intersection type
        total_train_samples = int((percentage / 100) * num_train_samples)
        total_val_samples = int((percentage / 100) * num_val_samples)

        # Create volume bins
        bin_edges = np.linspace(volume_range[0], volume_range[1], nbins + 1)
        
        # Initialize storage
        train_samples, val_samples = [], []
        leftover_pool = []

        # Phase 1: Uniform sampling from bins
        for i in range(nbins):
            # Get bin data
            bin_mask = (raw_data["IntersectionVolume"] >= bin_edges[i]) & \
                    (raw_data["IntersectionVolume"] < bin_edges[i+1])
            bin_data = raw_data[bin_mask]
            
            # Calculate target samples per bin
            train_target = total_train_samples // nbins
            val_target = total_val_samples // nbins

            # Sample training data
            n_train = min(train_target, len(bin_data))
            bin_train = bin_data.sample(n=n_train, replace=False, random_state=42+i)
            train_samples.append(bin_train)
            
            # Sample validation data from remaining
            remaining = bin_data.drop(bin_train.index)
            n_val = min(val_target, len(remaining))
            bin_val = remaining.sample(n=n_val, replace=False, random_state=42+nbins+i)
            val_samples.append(bin_val)
            
            # Collect leftovers
            leftover_pool.append(remaining.drop(bin_val.index))

        # Combine initial samples
        train_data = pd.concat(train_samples, ignore_index=True)
        val_data = pd.concat(val_samples, ignore_index=True)

        # Phase 2: Fill remaining needs from leftover pool
        leftover_pool = pd.concat(leftover_pool, ignore_index=True).sample(frac=1, random_state=44)
        
        # Calculate remaining needs
        remaining_train = max(0, total_train_samples - len(train_data))
        remaining_val = max(0, total_val_samples - len(val_data))

        # Fill training first
        if remaining_train > 0:
            add_train = leftover_pool.iloc[:remaining_train]
            train_data = pd.concat([train_data, add_train], ignore_index=True)
            leftover_pool = leftover_pool.iloc[remaining_train:]

        # Then fill validation
        if remaining_val > 0:
            add_val = leftover_pool.iloc[:remaining_val]
            val_data = pd.concat([val_data, add_val], ignore_index=True)

        # Final shuffle
        return (train_data.sample(frac=1, random_state=42).reset_index(drop=True),
                val_data.sample(frac=1, random_state=43).reset_index(drop=True))

    def _sample_data(self, raw_data, percentage, num_train_samples, num_val_samples):
        """Samples data based on the given percentage."""
        num_train = int((percentage / 100) * num_train_samples)
        num_val = int((percentage / 100) * num_val_samples)
        train_data = raw_data.iloc[:num_train]
        val_data = raw_data.iloc[num_train:num_train + num_val]
        return train_data, val_data

    def _combine(self, data_list):
        """Combines and shuffles a list of dataframes."""
        combined_data = pd.concat(data_list, ignore_index=True)
        return combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    @staticmethod
    def augment_data(data, config):
        """Applies augmentations to training data."""
        sort_type = config["augmentations"]["sort"]

        if sort_type:
            if sort_type == "x":
                data = gu.sort_by_X_coordinate(data)

            elif sort_type == "morton_code":
                data = gu.sort_by_morton_code(data)

            else:
                raise ValueError("Invalid sort augmentation specified.")

        return data

    @staticmethod
    def transform_data(data: pd.DataFrame, config) -> pd.DataFrame:
        """Transforms data based on the configuration."""

        if config["transformations"]["affine_linear_transformation"]:

            features = data.iloc[:, :-2]
            labels = data.iloc[:, -2:]
            
            transformed_features = features.apply(
                gu.apply_affine_linear_transformation, 
                axis=1,
                result_type='expand'
            )
            
            transformed_features.columns = [
                f'v{i//3 + 1}_{"xyz"[i % 3]}' for i in range(12)
            ]

            data = pd.concat([transformed_features, labels], axis=1)

        return data

    def _save_data(self):
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
        self.train_data.to_csv(train_data_file, index=False)
        self.val_data.to_csv(val_data_file, index=False)

        print(f"Training data saved to: {train_data_file}")
        print(f"Validation data saved to: {val_data_file}")
