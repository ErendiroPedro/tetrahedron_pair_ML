import os
import pandas as pd
import numpy as np
import src.GeometryUtils as gu
import torch

class CDataProcessor:
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

        print("-- Data Processed --")

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

        if original_data is None or original_data.empty:
            raise ValueError(f"No data available for {split} split.")

        original_count = original_data.shape[0]
        total_augmented = point_augmented_count + tetra_augmented_count

        if total_augmented > original_count:
            raise ValueError(f"Not enough original {split} samples to generate augmentations. Needed {total_augmented}, have {original_count}.")

        indices = np.random.permutation(original_count)

        point_indices = indices[:point_augmented_count]
        tetra_indices = indices[point_augmented_count : point_augmented_count + tetra_augmented_count]

        point_samples = original_data.iloc[point_indices, :-2]
        tetra_samples = original_data.iloc[tetra_indices, :-2]

        point_labels = original_data.iloc[point_indices, -2:]
        tetra_labels = original_data.iloc[tetra_indices, -2:]

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

        assert combined_data.shape[0] == original_count + total_augmented, "Data size mismatch after augmentation."
        
        # Update the appropriate dataset
        if split == 'train':
            self.train_data = combined_data
        elif split == 'val':
            self.val_data = combined_data
        
    def _load_data_for_intersection_type(self, raw_data_path, intersection_type):
        """Loads raw data for a specific intersection type."""
        folder_path = os.path.join(raw_data_path, intersection_type)
        raw_data_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]
        raw_data_list = [pd.read_csv(file) for file in raw_data_files]
        raw_data = pd.concat(raw_data_list, ignore_index=True)
        return raw_data.sample(frac=1, random_state=42).reset_index(drop=True)

    def _uniform_sample_by_volume(self, raw_data, percentage, num_train_samples, num_val_samples):
        """Uniformly samples data based on intersection volume with dynamic bin allocation."""
        volume_range = self.config["volume_range"]
        nbins = self.config["number_of_bins"]
        
        # Create volume bins
        bin_edges = np.linspace(volume_range[0], volume_range[1], nbins + 1)
        raw_data['bin'] = pd.cut(raw_data["IntersectionVolume"], bins=bin_edges, labels=False)
        
        # Calculate total targets
        total_train = int((percentage / 100) * num_train_samples)
        total_val = int((percentage / 100) * num_val_samples)
        
        # Phase 1: Initial proportional allocation
        bin_counts = raw_data['bin'].value_counts().sort_index()
        train_samples = []
        val_samples = []
        
        # Calculate ideal samples per bin
        train_per_bin = total_train / nbins
        val_per_bin = total_val / nbins
        
        # Dynamic allocation with redistribution
        remaining_train = total_train
        remaining_val = total_val
        
        for bin_idx in range(nbins):
            bin_population = bin_counts.get(bin_idx, 0)
            
            # Calculate actual possible allocation
            train_alloc = min(bin_population, max(1, int(train_per_bin)))
            val_alloc = min(bin_population - train_alloc, max(1, int(val_per_bin)))
            
            # Sample from bin
            bin_data = raw_data[raw_data['bin'] == bin_idx]
            
            # Train samples
            train = bin_data.sample(n=train_alloc, random_state=42+bin_idx)
            train_samples.append(train)
            remaining_train -= train_alloc
            
            # Validation samples
            remaining_in_bin = bin_data.drop(train.index)
            val = remaining_in_bin.sample(n=val_alloc, random_state=24+bin_idx)
            val_samples.append(val)
            remaining_val -= val_alloc
            
            # Remove used data
            raw_data = raw_data.drop(train.index).drop(val.index)
        
        # Phase 2: Redistribute remaining needs
        if remaining_train > 0 or remaining_val > 0:
            # Calculate weights based on remaining population
            weights = raw_data['bin'].map(bin_counts - train_per_bin - val_per_bin).fillna(0)
            weights = weights / weights.sum()
            
            # Sample remaining train
            if remaining_train > 0:
                add_train = raw_data.sample(n=remaining_train, weights=weights, 
                                        random_state=52, replace=False)
                train_samples.append(add_train)
                raw_data = raw_data.drop(add_train.index)
            
            # Sample remaining val
            if remaining_val > 0:
                add_val = raw_data.sample(n=remaining_val, weights=weights, 
                                        random_state=62, replace=False)
                val_samples.append(add_val)
        
        # Combine and shuffle
        train_data = pd.concat(train_samples).sample(frac=1, random_state=42)
        val_data = pd.concat(val_samples).sample(frac=1, random_state=42)

        # Drop the bin column before returning
        train_data = train_data.drop(columns=['bin'])
        val_data = val_data.drop(columns=['bin'])
        
        return train_data.reset_index(drop=True), val_data.reset_index(drop=True)

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
    def augment_data(data: pd.DataFrame, config) -> pd.DataFrame:
        """Applies augmentations to training data."""

        y_sorting = config["augmentations"].get("y_sorting")
        if y_sorting:
            if "difficulty_based" in y_sorting and y_sorting["difficulty_based"] is not None:
                difficulty_order = y_sorting["difficulty_based"]
                if difficulty_order == "harder_first":
                    data = gu.sort_by_difficulty_stepwise(data, easier_first=False)
                elif difficulty_order == "easier_first":
                    data = gu.sort_by_difficulty_stepwise(data, easier_first=True)  
                else:
                    raise ValueError("Invalid difficulty_order specified. Use 'harder_first' or 'easier_first'.")
            elif "spatial_based" in y_sorting and y_sorting["spatial_based"] is not None:
                # Placeholder for spatial-based sorting
                pass
                # data = gu.sort_by_spatial_based_y(data)
            # If y_sorting exists but all values are null, do nothing (random order)

        # if sort_type:
        #     if sort_type == "x_whole_dataset":
        #         data = gu.sort_by_x_coordinate(data)
        #     elif sort_type == "morton_code_whole_dataset":
        #         data = gu.sort_by_morton_code(data)
        #     elif sort_type == "x_each_tetrahedron":
        #         data = gu.sort_by_x_coordinate_alt(data)
        #     elif sort_type == "morton_code_each_tetrahedron":
        #         data = gu.sort_by_morton_code_alt(data)
        #     elif sort_type == "sort_by_intersection_volume_whole_dataset":
        #         data = gu.sort_by_intersection_volume_whole_dataset(data)
        #     else:
        #         raise ValueError("Invalid sort augmentation specified.")
                
        x_sorting = config["augmentations"].get("x_sorting")
        if x_sorting:
            if "volume_based" in x_sorting and x_sorting["volume_based"] is not None:
                volume_order = x_sorting["volume_based"]
                if volume_order == "bigger_first":
                    data = gu.volume_reordering(data, larger=True)
                elif volume_order == "smaller_first":
                    data = gu.volume_reordering(data, larger=False)
                else:
                    raise ValueError("Invalid volume_order specified. Use 'bigger_first' or 'smaller_first'.")
            elif "spatial_based" in x_sorting and x_sorting["spatial_based"] is not None:
                # Placeholder for spatial-based x sorting
                pass
            
        return data

    @staticmethod
    def transform_data(data: pd.DataFrame, config) -> pd.DataFrame:
        """Transforms data based on the configuration."""

        transformation_config = config.get("transformations", None)

        volume_scale_factor = config.get("volume_scale_factor", 1)
        if volume_scale_factor != 1:
            volume_column_idx = data.columns.get_loc("IntersectionVolume")
            if volume_column_idx >= 0:
                data.iloc[:, volume_column_idx] = data.iloc[:, volume_column_idx] * volume_scale_factor

        if not transformation_config:
            return data

        if isinstance(transformation_config, str):
            transformations_list = [transformation_config]
        elif isinstance(transformation_config, list):
            transformations_list = transformation_config
        else:
            raise TypeError("Transformations config must be a string or a list.")

        if not transformations_list:
            raise("No transformations specified.")
                
        
        features = data.iloc[:, :-2]
        labels = data.iloc[:, -2:]
        transformed_features = features

        if len(transformations_list) > 1:
 
            order = {
            "principal_axis_transformation": 0,
            "unitary_tetrahedron_transformation": 1
            }

            transformations_list.sort(key=lambda x: order.get(x, float('inf')))

            for t in transformations_list:
                if t not in order:
                    raise ValueError(f"Unknown or unsupported transformation type '{t}' found in config.")

        for transformation_type in transformations_list:
            if transformation_type == "unitary_tetrahedron_transformation":    
                transformed_features = transformed_features.apply(
                    gu.apply_unitary_tetrahedron_transformation, 
                    axis=1,
                    result_type='expand'
                )

            elif transformation_type == "principal_axis_transformation":               
                transformed_features = transformed_features.apply(
                    gu.apply_principal_axis_transformation, 
                    axis=1,
                    result_type='expand'
                )

            else:
                raise ValueError("Invalid transformation type specified.")
                
            
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
