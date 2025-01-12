import os
import pandas as pd
import src.GeometryUtils as gu

class DataProcessor:
    def __init__(self, processor_config):
        self.config = processor_config

    def process(self):
        train_data, val_data = self._load_and_sample_raw_data()
        train_data = self._apply_augmentations(train_data)
        self._save_processed_data(train_data, val_data)

    def _load_and_sample_raw_data(self):
        pass

    def _apply_augmentations(self, train_data):

        # Sorting
        if(self.config["augmentations"]["sort"]):
            if(self.config["augmentations"]["sort"] == "X"):
                train_data = gu.sort_by_X_coordinate(train_data)
            elif(self.config["augmentations"]["sort"] == "SFC"):
                train_data = gu.sort_by_space_filling_curve(train_data)
            else:
                raise ValueError("Invalid sort augmentation specified.")
            
        # Reordering
        if(self.config["augmentations"]["larger_tetrahedron_first"]):
            train_data = gu.larger_tetrahedron_first(train_data)
            
        # Permutation
        if(self.config["augmentations"]["vertex_permutation_augmentation_pct"] > 0):
            pass

        if(self.config["augmentations"]["tetrahedron_permutation_augmentation_pct"] > 0):
            pass

        # Transformation
        if(self.config["augmentations"]["rigid_transformation_augmentation_pct"] > 0):
            pass

        if(self.config["augmentations"]["affine_linear_transformation_augmentation_pct"] > 0):
            pass

    def _save_processed_data(self, train_data, val_data):
        pass