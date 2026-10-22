import os

import numpy as np
import pandas as pd
import torch


class TetrahedronDataset(torch.utils.data.Dataset):
    def __init__(self, data_source):
        if isinstance(data_source, str):
            df = pd.read_csv(data_source)
            print(f"   Loaded {len(df):,} samples from {os.path.basename(data_source)}")
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            print(f"   Using DataFrame with {len(df):,} samples")
        else:
            raise ValueError("data_source must be either a file path (str) or pandas DataFrame")
        
        self.tetrahedron_features = df.iloc[:, :-2].values.astype(np.float64)
        self.intersection_volume = df.iloc[:, -2].values.astype(np.float64)
        self.intersection_status = df.iloc[:, -1].values.astype(np.float64)
        

        self.tetrahedron_features = torch.tensor(self.tetrahedron_features, dtype=torch.float64)
        self.intersection_status = torch.tensor(self.intersection_status, dtype=torch.float64).reshape(-1, 1)
        self.intersection_volume = torch.tensor(self.intersection_volume, dtype=torch.float64).reshape(-1, 1)
        
        # Logs
        intersecting_count = (self.intersection_status > 0.5).sum().item()
        non_intersecting_count = len(self.intersection_status) - intersecting_count

        positive_volumes = self.intersection_volume[self.intersection_volume > 0]
        if len(positive_volumes) > 0:
            print(f"   Volume range: [{positive_volumes.min():.2e}, {positive_volumes.max():.2e}]")
        
        print(f"   Features shape: {self.tetrahedron_features.shape}")
        print(f"   Intersecting pairs: {intersecting_count:,} ({intersecting_count/len(self.intersection_status)*100:.1f}%)")
        print(f"   Non-intersecting pairs: {non_intersecting_count:,}")
        
        if intersecting_count > 0:
            intersecting_volumes = self.intersection_volume[self.intersection_status.squeeze() > 0.5]
            positive_volumes = intersecting_volumes[intersecting_volumes > 0]
            print(f"   Positive volumes: {len(positive_volumes):,}/{intersecting_count:,}")

    def __len__(self):
        return len(self.tetrahedron_features)

    def __getitem__(self, idx):
        return (
            self.tetrahedron_features[idx], 
            self.intersection_status[idx], 
            self.intersection_volume[idx]
        )
