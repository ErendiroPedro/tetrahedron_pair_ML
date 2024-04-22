import os
import json
import torch
from torch_geometric.data import Data, Dataset
from abc import ABC, abstractmethod

# class MeshProcessor(Dataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(MeshProcessor, self).__init__(root, transform, pre_transform)
#         self.last_processed_index_train = None
#         self.last_processed_index_val = None

#     @property
#     def raw_file_names(self):
#         return os.listdir(os.path.join(self.raw_dir))

class BaseProcessor(ABC, Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        Dataset.__init__(self, root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.last_processed_index_train = None
        self.last_processed_index_val = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _load_data(self, file_path):
        pass
    
    @abstractmethod
    def _process_item(self, item):
        pass

    def _get_file_paths(self, subset):
        return [os.path.join(self.raw_dir, subset, f) for f in os.listdir(os.path.join(self.raw_dir, subset))]
    
    def process_n_entries(self, subset, n):
        file_paths = self._get_file_paths(subset)
        if not file_paths:
            return []

        data = self._load_data(file_paths[0])  # Assumes processing from one file, enhance to handle multiple files

        last_index = self.last_processed_index_train if subset == 'train' else self.last_processed_index_val
        start_index = 0 if last_index is None else last_index + 1
        end_index = min(start_index + n, len(data))
        data_list = []

        for i in range(start_index, end_index):
            item = data[i]
            processed_item = self._process_item(item)
            data_list.append(processed_item)

        if subset == 'train':
            self.last_processed_index_train = end_index - 1
        else:
            self.last_processed_index_val = end_index - 1

        return data_list

class GraphProcessor(BaseProcessor, Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.data_length = 0

    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.data_length = len(data)
        return data
    
    def _process_item(self, item):
        tetra1 = item['tetrahedron_1']
        tetra2 = item['tetrahedron_2']
        intersection_status = item['intersection_status']

        data1 = self._json_to_graph(tetra1)
        data2 = self._json_to_graph(tetra2)
        label = torch.tensor(intersection_status, dtype=torch.float)

        combined_data = TetrahedronPairGraph(x_1=data1.x, edge_index_1=data1.edge_index,
                                             x_2=data2.x, edge_index_2=data2.edge_index,
                                             y=label)
        return combined_data

    def _json_to_graph(self, tetrahedron):
        x = torch.tensor(tetrahedron['vertices'], dtype=torch.float)
        edge_index = torch.tensor(tetrahedron['edges'], dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def __len__(self):
        return self.data_length
    
class TabularProcessor(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        BaseProcessor.__init__(self, root_dir)
        Dataset.__init__(self, root_dir, transform, pre_transform)
        self.data_length = 0

    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.data_length = len(data)
        return data

    def __len__(self):
        return self.data_length
    
    def _process_item(self, item):
        tetra1 = item['tetrahedron_1']
        tetra2 = item['tetrahedron_2']
        intersection_status = item['intersection_status']

        data1 = self._json_to_tabular(tetra1)
        data2 = self._json_to_tabular(tetra2)
        label = torch.tensor([intersection_status], dtype=torch.float)

        combined_data = torch.cat((data1, data2, label), dim=0)
        return combined_data
    
    def _json_to_tabular(self, tetrahedron):
        x = torch.tensor(tetrahedron['vertices'], dtype=torch.float)
        return x.view(-1)  # Flatten the tensor

####################
## Helper Classes ##
####################
    
class TetrahedronPairGraph(Data):
    def __init__(self, x_1=None, edge_index_1=None, x_2=None, edge_index_2=None, y=None):
        super(TetrahedronPairGraph, self).__init__()
        self.x_1 = x_1
        self.edge_index_1 = edge_index_1
        self.x_2 = x_2
        self.edge_index_2 = edge_index_2
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        elif key == 'edge_index_2':
            return self.x_2.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['edge_index_1', 'edge_index_2']:
            return 1
        elif key in ['x_1', 'x_2']:
            return 0
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)