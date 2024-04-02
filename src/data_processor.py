import os
import json
import torch
from torch_geometric.data import Data, Dataset

class DataProcessor(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DataProcessor, self).__init__(root, transform, pre_transform)
        self.last_processed_index_train = None
        self.last_processed_index_val = None

    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.raw_dir))

    def process_entries(self, subset, batch_size):

        last_index = self.last_processed_index_train if subset == 'train' else self.last_processed_index_val
        data_list = []
        all_files = os.listdir(os.path.join(self.raw_dir, subset))
        file_path = os.path.join(self.raw_dir, subset, all_files[0]) # ToDo: support for more than one file

        with open(file_path, 'r') as f:
            json_data = json.load(f)

        start_index = 0 if last_index is None else last_index + 1
        end_index = min(start_index + batch_size, len(json_data))

        for i in range(start_index, end_index):
            item = json_data[i]
            processed_item = self._process_item(item)
            data_list.append(processed_item)

        if data_list:
            if subset == 'train':
                self.last_processed_index_train = end_index - 1
            else:
                self.last_processed_index_val = end_index - 1


        return data_list

    def _process_item(self, item):    

        tetra1 = item['tetra1']
        tetra2 = item['tetra2']
        intersection_status = item['intersection_status']

        data1 = self._tetrahedron_to_graph(tetra1)
        data2 = self._tetrahedron_to_graph(tetra2)
        label = torch.tensor(intersection_status, dtype=torch.float)

        combined_data = TetrahedronPairData(x_1=data1.x, edge_index_1=data1.edge_index,
                                            x_2=data2.x, edge_index_2=data2.edge_index,
                                            y=label)
        return combined_data

    def _tetrahedron_to_graph(self, tetrahedron):
        x = torch.tensor(tetrahedron['vertices'], dtype=torch.float)
        edge_index = torch.tensor(tetrahedron['edges'], dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

class TetrahedronPairData(Data):
    def __init__(self, x_1=None, edge_index_1=None, x_2=None, edge_index_2=None, y=None):
        super(TetrahedronPairData, self).__init__()
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