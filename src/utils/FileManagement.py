from typing import Dict
import os

class ConfigResolver:
    """Handles recursive path resolution based on a 'home' directory."""
    @staticmethod
    def resolve(config: Dict) -> Dict:
        home = config.get('home')
        if not home:
            raise ValueError("The 'home' path is missing from the configuration.")

        def _walk(node):
            if isinstance(node, dict):
                return {k: _walk(v) for k, v in node.items()}
            if isinstance(node, str) and (node.startswith('/') or './' in node or any(ext in node for ext in ['.csv', '.pt', '.pth', '.ipynb'])):
                return node if os.path.isabs(node) else os.path.join(home, node.lstrip('/'))
            return node
        
        return _walk(config)


class FileSystemScanner:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
    
    def discover_files(self, intersection_type):
        intersection_folder = os.path.join(self.raw_data_path, intersection_type)
        
        if not os.path.exists(intersection_folder):
            raise ValueError(f"  Warning: Folder not found: {intersection_folder}")
        
        return [os.path.join(intersection_folder, f) 
                for f in os.listdir(intersection_folder) 
                if f.endswith(".csv")]


class DataInventory:
    def __init__(self, file_processor):
        self.file_processor = file_processor
    
    def count_raw_data(self, raw_data_path):
        total = 0
        print("Counting raw data files:")
        
        for intersection_type in os.listdir(raw_data_path):
            folder_path = os.path.join(raw_data_path, intersection_type)
            if os.path.isdir(folder_path):
                folder_total = 0
                for csv_file in os.listdir(folder_path):
                    if csv_file.endswith('.csv'):
                        file_path = os.path.join(folder_path, csv_file)
                        try:
                            count = self.file_processor.count_samples_in_file(file_path)
                            folder_total += count
                        except Exception as e:
                            print(f"  Error reading {csv_file}: {e}")
                total += folder_total
                print(f"  {intersection_type} total: {folder_total:,} rows")
        
        print(f"Total raw data: {total:,} rows")
        return total
