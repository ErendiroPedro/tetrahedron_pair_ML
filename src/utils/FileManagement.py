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