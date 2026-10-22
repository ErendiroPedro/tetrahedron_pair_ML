import torch

class ModelUtility:
    """Handles TorchScript wrapping and complex weight mapping."""
    
    class TorchScriptWrapper(torch.nn.Module):
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.task = getattr(model, 'task', config['evaluator_config'].get('task', 'IntersectionStatus_IntersectionVolume'))
            self.volume_scale_factor = getattr(model, 'volume_scale_factor', config['evaluator_config'].get('volume_scale_factor', 1.0))

        def predict(self, x):
            self.model.eval()
            with torch.no_grad():
                out = self.model.forward(x) 
                if self.task == 'IntersectionStatus':
                    return (out > 0.5).int().squeeze()
                elif self.task == 'IntersectionVolume':
                    return out.squeeze() / self.volume_scale_factor
                elif self.task == 'IntersectionStatus_IntersectionVolume':
                    return torch.stack([(out[:, 0] > 0.5).int(), out[:, 1] / self.volume_scale_factor], dim=1)
                raise ValueError(f"Unknown task: {self.task}")

        def forward(self, x): return self.model(x)

    @staticmethod
    def map_weights(torchscript_model, target_model):
        """Logic to map parameters from TorchScript back to PyTorch."""
        ts_state = torchscript_model.state_dict()
        target_state = target_model.state_dict()
        matched = 0
        
        for name, param in target_state.items():
            # Try exact match or fuzzy match for TorchScript name mangling
            for ts_name, ts_param in ts_state.items():
                clean_ts_name = ts_name.replace('original_name.', '').replace('_c.', '.')
                if name == clean_ts_name and param.shape == ts_param.shape:
                    target_state[name].copy_(ts_param)
                    matched += 1
                    break
        print(f"Successfully mapped {matched}/{len(target_state)} parameters.")
        return target_model
