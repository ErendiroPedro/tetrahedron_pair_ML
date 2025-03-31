import os
import base64
import torch
from io import BytesIO
from PIL import Image
import datetime
import json
import yaml


class CArtifactsManager:
    def __init__(self, config):
        self.config = config
        self.artifacts_path = self._create_experiment_folder()

    def save_artifacts(self, *args):
        """
        Save various types of artifacts including PyTorch models, images, and text files.

        Args:
            *args: Unnamed artifacts to save.
                - torch.nn.Module: Saved with weights and architecture.
                - str: Saved as a text file or decoded image if Base64-encoded.
                - dict with "image" and "type" keys: Encoded in Base64 and saved as a text file.
        """
        for index, artifact in enumerate(args):
            if isinstance(artifact, torch.nn.Module):
                self._save_model(artifact)
            elif isinstance(artifact, str) and self._is_base64_image(artifact):
                self._save_base64_image(artifact, f"loss_report")
            elif isinstance(artifact, dict):
                self._save_evaluation_report(artifact)
            else:
                raise ValueError(f"Unsupported artifact type for argument at position {index}")

    def _create_experiment_folder(self):
        """
        Generate a descriptive folder name based on the configuration and save the config file in the folder.

        Returns:
            str: Path to the created folder.
        """
        # Generate a timestamp in reverse chronological order for sorting
        id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Extract relevant configuration details
        model_architecture = self.config['model_config']['architecture']['use']
        task = self.config['model_config']['task']
        num_train_samples = self.config['processor_config']['num_train_samples']
        num_val_samples = self.config['processor_config']['num_val_samples']
        volume_range = "-".join(map(str, self.config['processor_config']['volume_range']))
        epochs = self.config['trainer_config']['epochs']
        optimizer = self.config['trainer_config']['optimizer']
        learning_rate = self.config['trainer_config']['learning_rate']

        # Add distribution details
        distributions = self.config['processor_config']['intersection_distributions']
        dist_str = "_".join(f"{key}-{value}" for key, value in distributions.items() if value > 0)

        # Construct the folder name
        folder_name = (
            f"id-{id}_{model_architecture}_{task}_"
            f"{num_train_samples}-train_{num_val_samples}-val_"
            f"vol-{volume_range}_epochs-{epochs}_{optimizer}-{learning_rate}_"
            f"distributions-{dist_str}"
        )
        
        # Create the folder
        artifacts_path = self.config['artifacts_config']['save_artifacts_to']
        full_folder_path = os.path.join(artifacts_path, folder_name)
        os.makedirs(full_folder_path, exist_ok=True)

        # Save config
        config_file_path = os.path.join(full_folder_path, "config.yaml")
        with open(config_file_path, 'w') as config_file:
            yaml.dump(self.config, config_file, default_flow_style=False)

        return full_folder_path

    def _save_model(self, model):
        """
        Save a PyTorch model in TorchScript format for use in C++.

        Args:
            model (torch.nn.Module): PyTorch model to save.
        """
        script_model = torch.jit.script(model)
        script_model_path = os.path.join(self.artifacts_path, "model.pt")
        script_model.save(script_model_path)
        print(f"Model saved at {script_model_path}")

    def _save_evaluation_report(self, report):
        """
        Save the evaluation report as a JSON file.

        Args:
            report (dict): The evaluation report dictionary to save.
        """
        filepath = os.path.join(self.artifacts_path, f"evaluation_report.json")
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

    def _is_base64_image(self, base64_str):
        """
        Check if a string is a Base64-encoded image.

        Args:
            base64_str (str): The string to check.
        
        Returns:
            bool: True if the string is a Base64-encoded image, otherwise False.
        """
        try:
            base64.b64decode(base64_str)  # Check if it decodes correctly
            return True
        except Exception:
            return False

    def _save_base64_image(self, base64_str, filename):
        """
        Decode a Base64-encoded image and save it as a file.

        Args:
            base64_str (str): The Base64 string of the image.
            filename (str): Name of the file to save the image as.
        """
        decoded_image = base64.b64decode(base64_str)
        image_buffer = BytesIO(decoded_image)
        image = Image.open(image_buffer)
        filepath = os.path.join(self.artifacts_path, f"{filename}.png")
        image.save(filepath, format="PNG")