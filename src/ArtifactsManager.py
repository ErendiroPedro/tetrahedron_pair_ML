import os
import base64
import torch
import pickle
from io import BytesIO
from PIL import Image
import random
import string
import datetime


class ArtifactsManager:
    def __init__(self, config):
        self.config = config
        self.artifacts_path = os.path.join(config['artifacts_config']['save_artifacts_to'], self._generate_folder_name())
        os.makedirs(self.artifacts_path, exist_ok=True)

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
                self._save_model(artifact, f"model_{index}")
            elif isinstance(artifact, str):
                if self._is_base64_image(artifact):
                    self._save_base64_image(artifact, f"image_{index}")
                else:
                    self._save_text(artifact, f"text_{index}")
            else:
                raise ValueError(f"Unsupported artifact type for argument at position {index}")

    def _get_artifacts_path(self):
        """
        Get the path where artifacts are saved.

        Returns:
            str: The path where artifacts are saved.
        """
        return self.artifacts_path

    def _generate_folder_name(self):
        """
        Generate a descriptive folder name based on the configuration.

        Args:
            config (dict): The experiment configuration dictionary.

        Returns:
            str: A descriptive folder name.
        """
        id = datetime.datetime.now().strftime("%S%M%H-%Y%m%d")
        model_architecture = self.config['model_config']['architecture']
        task = self.config['model_config']['task']
        num_train_samples = self.config['processor_config']['num_train_samples']
        num_val_samples = self.config['processor_config']['num_val_samples']
        volume_range = "-".join(map(str, self.config['processor_config']['volume_range']))
        epochs = self.config['trainer_config']['epochs']
        optimizer = self.config['trainer_config']['optimizer']
        learning_rate = self.config['trainer_config']['learning_rate']
        folder_name = (
            f"id-{id}_{model_architecture}_{task}_"
            f"{num_train_samples}-train_{num_val_samples}-val_"
            f"vol-{volume_range}_epochs-{epochs}_{optimizer}-{learning_rate}"
        )
        
        return folder_name

    def _save_model(self, model, filename):
        """
        Save a PyTorch model including its class definition, weights, and architecture.

        Args:
            model (torch.nn.Module): PyTorch model to save.
            filename (str): Base name of the file (without extension).
        """
        # Save the full model object (class and weights)
        full_model_path = os.path.join(self.artifacts_path, f"model.pkl")
        with open(full_model_path, "wb") as f:
            pickle.dump(model, f)

    def _save_text(self, content, filename):
        """
        Save a string as a text file.

        Args:
            content (str): The text content to save.
            filename (str): Name of the file to save the text as.
        """
        filepath = os.path.join(self.artifacts_path, f"{filename}.txt")
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

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
        filepath = os.path.join(self.artifacts_path, f"loss_report.png")
        image.save(filepath, format="PNG")