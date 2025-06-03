import torch
import pandas as pd
import numpy as np
import time
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import csv

class Sample:
    def __init__(self, input_tensor: torch.Tensor, volume_tensor: torch.Tensor, label_tensor: torch.Tensor,):
        self.input = input_tensor    # [24]
        self.volume = volume_tensor  # [1]
        self.label = label_tensor    # [1]

def load_csv(filename: str) -> List[Sample]:
    """Load CSV data and convert to Sample objects - replicating C++ behavior exactly"""
    data = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip header
        next(csv_reader)
        
        for row in csv_reader:
            values = [float(cell) for cell in row]
            
            assert len(values) == 26, f"Expected 26 columns, got {len(values)}"
            
            # Split into input and targets (same as C++)
            input_values = values[:24]
            label = values[24]
            volume = values[25]
            
            # Create tensors - remove unsqueeze since we'll batch properly
            input_tensor = torch.tensor(input_values, dtype=torch.float32)  # [24]
            label_tensor = torch.tensor([label], dtype=torch.float32)
            volume_tensor = torch.tensor([volume], dtype=torch.float32)
            
            data.append(Sample(input_tensor, label_tensor, volume_tensor))
    
    return data

class FileUtils:
    @staticmethod
    def file_exists_and_readable(path: str) -> bool:
        try:
            with open(path, 'r') as f:
                pass
            return True
        except:
            return False
    
    @staticmethod
    def validate_file(path: str, description: str):
        if not FileUtils.file_exists_and_readable(path):
            raise RuntimeError(f"Error: {description} does not exist or is not readable: {path}")

class ModelManager:
    @staticmethod
    def load_model(model_path: str) -> torch.jit.ScriptModule:
        print(f"Loading model from: {model_path}")
        
        FileUtils.validate_file(model_path, "Model file")
        
        # Load TorchScript model (equivalent to torch::jit::load)
        scripted_net = torch.jit.load(model_path, map_location='cpu')
        print("Model loaded successfully!")
        
        return scripted_net

class DataManager:
    @staticmethod
    def load_dataset(csv_path: str) -> List[Sample]:
        print(f"Loading dataset from: {csv_path}")
        
        FileUtils.validate_file(csv_path, "CSV file")
        
        dataset = load_csv(csv_path)
        
        if not dataset:
            raise RuntimeError("Dataset is empty!")
        
        print(f"Dataset loaded successfully! Samples: {len(dataset)}")
        return dataset
    
    @staticmethod
    def apply_transformations(dataset: List[Sample]):
        """Placeholder for data transformations"""
        print("Applying data transformations... (placeholder)")
        # TODO: Implement data transformations here
        # - Normalization
        # - Augmentation
        # - Preprocessing

class Evaluator:
    class EvaluationResults:
        def __init__(self, accuracy: float, avg_inference_time_ms: float, 
                     samples_per_second: float, correct_predictions: int, total_samples: int):
            self.accuracy = accuracy
            self.avg_inference_time_ms = avg_inference_time_ms
            self.samples_per_second = samples_per_second
            self.correct_predictions = correct_predictions
            self.total_samples = total_samples
    
    @staticmethod
    def extract_predictions(output: torch.Tensor, task: str) -> List[float]:
        """Extract predictions from model output - replicating C++ logic exactly"""
        # Convert to double if needed (equivalent to C++ conversion)
        if output.dtype != torch.float64:
            output = output.to(torch.float64)
        
        # Access elements like C++ accessor
        if task == "IntersectionStatus":
            return [1.0 if output[0, 0].item() > 0.5 else 0.0]
        elif task == "IntersectionVolume":
            return [output[0, 0].item() / 1000000.0]  # Use correct scale factor
        elif task == "IntersectionStatus_IntersectionVolume":
            cls_pred = 1.0 if output[0, 0].item() > 0.5 else 0.0
            reg_pred = output[0, 1].item() / 1000000.0  # Use correct scale factor
            return [cls_pred, reg_pred]
        else:
            raise RuntimeError(f"Unknown task: {task}")
    
    @staticmethod
    def evaluate_model(model: torch.jit.ScriptModule, dataset: List[Sample], 
                      batch_size: int = 32) -> 'Evaluator.EvaluationResults':
        print(f"Starting evaluation with batch_size={batch_size}...")
        
        correct = 0
        total = 0
        total_inference_time_ms = 0.0
        
        total_start = time.time()
        
        # Set model to eval mode
        model.eval()
        
        with torch.no_grad():  # Equivalent to C++ inference mode
            # Process in batches
            for batch_idx in range(0, len(dataset), batch_size):
                batch_samples = dataset[batch_idx:batch_idx + batch_size]
                
                # Stack inputs into a proper batch
                batch_inputs = torch.stack([sample.input for sample in batch_samples])  # [batch_size, 24]
                batch_labels = torch.stack([sample.label for sample in batch_samples])   # [batch_size, 1]
                batch_volumes = torch.stack([sample.volume for sample in batch_samples]) # [batch_size, 1]
                
                batch_inputs = batch_inputs.to('cpu')  # Ensure CPU
                
                # Inference timing
                start = time.time()
                batch_output = model(batch_inputs)  # Proper batch processing
                end = time.time()
                
                inference_time_ms = (end - start) * 1000.0
                total_inference_time_ms += inference_time_ms
                
                # Process each sample in the batch
                for i, sample in enumerate(batch_samples):
                    # Extract single sample output [1, output_dim]
                    single_output = batch_output[i:i+1]
                    
                    # Extract predictions
                    preds = Evaluator.extract_predictions(single_output, "IntersectionStatus_IntersectionVolume")
                    
                    # Classification check
                    predicted_cls = int(preds[0])
                    true_cls = int(batch_labels[i].item())
                    
                    if predicted_cls == true_cls:
                        correct += 1
                    
                    total += 1
                    
                    # Debug: Print first few predictions (same as C++)
                    if total <= 5:
                        print(f"Sample {total}:")
                        print(f"  Batch input shape: {batch_inputs.shape}")
                        print(f"  Single output: {single_output}")
                        print(f"  Predictions: [{preds[0]}, {preds[1]}]")
                        print(f"  True label: {true_cls}")
                
        total_end = time.time()
        total_duration_sec = total_end - total_start
        
        return Evaluator.EvaluationResults(
            accuracy=(correct / total) * 100.0,
            avg_inference_time_ms=total_inference_time_ms / total,
            samples_per_second=total / total_duration_sec,
            correct_predictions=correct,
            total_samples=total
        )
    
    @staticmethod
    def print_results(results: 'Evaluator.EvaluationResults'):
        print("\n=== EVALUATION RESULTS ===")
        print(f"Accuracy: {results.accuracy}%")
        print(f"Correct predictions: {results.correct_predictions}/{results.total_samples}")
        print(f"Average inference time per sample: {results.avg_inference_time_ms} ms")
        print(f"Samples per second: {results.samples_per_second}")

class ArgumentParser:
    class Args:
        def __init__(self, model_path: str, csv_path: str, batch_size: int = 32):
            self.model_path = model_path
            self.csv_path = csv_path
            self.batch_size = batch_size
    
    @staticmethod
    def parse_arguments() -> 'ArgumentParser.Args':
        parser = argparse.ArgumentParser(description='PyTorch Model Evaluator')
        parser.add_argument('model_path', help='Path to the TorchScript model file')
        parser.add_argument('csv_path', help='Path to the CSV dataset file')
        parser.add_argument('--batch_size', type=int, default=32, 
                          help='Batch size for evaluation (default: 32)')
        
        args = parser.parse_args()
        return ArgumentParser.Args(args.model_path, args.csv_path, args.batch_size)

def main():
    try:
        # Parse arguments
        args = ArgumentParser.parse_arguments()
        
        # Load model
        model = ModelManager.load_model(args.model_path)
        
        # Load and transform data
        dataset = DataManager.load_dataset(args.csv_path)
        DataManager.apply_transformations(dataset)
        
        # Evaluate model with configurable batch size
        results = Evaluator.evaluate_model(model, dataset, batch_size=args.batch_size)
        
        # Print results
        Evaluator.print_results(results)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    exit(main())