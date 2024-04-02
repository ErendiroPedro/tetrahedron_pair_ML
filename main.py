import os
import argparse
import logging
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='ML Workflow Manager')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    return parser.parse_args()

class MLWorkflowManager:
    def __init__(self, dataset_root, model_path):
        self.dataset_dir = dataset_root
        self.model_path = model_path

    def run(self):
        try:
            model_trainer = ModelTrainer(self.dataset_dir)
            model_trainer.train_and_validate()

            # evaluator = ModelEvaluator(self.dataset_dir, self.model_path)
            # evaluator.evaluate()
        except Exception as e:
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    
    ml_workflow_manager = MLWorkflowManager(args.dataset_root, args.model_path)
    ml_workflow_manager.run()