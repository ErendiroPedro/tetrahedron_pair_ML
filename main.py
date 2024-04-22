import os
import argparse
import logging
from src.model_trainer import TabularTrainer, GraphTrainer
from src.data_processor import TabularProcessor, GraphProcessor
from src.model_architecture import TabularClassifier, GraphPairClassifier
from src.model_evaluator import ModelEvaluator
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='ML Workflow Manager')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    return parser.parse_args()

class MLWorkflowManager:
    def __init__(self, dataset_root, model_path):
        self.dataset_dir = dataset_root
        self.model_path = model_path
        self.batch_size = 750
        self.epochs = 200
        self.model = GraphPairClassifier()
        self.trainer = GraphTrainer(self.model, self.optimizer, self.data_processor, batch_size=self.batch_size, epochs=self.epochs)
        self.evaluator = ModelEvaluator(raw_data_root=self.dataset_dir, model_path=self.model_path, batch_size=self.batch_size, epochs = self.epochs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.data_processor = GraphProcessor(dataset_root)
        
        
    def run(self):
        print("Running ML Workflow Manager")
        # print(torch.cuda.is_available())
        torch.manual_seed(13) # Deterministic

        trainer = self.trainer
        trainer.train_and_validate()

        evaluator = self.evaluator
        evaluator.evaluate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    
    ml_workflow_manager = MLWorkflowManager(args.dataset_root, args.model_path)
    ml_workflow_manager.run()