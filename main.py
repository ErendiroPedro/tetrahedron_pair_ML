import os
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

DATASET_ROOT = '/home/sei/tetrahedron_pair_ML/data'

class MLWorkflowManager:
    def __init__(self, dataset_root):
        self.dataset_dir = dataset_root
        self.model_path = '/home/sei/tetrahedron_pair_ML/model_state_dict.pt'

    def run(self):

        model_trainer = ModelTrainer(self.dataset_dir)
        model_trainer.train_and_validate()

        # evaluator = ModelEvaluator(self.dataset_dir, self.model_path)
        # evaluator.evaluate()

if __name__ == "__main__":
    
    ml_workflow_manager = MLWorkflowManager(DATASET_ROOT)
    ml_workflow_manager.run()
