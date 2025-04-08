import os
import pytest
from unittest.mock import patch, MagicMock, Mock
from src.CPipelineOrchestrator import CPipelineOrchestrator

# filepath: /home/sei/tetrahedron_pair_ML/src/test_CPipelineOrchestrator.py

# Import the class to test

class TestCPipelineOrchestrator:
    @pytest.fixture
    def mock_dependencies(self):
        """Set up mocks for all dependencies"""
        with patch('src.CPipelineOrchestrator.CDataProcessor') as mock_processor, \
             patch('src.CPipelineOrchestrator.CModelBuilder') as mock_builder, \
             patch('src.CPipelineOrchestrator.CModelTrainer') as mock_trainer, \
             patch('src.CPipelineOrchestrator.CArtifactsManager') as mock_artifacts, \
             patch('src.CPipelineOrchestrator.CEvaluator') as mock_evaluator, \
             patch('src.CPipelineOrchestrator.yaml.safe_load') as mock_yaml_load:
            
            # Configure yaml mock to return a valid config
            mock_yaml_load.return_value = {
                'home': '/home/test',
                'processor_config': {'skip_processing': True},
                'model_config': {'skip_building': False},
                'trainer_config': {'skip_training': False},
                'evaluator_config': {'skip_evaluation': False},
                'artifacts_config': {}
            }
            
            # Configure the builder mock to return a model
            mock_model = MagicMock()
            mock_builder.return_value.build.return_value = mock_model
            
            # Configure the trainer mock
            mock_trainer.return_value.train.return_value = (mock_model, {'loss': 0.1})
            
            # Yield all mocks for use in tests
            yield {
                'processor': mock_processor,
                'builder': mock_builder,
                'trainer': mock_trainer,
                'artifacts': mock_artifacts,
                'evaluator': mock_evaluator,
                'yaml_load': mock_yaml_load,
                'model': mock_model
            }
    
    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Create an orchestrator instance with mocked dependencies"""
        with patch('builtins.open', MagicMock()):
            return CPipelineOrchestrator('mock_config.yaml')
    
    def test_full_pipeline_execution(self, orchestrator, mock_dependencies):
        """Test successful execution of the full pipeline"""
        # Execute the pipeline
        orchestrator.run()
        
        # Verify the correct methods were called
        mock_dependencies['builder'].return_value.build.assert_called_once()
        mock_dependencies['trainer'].return_value.train.assert_called_once()
        mock_dependencies['evaluator'].return_value.evaluate.assert_called_once()
    
    def test_skip_model_building(self, mock_dependencies):
        """Test pipeline execution with model building skipped"""
        # Modify config to skip building
        mock_dependencies['yaml_load'].return_value['model_config']['skip_building'] = True
        
        # Create orchestrator with modified config
        with patch('builtins.open', MagicMock()):
            orchestrator = CPipelineOrchestrator('mock_config.yaml')
        
        # Mock _build_model_step to inject a model into the state
        with patch.object(orchestrator, '_build_model_step') as mock_build_step:
            def side_effect(state):
                state['model'] = mock_dependencies['model']
            mock_build_step.side_effect = side_effect
            
            # Run the pipeline
            orchestrator.run()
        
        # Verify build was not called directly (we patched it)
        mock_dependencies['builder'].return_value.build.assert_not_called()
        
        # But training and evaluation should still run
        mock_dependencies['trainer'].return_value.train.assert_called_once()
        mock_dependencies['evaluator'].return_value.evaluate.assert_called_once()
    
    def test_skip_evaluation(self, mock_dependencies):
        """Test pipeline execution with evaluation skipped"""
        # Modify config to skip evaluation
        mock_dependencies['yaml_load'].return_value['evaluator_config']['skip_evaluation'] = True
        
        # Create orchestrator with modified config
        with patch('builtins.open', MagicMock()):
            orchestrator = CPipelineOrchestrator('mock_config.yaml')
        
        # Run the pipeline
        orchestrator.run()
        
        # Verify build and train were called
        mock_dependencies['builder'].return_value.build.assert_called_once()
        mock_dependencies['trainer'].return_value.train.assert_called_once()
        
        # But evaluation should be skipped
        mock_dependencies['evaluator'].return_value.evaluate.assert_not_called()
    
    def test_missing_model_error(self, mock_dependencies):
        """Test error handling when model is missing for training"""
        # Make the builder return None instead of a model
        mock_dependencies['builder'].return_value.build.return_value = None
        
        # Create orchestrator
        with patch('builtins.open', MagicMock()):
            orchestrator = CPipelineOrchestrator('mock_config.yaml')
        
        # Run should raise ValueError due to missing model
        with pytest.raises(ValueError, match="Model is required for training but was not provided"):
            orchestrator.run()
    
    def test_load_model_path(self, mock_dependencies):
        """Test loading a model from path instead of training"""
        # Configure to skip training and provide model path
        config = mock_dependencies['yaml_load'].return_value
        config['trainer_config']['skip_training'] = True
        config['evaluator_config']['model_path'] = '/models/test_model.pt'
        
        # Mock torch.jit.load
        with patch('builtins.open', MagicMock()), \
             patch('torch.jit.load', return_value=mock_dependencies['model']):
            
            orchestrator = CPipelineOrchestrator('mock_config.yaml')
            
            # Run the pipeline
            orchestrator.run()
        
        # Training should be skipped, but evaluation should run
        mock_dependencies['trainer'].return_value.train.assert_not_called()
        mock_dependencies['evaluator'].return_value.evaluate.assert_called_once()