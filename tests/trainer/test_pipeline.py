"""
Test suite for Newsies Trainer Pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import json
from datetime import datetime

from newsies_trainer.pipelines.train_model import train_model_pipeline


class TestTrainerPipeline:
    """Test cases for model training pipeline"""

    @pytest.fixture
    def mock_redis_task_status(self):
        """Mock Redis task status for testing"""
        with patch('newsies_trainer.pipelines.train_model.redis_task_status') as mock:
            yield mock

    @pytest.fixture
    def mock_model_loader(self):
        """Mock model loading functionality"""
        with patch('newsies_trainer.llm.load_latest.load_model') as mock:
            yield mock

    @pytest.fixture
    def mock_data_loader(self):
        """Mock training data loading"""
        with patch('newsies_trainer.pipelines.train_model.load_training_data') as mock:
            yield mock

    @pytest.fixture
    def mock_lora_adapter(self):
        """Mock LoRA adapter functionality"""
        with patch('newsies_trainer.llm.lora.create_lora_adapter') as mock:
            yield mock

    @pytest.fixture
    def mock_trainer(self):
        """Mock model trainer"""
        with patch('newsies_trainer.pipelines.train_model.Trainer') as mock:
            yield mock

    def test_train_model_pipeline_success(self, mock_redis_task_status, 
                                        mock_model_loader, mock_data_loader,
                                        mock_lora_adapter, mock_trainer):
        """Test successful execution of training pipeline"""
        task_id = "test_trainer_task_123"
        
        # Mock successful execution of all steps
        mock_model_loader.return_value = Mock()  # Mock model
        mock_data_loader.return_value = {"train": [], "eval": []}  # Mock datasets
        mock_lora_adapter.return_value = Mock()  # Mock LoRA adapter
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {"train_loss": 0.5}
        
        # Execute pipeline
        train_model_pipeline(task_id)
        
        # Verify all steps were called
        mock_model_loader.assert_called_once()
        mock_data_loader.assert_called_once()
        mock_lora_adapter.assert_called_once()
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        
        # Verify task status updates
        assert mock_redis_task_status.set_status.call_count >= 5

    def test_train_model_pipeline_gpu_unavailable(self, mock_redis_task_status):
        """Test pipeline behavior when GPU is unavailable"""
        task_id = "test_trainer_no_gpu"
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('newsies_trainer.pipelines.train_model.load_model') as mock_loader:
                mock_loader.return_value = Mock()
                
                # Should continue with CPU training
                train_model_pipeline(task_id)
                
                # Verify warning was logged about CPU usage
                status_calls = mock_redis_task_status.set_status.call_args_list
                status_messages = [str(call) for call in status_calls]
                assert any("cpu" in msg.lower() for msg in status_messages)

    def test_train_model_pipeline_data_loading_failure(self, mock_redis_task_status, mock_data_loader):
        """Test pipeline failure at data loading step"""
        task_id = "test_trainer_data_fail"
        
        # Mock failure at data loading step
        mock_data_loader.side_effect = Exception("Failed to load training data")
        
        with pytest.raises(Exception) as exc_info:
            train_model_pipeline(task_id)
        
        assert "Failed to load training data" in str(exc_info.value)
        
        # Verify error status was set
        error_calls = [call for call in mock_redis_task_status.set_status.call_args_list 
                      if "error" in str(call)]
        assert len(error_calls) > 0

    def test_pipeline_status_updates(self, mock_redis_task_status, mock_model_loader,
                                   mock_data_loader, mock_lora_adapter, mock_trainer):
        """Test that pipeline updates task status at each step"""
        task_id = "test_status_updates"
        
        # Mock successful execution
        mock_model_loader.return_value = Mock()
        mock_data_loader.return_value = {"train": [], "eval": []}
        mock_lora_adapter.return_value = Mock()
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {"train_loss": 0.5}
        
        train_model_pipeline(task_id)
        
        # Verify status updates for each step
        status_calls = mock_redis_task_status.set_status.call_args_list
        status_messages = [call[0][1] for call in status_calls]
        
        expected_statuses = [
            "started",
            "running - step: loading base model",
            "running - step: preparing training data",
            "running - step: creating LoRA adapter",
            "running - step: training model",
            "running - step: saving model",
            "complete"
        ]
        
        for expected_status in expected_statuses:
            assert any(expected_status in status for status in status_messages)


class TestLoRAAdapter:
    """Test LoRA adapter functionality"""

    def test_create_lora_adapter(self):
        """Test LoRA adapter creation"""
        from newsies_trainer.llm.lora import create_lora_adapter
        
        with patch('peft.LoraConfig') as mock_config:
            with patch('peft.get_peft_model') as mock_get_peft:
                mock_model = Mock()
                mock_config.return_value = Mock()
                mock_get_peft.return_value = Mock()
                
                adapter = create_lora_adapter(mock_model, rank=16, alpha=32)
                
                mock_config.assert_called_once()
                mock_get_peft.assert_called_once()
                assert adapter is not None

    def test_lora_config_parameters(self):
        """Test LoRA configuration parameters"""
        from newsies_trainer.llm.lora import create_lora_config
        
        with patch('peft.LoraConfig') as mock_config:
            config = create_lora_config(
                rank=8,
                alpha=16,
                target_modules=["q_proj", "v_proj"],
                dropout=0.1
            )
            
            mock_config.assert_called_once_with(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )

    def test_merge_and_save_adapter(self):
        """Test merging and saving LoRA adapter"""
        from newsies_trainer.llm.lora import merge_and_save_adapter
        
        mock_model = Mock()
        save_path = "/tmp/test_model"
        
        with patch('torch.save') as mock_save:
            merge_and_save_adapter(mock_model, save_path)
            
            mock_model.merge_and_unload.assert_called_once()
            mock_save.assert_called()


class TestModelTraining:
    """Test model training functionality"""

    def test_training_arguments_setup(self):
        """Test training arguments configuration"""
        from newsies_trainer.training.config import get_training_arguments
        
        args = get_training_arguments(
            output_dir="/tmp/output",
            num_epochs=3,
            batch_size=4,
            learning_rate=2e-4
        )
        
        assert args.output_dir == "/tmp/output"
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 4
        assert args.learning_rate == 2e-4

    def test_data_collator_setup(self):
        """Test data collator configuration"""
        from newsies_trainer.training.data import get_data_collator
        
        mock_tokenizer = Mock()
        collator = get_data_collator(mock_tokenizer)
        
        assert collator is not None

    def test_training_loop_monitoring(self):
        """Test training loop with monitoring"""
        from newsies_trainer.training.trainer import CustomTrainer
        
        with patch('transformers.Trainer') as mock_trainer_class:
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            mock_trainer.train.return_value = {"train_loss": 0.5, "eval_loss": 0.6}
            
            trainer = CustomTrainer(
                model=Mock(),
                args=Mock(),
                train_dataset=Mock(),
                eval_dataset=Mock()
            )
            
            results = trainer.train()
            
            assert "train_loss" in results
            assert "eval_loss" in results

    @pytest.mark.gpu
    def test_gpu_memory_management(self):
        """Test GPU memory management during training"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        from newsies_trainer.utils.gpu import manage_gpu_memory
        
        # Test memory cleanup
        initial_memory = torch.cuda.memory_allocated()
        manage_gpu_memory()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should be cleaned up
        assert final_memory <= initial_memory


class TestDataPreparation:
    """Test training data preparation"""

    def test_tokenize_dataset(self):
        """Test dataset tokenization"""
        from newsies_trainer.data.preprocessing import tokenize_dataset
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1]
        }
        
        dataset = [
            {"text": "This is a sample text for training."},
            {"text": "Another sample text for the model."}
        ]
        
        tokenized = tokenize_dataset(dataset, mock_tokenizer, max_length=512)
        
        assert len(tokenized) == 2
        assert all("input_ids" in item for item in tokenized)
        assert all("attention_mask" in item for item in tokenized)

    def test_prepare_training_data(self):
        """Test training data preparation"""
        from newsies_trainer.data.preparation import prepare_training_data
        
        raw_articles = [
            {"title": "Article 1", "content": "Content 1", "summary": "Summary 1"},
            {"title": "Article 2", "content": "Content 2", "summary": "Summary 2"}
        ]
        
        with patch('newsies_trainer.data.preparation.format_for_training') as mock_format:
            mock_format.return_value = [
                {"input": "Content 1", "output": "Summary 1"},
                {"input": "Content 2", "output": "Summary 2"}
            ]
            
            training_data = prepare_training_data(raw_articles)
            
            assert len(training_data) == 2
            assert all("input" in item for item in training_data)
            assert all("output" in item for item in training_data)

    def test_data_validation(self):
        """Test training data validation"""
        from newsies_trainer.data.validation import validate_training_data
        
        # Valid data
        valid_data = [
            {"input": "Input text", "output": "Output text"},
            {"input": "Another input", "output": "Another output"}
        ]
        
        assert validate_training_data(valid_data) is True
        
        # Invalid data
        invalid_data = [
            {"input": "", "output": "Output text"},  # Empty input
            {"input": "Input text"}  # Missing output
        ]
        
        assert validate_training_data(invalid_data) is False


class TestModelEvaluation:
    """Test model evaluation functionality"""

    def test_evaluate_model_performance(self):
        """Test model performance evaluation"""
        from newsies_trainer.evaluation.metrics import evaluate_model
        
        mock_model = Mock()
        mock_eval_dataset = Mock()
        
        with patch('newsies_trainer.evaluation.metrics.compute_perplexity') as mock_perplexity:
            with patch('newsies_trainer.evaluation.metrics.compute_bleu_score') as mock_bleu:
                mock_perplexity.return_value = 15.5
                mock_bleu.return_value = 0.75
                
                metrics = evaluate_model(mock_model, mock_eval_dataset)
                
                assert "perplexity" in metrics
                assert "bleu_score" in metrics
                assert metrics["perplexity"] == 15.5
                assert metrics["bleu_score"] == 0.75

    def test_generate_sample_outputs(self):
        """Test sample output generation for evaluation"""
        from newsies_trainer.evaluation.generation import generate_samples
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.decode.return_value = "Generated sample text"
        
        samples = generate_samples(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompts=["Test prompt"],
            max_length=50
        )
        
        assert len(samples) == 1
        assert samples[0] == "Generated sample text"


@pytest.mark.integration
class TestTrainerIntegration:
    """Integration tests for trainer components"""

    def test_end_to_end_training_workflow(self):
        """Test complete training workflow"""
        # This would test the complete workflow with mocked external dependencies
        task_id = "integration_test_task"
        
        with patch('newsies_trainer.pipelines.train_model.redis_task_status') as mock_redis:
            with patch('newsies_trainer.llm.load_latest.load_model') as mock_loader:
                with patch('newsies_trainer.pipelines.train_model.load_training_data') as mock_data:
                    # Mock the complete workflow
                    mock_loader.return_value = Mock()
                    mock_data.return_value = {"train": [], "eval": []}
                    
                    try:
                        train_model_pipeline(task_id)
                    except Exception as e:
                        # Expected due to missing actual model/data
                        assert "mock" in str(e).lower() or "tensor" in str(e).lower()


@pytest.mark.unit
class TestTrainerUtilities:
    """Test utility functions in trainer package"""

    def test_model_size_calculation(self):
        """Test model size calculation utility"""
        from newsies_trainer.utils.model_info import calculate_model_size
        
        mock_model = Mock()
        mock_model.parameters.return_value = [
            torch.randn(100, 50),  # 5000 parameters
            torch.randn(50, 25)    # 1250 parameters
        ]
        
        size = calculate_model_size(mock_model)
        assert size == 6250  # Total parameters

    def test_checkpoint_management(self):
        """Test model checkpoint management"""
        from newsies_trainer.utils.checkpoints import save_checkpoint, load_checkpoint
        
        mock_model = Mock()
        mock_optimizer = Mock()
        checkpoint_path = "/tmp/test_checkpoint.pt"
        
        with patch('torch.save') as mock_save:
            save_checkpoint(mock_model, mock_optimizer, 100, checkpoint_path)
            mock_save.assert_called_once()
        
        with patch('torch.load') as mock_load:
            mock_load.return_value = {
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "epoch": 100
            }
            
            checkpoint = load_checkpoint(checkpoint_path)
            assert checkpoint["epoch"] == 100

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        from newsies_trainer.utils.scheduling import get_lr_scheduler
        
        mock_optimizer = Mock()
        
        with patch('transformers.get_linear_schedule_with_warmup') as mock_scheduler:
            scheduler = get_lr_scheduler(
                optimizer=mock_optimizer,
                num_warmup_steps=100,
                num_training_steps=1000
            )
            
            mock_scheduler.assert_called_once_with(
                mock_optimizer, 100, 1000
            )
