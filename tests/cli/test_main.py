"""
Test suite for Newsies CLI Main Interface
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from newsies_cli.main import main, dispatch_command, interactive_mode


class TestCLIMain:
    """Test cases for CLI main interface"""

    @pytest.fixture
    def mock_redis_task_status(self):
        """Mock Redis task status for testing"""
        with patch('newsies_cli.main.redis_task_status') as mock:
            yield mock

    @pytest.fixture
    def mock_pipeline_calls(self):
        """Mock pipeline execution calls"""
        with patch('newsies_cli.main.run_get_articles') as mock_get:
            with patch('newsies_cli.main.run_analyze') as mock_analyze:
                with patch('newsies_cli.main.run_train_model') as mock_train:
                    yield {
                        'get_articles': mock_get,
                        'analyze': mock_analyze,
                        'train_model': mock_train
                    }

    def test_main_get_articles_command(self, mock_redis_task_status, mock_pipeline_calls):
        """Test main function with get-articles command"""
        test_args = ['newsies-cli', 'get-articles']
        
        with patch('sys.argv', test_args):
            main()
        
        mock_pipeline_calls['get_articles'].assert_called_once()
        mock_redis_task_status.create_task.assert_called_once()

    def test_main_analyze_command(self, mock_redis_task_status, mock_pipeline_calls):
        """Test main function with analyze command"""
        test_args = ['newsies-cli', 'analyze']
        
        with patch('sys.argv', test_args):
            main()
        
        mock_pipeline_calls['analyze'].assert_called_once()
        mock_redis_task_status.create_task.assert_called_once()

    def test_main_train_command(self, mock_redis_task_status, mock_pipeline_calls):
        """Test main function with train-model command"""
        test_args = ['newsies-cli', 'train-model']
        
        with patch('sys.argv', test_args):
            main()
        
        mock_pipeline_calls['train_model'].assert_called_once()
        mock_redis_task_status.create_task.assert_called_once()

    def test_main_interactive_mode(self, mock_redis_task_status):
        """Test main function entering interactive mode"""
        test_args = ['newsies-cli', 'cli']
        
        with patch('sys.argv', test_args):
            with patch('newsies_cli.main.interactive_mode') as mock_interactive:
                main()
        
        mock_interactive.assert_called_once()

    def test_main_invalid_command(self):
        """Test main function with invalid command"""
        test_args = ['newsies-cli', 'invalid-command']
        
        with patch('sys.argv', test_args):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                
                error_output = mock_stderr.getvalue()
                assert "invalid-command" in error_output or "usage" in error_output.lower()

    def test_dispatch_command_success(self, mock_pipeline_calls):
        """Test successful command dispatch"""
        task_id = "test_task_123"
        
        result = dispatch_command('get-articles', task_id)
        
        mock_pipeline_calls['get_articles'].assert_called_once_with(task_id=task_id)
        assert result is True

    def test_dispatch_command_failure(self, mock_pipeline_calls):
        """Test command dispatch with pipeline failure"""
        task_id = "test_task_fail"
        mock_pipeline_calls['get_articles'].side_effect = Exception("Pipeline failed")
        
        result = dispatch_command('get-articles', task_id)
        
        assert result is False

    def test_command_with_task_id_argument(self, mock_redis_task_status, mock_pipeline_calls):
        """Test command execution with explicit task ID"""
        test_args = ['newsies-cli', 'get-articles', '--task-id', 'custom_task_123']
        
        with patch('sys.argv', test_args):
            main()
        
        # Should use provided task ID instead of creating new one
        mock_redis_task_status.create_task.assert_not_called()
        mock_pipeline_calls['get_articles'].assert_called_once()


class TestInteractiveMode:
    """Test interactive CLI mode"""

    @pytest.fixture
    def mock_input_output(self):
        """Mock input/output for interactive testing"""
        with patch('builtins.input') as mock_input:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                yield mock_input, mock_stdout

    def test_interactive_mode_help_command(self, mock_input_output):
        """Test help command in interactive mode"""
        mock_input, mock_stdout = mock_input_output
        mock_input.side_effect = ['help', 'exit']
        
        interactive_mode()
        
        output = mock_stdout.getvalue()
        assert "available commands" in output.lower() or "help" in output.lower()

    def test_interactive_mode_status_command(self, mock_input_output):
        """Test status command in interactive mode"""
        mock_input, mock_stdout = mock_input_output
        mock_input.side_effect = ['status', 'exit']
        
        with patch('newsies_cli.main.get_task_status') as mock_status:
            mock_status.return_value = {"active_tasks": 2, "completed_tasks": 5}
            
            interactive_mode()
            
            output = mock_stdout.getvalue()
            assert "active" in output.lower() or "tasks" in output.lower()

    def test_interactive_mode_pipeline_execution(self, mock_input_output):
        """Test pipeline execution in interactive mode"""
        mock_input, mock_stdout = mock_input_output
        mock_input.side_effect = ['get-articles', 'exit']
        
        with patch('newsies_cli.main.dispatch_command') as mock_dispatch:
            mock_dispatch.return_value = True
            
            interactive_mode()
            
            mock_dispatch.assert_called_once_with('get-articles', unittest.mock.ANY)

    def test_interactive_mode_invalid_command(self, mock_input_output):
        """Test invalid command in interactive mode"""
        mock_input, mock_stdout = mock_input_output
        mock_input.side_effect = ['invalid-command', 'exit']
        
        interactive_mode()
        
        output = mock_stdout.getvalue()
        assert "unknown" in output.lower() or "invalid" in output.lower()

    def test_interactive_mode_exit_commands(self, mock_input_output):
        """Test various exit commands in interactive mode"""
        mock_input, mock_stdout = mock_input_output
        
        # Test 'exit' command
        mock_input.side_effect = ['exit']
        interactive_mode()
        
        # Test 'quit' command
        mock_input.side_effect = ['quit']
        interactive_mode()
        
        # Test Ctrl+C handling
        mock_input.side_effect = KeyboardInterrupt()
        interactive_mode()  # Should handle gracefully


class TestCLIUtilities:
    """Test CLI utility functions"""

    def test_format_task_status(self):
        """Test task status formatting"""
        from newsies_cli.utils.formatting import format_task_status
        
        task_status = {
            "task_id": "task_123",
            "status": "running",
            "progress": 75,
            "message": "Processing articles"
        }
        
        formatted = format_task_status(task_status)
        
        assert "task_123" in formatted
        assert "running" in formatted
        assert "75%" in formatted
        assert "Processing articles" in formatted

    def test_format_pipeline_results(self):
        """Test pipeline results formatting"""
        from newsies_cli.utils.formatting import format_pipeline_results
        
        results = {
            "articles_processed": 50,
            "summaries_generated": 45,
            "entities_extracted": 200,
            "execution_time": 120.5
        }
        
        formatted = format_pipeline_results(results)
        
        assert "50" in formatted
        assert "45" in formatted
        assert "200" in formatted
        assert "120.5" in formatted or "2:00" in formatted

    def test_validate_command_arguments(self):
        """Test command argument validation"""
        from newsies_cli.utils.validation import validate_command_args
        
        # Valid arguments
        valid_args = {
            'command': 'get-articles',
            'task_id': 'valid_task_123'
        }
        assert validate_command_args(valid_args) is True
        
        # Invalid arguments
        invalid_args = {
            'command': '',
            'task_id': None
        }
        assert validate_command_args(invalid_args) is False

    def test_parse_cli_arguments(self):
        """Test CLI argument parsing"""
        from newsies_cli.utils.args import parse_arguments
        
        test_args = ['get-articles', '--task-id', 'custom_task', '--verbose']
        
        parsed = parse_arguments(test_args)
        
        assert parsed.command == 'get-articles'
        assert parsed.task_id == 'custom_task'
        assert parsed.verbose is True


class TestCLIConfiguration:
    """Test CLI configuration management"""

    def test_load_cli_config(self):
        """Test loading CLI configuration"""
        from newsies_cli.config.settings import load_config
        
        with patch('newsies_cli.config.settings.os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"verbose": true, "timeout": 300}')):
                config = load_config()
                
                assert config['verbose'] is True
                assert config['timeout'] == 300

    def test_default_cli_config(self):
        """Test default CLI configuration"""
        from newsies_cli.config.settings import get_default_config
        
        config = get_default_config()
        
        assert 'verbose' in config
        assert 'timeout' in config
        assert 'redis_host' in config
        assert isinstance(config['verbose'], bool)

    def test_save_cli_config(self):
        """Test saving CLI configuration"""
        from newsies_cli.config.settings import save_config
        
        config = {'verbose': True, 'timeout': 600}
        
        with patch('builtins.open', mock_open()) as mock_file:
            save_config(config)
            
            mock_file.assert_called_once()
            # Verify JSON was written
            handle = mock_file()
            written_data = ''.join(call[0][0] for call in handle.write.call_args_list)
            assert 'verbose' in written_data
            assert 'timeout' in written_data


class TestCLILogging:
    """Test CLI logging functionality"""

    def test_setup_cli_logging(self):
        """Test CLI logging setup"""
        from newsies_cli.utils.logging import setup_logging
        
        with patch('logging.basicConfig') as mock_config:
            setup_logging(verbose=True)
            
            mock_config.assert_called_once()
            call_args = mock_config.call_args[1]
            assert call_args['level'] == logging.DEBUG

    def test_log_command_execution(self):
        """Test command execution logging"""
        from newsies_cli.utils.logging import log_command_execution
        
        with patch('newsies_cli.utils.logging.logger') as mock_logger:
            log_command_execution('get-articles', 'task_123', success=True)
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert 'get-articles' in call_args
            assert 'task_123' in call_args

    def test_log_error_handling(self):
        """Test error logging"""
        from newsies_cli.utils.logging import log_error
        
        with patch('newsies_cli.utils.logging.logger') as mock_logger:
            test_error = Exception("Test error message")
            log_error(test_error, context="CLI execution")
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert 'Test error message' in call_args
            assert 'CLI execution' in call_args


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI components"""

    def test_full_cli_workflow(self):
        """Test complete CLI workflow"""
        # This would test the complete workflow with mocked external dependencies
        with patch('newsies_cli.main.redis_task_status') as mock_redis:
            with patch('newsies_cli.main.run_get_articles') as mock_pipeline:
                mock_redis.create_task.return_value = "integration_test_task"
                mock_pipeline.return_value = True
                
                # Test command line execution
                test_args = ['newsies-cli', 'get-articles']
                with patch('sys.argv', test_args):
                    main()
                
                mock_redis.create_task.assert_called_once()
                mock_pipeline.assert_called_once()

    def test_cli_with_redis_unavailable(self):
        """Test CLI behavior when Redis is unavailable"""
        with patch('newsies_cli.main.redis_task_status') as mock_redis:
            mock_redis.create_task.side_effect = Exception("Redis connection failed")
            
            test_args = ['newsies-cli', 'get-articles']
            with patch('sys.argv', test_args):
                with pytest.raises(SystemExit):
                    main()


@pytest.mark.unit
class TestCLIHelpers:
    """Test CLI helper functions"""

    def test_generate_task_id(self):
        """Test task ID generation"""
        from newsies_cli.utils.helpers import generate_task_id
        
        task_id1 = generate_task_id()
        task_id2 = generate_task_id()
        
        assert task_id1 != task_id2
        assert len(task_id1) > 0
        assert isinstance(task_id1, str)

    def test_format_duration(self):
        """Test duration formatting"""
        from newsies_cli.utils.helpers import format_duration
        
        # Test seconds
        assert format_duration(45) == "45s"
        
        # Test minutes and seconds
        assert format_duration(125) == "2m 5s"
        
        # Test hours, minutes, and seconds
        assert format_duration(3725) == "1h 2m 5s"

    def test_truncate_text(self):
        """Test text truncation utility"""
        from newsies_cli.utils.helpers import truncate_text
        
        long_text = "This is a very long text that should be truncated"
        truncated = truncate_text(long_text, max_length=20)
        
        assert len(truncated) <= 23  # 20 + "..."
        assert truncated.endswith("...")

    def test_colorize_output(self):
        """Test output colorization"""
        from newsies_cli.utils.helpers import colorize
        
        # Test with color support
        with patch('newsies_cli.utils.helpers.supports_color', return_value=True):
            colored = colorize("Success", "green")
            assert "\033[" in colored  # ANSI escape codes
        
        # Test without color support
        with patch('newsies_cli.utils.helpers.supports_color', return_value=False):
            plain = colorize("Success", "green")
            assert plain == "Success"
