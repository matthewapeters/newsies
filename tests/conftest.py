"""
Global test configuration and fixtures for Newsies test suite
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add package directories to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "newsies-common"))
sys.path.insert(0, str(project_root / "newsies-api"))
sys.path.insert(0, str(project_root / "newsies-scraper"))
sys.path.insert(0, str(project_root / "newsies-analyzer"))
sys.path.insert(0, str(project_root / "newsies-trainer"))
sys.path.insert(0, str(project_root / "newsies-cli"))
sys.path.insert(0, str(project_root / "newsies-clients"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with required configuration"""
    # Mock environment variables
    test_env = {
        'CHROMADB_HOST': 'localhost',
        'CHROMADB_PORT': '8000',
        'CHROMADB_USER': 'test_user',
        'CHROMADB_CREDS': 'test_credentials',
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': '6379',
        'REDIS_USER': 'test_user',
        'REDIS_CREDS': 'test_credentials',
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128'
    }
    
    # Set environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Cleanup environment variables
    for key in test_env.keys():
        os.environ.pop(key, None)


@pytest.fixture
def mock_redis_connection():
    """Mock Redis connection for all tests"""
    with patch('redis.Redis') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock_instance.ping.return_value = True
        mock_instance.set.return_value = True
        mock_instance.get.return_value = b'test_value'
        mock_instance.delete.return_value = 1
        mock_instance.exists.return_value = True
        yield mock_instance


@pytest.fixture
def mock_chromadb_client():
    """Mock ChromaDB client for all tests"""
    with patch('chromadb.Client') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        
        # Mock collection
        mock_collection = Mock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_collection.add.return_value = None
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'distances': [[0.1, 0.2]],
            'documents': [['Document 1', 'Document 2']]
        }
        
        yield mock_instance


@pytest.fixture
def mock_task_status():
    """Mock task status system for all tests"""
    with patch('newsies_common.redis_task_status.RedisTaskStatus') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        
        mock_instance.create_task.return_value = 'test_task_123'
        mock_instance.set_status.return_value = True
        mock_instance.get_status.return_value = 'running'
        mock_instance.complete_task.return_value = True
        mock_instance.fail_task.return_value = True
        
        yield mock_instance


@pytest.fixture
def sample_articles():
    """Sample articles for testing"""
    return [
        {
            'id': 'article_1',
            'title': 'AI Breakthrough in Healthcare',
            'content': 'Artificial intelligence has made significant strides in medical diagnosis.',
            'url': 'https://example.com/article1',
            'published_date': '2024-01-01T00:00:00Z',
            'summary': 'AI improves medical diagnosis accuracy.'
        },
        {
            'id': 'article_2',
            'title': 'Climate Change Impact',
            'content': 'Global warming continues to affect weather patterns worldwide.',
            'url': 'https://example.com/article2',
            'published_date': '2024-01-02T00:00:00Z',
            'summary': 'Climate change affects global weather patterns.'
        }
    ]


@pytest.fixture
def mock_model_components():
    """Mock ML model components for testing"""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('transformers.AutoTokenizer') as mock_tokenizer:
            with patch('transformers.AutoModel') as mock_model:
                mock_tokenizer_instance = Mock()
                mock_model_instance = Mock()
                
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                mock_model.from_pretrained.return_value = mock_model_instance
                
                # Mock tokenizer methods
                mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
                mock_tokenizer_instance.decode.return_value = "decoded text"
                
                yield {
                    'tokenizer': mock_tokenizer_instance,
                    'model': mock_model_instance
                }


@pytest.fixture
def mock_external_apis():
    """Mock external API calls for testing"""
    with patch('requests.get') as mock_get:
        with patch('requests.post') as mock_post:
            # Mock successful API responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'success', 'data': []}
            mock_response.text = 'Mock response text'
            
            mock_get.return_value = mock_response
            mock_post.return_value = mock_response
            
            yield {
                'get': mock_get,
                'post': mock_post,
                'response': mock_response
            }


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests when CUDA unavailable"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    
    if not cuda_available:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
