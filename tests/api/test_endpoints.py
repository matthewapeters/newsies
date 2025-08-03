"""
Test suite for Newsies API endpoints
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import json

from newsies_api.api.app import app


class TestNewsiesAPIEndpoints:
    """Test cases for Newsies API endpoints"""

    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_redis_task_status(self):
        """Mock Redis task status for testing"""
        with patch('newsies_api.api.app.redis_task_status') as mock:
            yield mock

    @pytest.fixture
    def mock_background_tasks(self):
        """Mock FastAPI background tasks"""
        with patch('newsies_api.api.app.BackgroundTasks') as mock:
            yield mock

    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_get_news_endpoint_success(self, client, mock_redis_task_status, mock_background_tasks):
        """Test successful get-news endpoint execution"""
        # Mock task status creation
        mock_redis_task_status.set_status.return_value = True
        mock_redis_task_status.get_status.return_value = {
            "task_id": "test_task_123",
            "status": "queued",
            "task": "get-articles"
        }
        
        response = client.post("/v1/run/get-news")
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["task"] == "get-articles"

    def test_analyze_endpoint_success(self, client, mock_redis_task_status, mock_background_tasks):
        """Test successful analyze endpoint execution"""
        mock_redis_task_status.set_status.return_value = True
        mock_redis_task_status.get_status.return_value = {
            "task_id": "analyze_task_123",
            "status": "queued",
            "task": "analyze"
        }
        
        response = client.post("/v1/run/analyze")
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["task"] == "analyze"

    def test_train_llm_endpoint_success(self, client, mock_redis_task_status, mock_background_tasks):
        """Test successful train-llm endpoint execution"""
        mock_redis_task_status.set_status.return_value = True
        mock_redis_task_status.get_status.return_value = {
            "task_id": "train_task_123",
            "status": "queued",
            "task": "train-model"
        }
        
        response = client.post("/v1/run/train-llm")
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["task"] == "train-model"

    def test_daily_pipeline_endpoint_success(self, client, mock_redis_task_status, mock_background_tasks):
        """Test successful daily pipeline endpoint execution"""
        mock_redis_task_status.set_status.return_value = True
        mock_redis_task_status.get_status.return_value = {
            "task_id": "daily_task_123",
            "status": "queued",
            "task": "daily-pipeline"
        }
        
        response = client.post("/v1/run/daily-pipeline")
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["task"] == "daily-pipeline"

    def test_get_tasks_endpoint(self, client, mock_redis_task_status):
        """Test get tasks endpoint"""
        mock_tasks = [
            {"task_id": "task_1", "status": "running", "task": "get-articles"},
            {"task_id": "task_2", "status": "complete", "task": "analyze"},
            {"task_id": "task_3", "status": "queued", "task": "train-model"}
        ]
        mock_redis_task_status.list_all_tasks.return_value = mock_tasks
        
        response = client.get("/v1/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 3
        assert data["tasks"][0]["task_id"] == "task_1"

    def test_get_task_status_endpoint(self, client, mock_redis_task_status):
        """Test get specific task status endpoint"""
        task_id = "test_task_123"
        mock_task_data = {
            "task_id": task_id,
            "status": "running",
            "task": "get-articles",
            "progress": 50
        }
        mock_redis_task_status.get_status.return_value = mock_task_data
        
        response = client.get(f"/v1/tasks/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "running"
        assert data["progress"] == 50

    def test_get_task_status_not_found(self, client, mock_redis_task_status):
        """Test get task status for non-existent task"""
        task_id = "nonexistent_task"
        mock_redis_task_status.get_status.return_value = None
        
        response = client.get(f"/v1/tasks/{task_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_collections_endpoint(self, client):
        """Test get collections endpoint"""
        with patch('newsies_api.api.app.chromadb_client') as mock_chromadb:
            mock_collections = [
                {"name": "news_articles", "count": 1500},
                {"name": "summaries", "count": 750}
            ]
            mock_chromadb.list_collections.return_value = mock_collections
            
            response = client.get("/v1/collections")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["collections"]) == 2
            assert data["collections"][0]["name"] == "news_articles"

    def test_query_endpoint_success(self, client):
        """Test successful query endpoint"""
        with patch('newsies_api.api.app.session_client') as mock_session:
            mock_session.query.return_value = {
                "response": "Test query response",
                "sources": ["source1", "source2"],
                "session_id": "session_123"
            }
            
            query_data = {"query": "What's the latest news?"}
            response = client.post("/v1/query", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Test query response"
            assert len(data["sources"]) == 2

    def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query"""
        query_data = {"query": ""}
        response = client.post("/v1/query", json=query_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower()

    def test_redis_connection_error(self, client):
        """Test handling Redis connection errors"""
        with patch('newsies_api.api.app.redis_task_status') as mock_redis:
            mock_redis.set_status.side_effect = ConnectionError("Redis unavailable")
            
            response = client.post("/v1/run/get-news")
            
            assert response.status_code == 503
            data = response.json()
            assert "service unavailable" in data["detail"].lower()

    def test_authentication_required(self, client):
        """Test endpoints that require authentication"""
        # Test without authentication
        response = client.post("/v1/admin/reset-tasks")
        
        assert response.status_code == 401

    def test_rate_limiting(self, client):
        """Test API rate limiting"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.post("/v1/run/get-news")
            responses.append(response)
        
        # Check if rate limiting is applied
        status_codes = [r.status_code for r in responses]
        assert any(code == 429 for code in status_codes)  # Too Many Requests

    @pytest.mark.integration
    def test_full_workflow_integration(self, client):
        """Integration test for complete workflow"""
        # 1. Start get-news task
        response = client.post("/v1/run/get-news")
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        
        # 2. Check task status
        response = client.get(f"/v1/tasks/{task_id}")
        assert response.status_code == 200
        
        # 3. List all tasks
        response = client.get("/v1/tasks")
        assert response.status_code == 200
        assert any(task["task_id"] == task_id for task in response.json()["tasks"])

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/v1/run/get-news")
        
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    def test_content_type_validation(self, client):
        """Test content type validation"""
        # Send invalid content type
        response = client.post("/v1/query", 
                             data="invalid data", 
                             headers={"Content-Type": "text/plain"})
        
        assert response.status_code == 415  # Unsupported Media Type


@pytest.mark.unit
class TestAPIUtilities:
    """Test API utility functions"""

    def test_generate_task_id(self):
        """Test task ID generation"""
        from newsies_api.api.app import generate_task_id
        
        task_id1 = generate_task_id()
        task_id2 = generate_task_id()
        
        assert task_id1 != task_id2
        assert len(task_id1) > 10  # Reasonable length
        assert isinstance(task_id1, str)

    def test_validate_query_input(self):
        """Test query input validation"""
        from newsies_api.api.app import validate_query_input
        
        # Valid query
        assert validate_query_input("What's the news?") is True
        
        # Invalid queries
        assert validate_query_input("") is False
        assert validate_query_input(None) is False
        assert validate_query_input("x" * 10000) is False  # Too long

    def test_format_task_response(self):
        """Test task response formatting"""
        from newsies_api.api.app import format_task_response
        
        task_data = {
            "task_id": "test_123",
            "status": "running",
            "task": "get-articles"
        }
        
        formatted = format_task_response(task_data)
        
        assert "task_id" in formatted
        assert "status" in formatted
        assert "timestamp" in formatted
