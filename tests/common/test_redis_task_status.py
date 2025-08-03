"""
Test suite for Redis-based distributed task status system
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from newsies_common.redis_task_status import RedisTaskStatus


class TestRedisTaskStatus:
    """Test cases for RedisTaskStatus class"""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing"""
        with patch('newsies_common.redis_task_status.redis_client') as mock:
            yield mock

    @pytest.fixture
    def task_status(self, mock_redis):
        """RedisTaskStatus instance with mocked Redis"""
        return RedisTaskStatus()

    def test_set_task_status_success(self, task_status, mock_redis):
        """Test successful task status setting"""
        task_id = "test_task_123"
        status = "running"
        
        mock_redis.set.return_value = True
        
        result = task_status.set_status(task_id, status)
        
        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == f"task:{task_id}"
        
        # Verify the stored data structure
        stored_data = json.loads(call_args[0][1])
        assert stored_data["status"] == status
        assert "timestamp" in stored_data
        assert "task_id" in stored_data

    def test_get_task_status_success(self, task_status, mock_redis):
        """Test successful task status retrieval"""
        task_id = "test_task_123"
        expected_data = {
            "task_id": task_id,
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "session_id": "session_123",
            "username": "testuser"
        }
        
        mock_redis.get.return_value = json.dumps(expected_data).encode('utf-8')
        
        result = task_status.get_status(task_id)
        
        assert result == expected_data
        mock_redis.get.assert_called_once_with(f"task:{task_id}")

    def test_get_task_status_not_found(self, task_status, mock_redis):
        """Test task status retrieval when task doesn't exist"""
        task_id = "nonexistent_task"
        mock_redis.get.return_value = None
        
        result = task_status.get_status(task_id)
        
        assert result is None
        mock_redis.get.assert_called_once_with(f"task:{task_id}")

    def test_update_task_status(self, task_status, mock_redis):
        """Test updating existing task status"""
        task_id = "test_task_123"
        existing_data = {
            "task_id": task_id,
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }
        
        mock_redis.get.return_value = json.dumps(existing_data).encode('utf-8')
        mock_redis.set.return_value = True
        
        updates = {"status": "complete", "result": "success"}
        result = task_status.update_status(task_id, updates)
        
        assert result is True
        mock_redis.set.assert_called_once()
        
        # Verify the updated data
        call_args = mock_redis.set.call_args
        updated_data = json.loads(call_args[0][1])
        assert updated_data["status"] == "complete"
        assert updated_data["result"] == "success"
        assert updated_data["task_id"] == task_id

    def test_list_tasks_by_status(self, task_status, mock_redis):
        """Test listing tasks by status"""
        mock_redis.scan_iter.return_value = [
            b"task:task_1", b"task:task_2", b"task:task_3"
        ]
        
        task_data = [
            {"task_id": "task_1", "status": "running"},
            {"task_id": "task_2", "status": "complete"},
            {"task_id": "task_3", "status": "running"}
        ]
        
        mock_redis.get.side_effect = [
            json.dumps(data).encode('utf-8') for data in task_data
        ]
        
        running_tasks = task_status.list_tasks_by_status("running")
        
        assert len(running_tasks) == 2
        assert all(task["status"] == "running" for task in running_tasks)

    def test_delete_task(self, task_status, mock_redis):
        """Test task deletion"""
        task_id = "test_task_123"
        mock_redis.delete.return_value = 1
        
        result = task_status.delete_task(task_id)
        
        assert result is True
        mock_redis.delete.assert_called_once_with(f"task:{task_id}")

    def test_redis_connection_error(self, task_status):
        """Test handling Redis connection errors"""
        with patch('newsies_common.redis_task_status.redis_client') as mock_redis:
            mock_redis.set.side_effect = ConnectionError("Redis unavailable")
            
            result = task_status.set_status("test_task", "running")
            
            assert result is False

    def test_task_expiration(self, task_status, mock_redis):
        """Test task expiration setting"""
        task_id = "test_task_123"
        status = "running"
        expiration_seconds = 3600
        
        mock_redis.set.return_value = True
        
        task_status.set_status(task_id, status, expire_seconds=expiration_seconds)
        
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[1]['ex'] == expiration_seconds

    @pytest.mark.integration
    def test_redis_integration(self):
        """Integration test with real Redis (requires Redis running)"""
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            client.ping()
            
            task_status = RedisTaskStatus()
            task_id = "integration_test_task"
            
            # Test full workflow
            assert task_status.set_status(task_id, "running") is True
            
            retrieved = task_status.get_status(task_id)
            assert retrieved["status"] == "running"
            
            assert task_status.update_status(task_id, {"status": "complete"}) is True
            
            updated = task_status.get_status(task_id)
            assert updated["status"] == "complete"
            
            assert task_status.delete_task(task_id) is True
            
        except (ImportError, redis.ConnectionError):
            pytest.skip("Redis not available for integration test")


@pytest.mark.unit
class TestRedisTaskStatusEdgeCases:
    """Test edge cases and error conditions"""

    def test_invalid_json_data(self):
        """Test handling of corrupted JSON data in Redis"""
        with patch('newsies_common.redis_task_status.redis_client') as mock_redis:
            mock_redis.get.return_value = b"invalid json data"
            
            task_status = RedisTaskStatus()
            result = task_status.get_status("test_task")
            
            assert result is None

    def test_empty_task_id(self):
        """Test handling of empty task ID"""
        task_status = RedisTaskStatus()
        
        with pytest.raises(ValueError):
            task_status.set_status("", "running")

    def test_none_task_id(self):
        """Test handling of None task ID"""
        task_status = RedisTaskStatus()
        
        with pytest.raises(ValueError):
            task_status.set_status(None, "running")

    def test_large_status_data(self):
        """Test handling of large status data"""
        with patch('newsies_common.redis_task_status.redis_client') as mock_redis:
            mock_redis.set.return_value = True
            
            task_status = RedisTaskStatus()
            large_data = "x" * 10000  # 10KB of data
            
            result = task_status.set_status("test_task", "running", 
                                          metadata={"large_field": large_data})
            
            assert result is True
            mock_redis.set.assert_called_once()
