"""
Test suite for Redis client functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import redis

from newsies_clients.redis_client import RedisClient


class TestRedisClient:
    """Test cases for Redis client operations"""

    @pytest.fixture
    def mock_redis_connection(self):
        """Mock Redis connection for testing"""
        with patch('newsies_clients.redis_client.redis.Redis') as mock:
            yield mock

    @pytest.fixture
    def redis_client(self, mock_redis_connection):
        """RedisClient instance with mocked connection"""
        return RedisClient()

    def test_redis_client_initialization(self, mock_redis_connection):
        """Test Redis client initialization"""
        client = RedisClient(host='localhost', port=6379, db=0)
        
        mock_redis_connection.assert_called_once()
        assert client is not None

    def test_set_and_get_operations(self, redis_client, mock_redis_connection):
        """Test basic set and get operations"""
        mock_conn = mock_redis_connection.return_value
        mock_conn.set.return_value = True
        mock_conn.get.return_value = b'test_value'
        
        # Test set operation
        result = redis_client.set('test_key', 'test_value')
        assert result is True
        mock_conn.set.assert_called_once_with('test_key', 'test_value')
        
        # Test get operation
        value = redis_client.get('test_key')
        assert value == 'test_value'
        mock_conn.get.assert_called_once_with('test_key')

    def test_json_operations(self, redis_client, mock_redis_connection):
        """Test JSON serialization/deserialization operations"""
        mock_conn = mock_redis_connection.return_value
        test_data = {'key': 'value', 'number': 42}
        
        mock_conn.set.return_value = True
        mock_conn.get.return_value = json.dumps(test_data).encode('utf-8')
        
        # Test set JSON
        result = redis_client.set_json('test_key', test_data)
        assert result is True
        
        # Test get JSON
        retrieved_data = redis_client.get_json('test_key')
        assert retrieved_data == test_data

    def test_list_operations(self, redis_client, mock_redis_connection):
        """Test Redis list operations"""
        mock_conn = mock_redis_connection.return_value
        mock_conn.lpush.return_value = 1
        mock_conn.rpop.return_value = b'item1'
        mock_conn.llen.return_value = 5
        
        # Test list push
        result = redis_client.list_push('test_list', 'item1')
        assert result == 1
        
        # Test list pop
        item = redis_client.list_pop('test_list')
        assert item == 'item1'
        
        # Test list length
        length = redis_client.list_length('test_list')
        assert length == 5

    def test_hash_operations(self, redis_client, mock_redis_connection):
        """Test Redis hash operations"""
        mock_conn = mock_redis_connection.return_value
        mock_conn.hset.return_value = 1
        mock_conn.hget.return_value = b'field_value'
        mock_conn.hgetall.return_value = {b'field1': b'value1', b'field2': b'value2'}
        
        # Test hash set
        result = redis_client.hash_set('test_hash', 'field1', 'field_value')
        assert result == 1
        
        # Test hash get
        value = redis_client.hash_get('test_hash', 'field1')
        assert value == 'field_value'
        
        # Test hash get all
        all_values = redis_client.hash_get_all('test_hash')
        assert all_values == {'field1': 'value1', 'field2': 'value2'}

    def test_expiration_operations(self, redis_client, mock_redis_connection):
        """Test key expiration operations"""
        mock_conn = mock_redis_connection.return_value
        mock_conn.expire.return_value = True
        mock_conn.ttl.return_value = 3600
        
        # Test set expiration
        result = redis_client.set_expiration('test_key', 3600)
        assert result is True
        
        # Test get TTL
        ttl = redis_client.get_ttl('test_key')
        assert ttl == 3600

    def test_connection_error_handling(self, redis_client):
        """Test handling of Redis connection errors"""
        with patch('newsies_clients.redis_client.redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = redis.ConnectionError("Connection failed")
            
            # Should handle connection error gracefully
            result = redis_client.health_check()
            assert result is False

    def test_key_pattern_operations(self, redis_client, mock_redis_connection):
        """Test key pattern matching operations"""
        mock_conn = mock_redis_connection.return_value
        mock_conn.keys.return_value = [b'task:1', b'task:2', b'task:3']
        
        keys = redis_client.get_keys_by_pattern('task:*')
        assert len(keys) == 3
        assert 'task:1' in keys

    def test_batch_operations(self, redis_client, mock_redis_connection):
        """Test batch operations using pipeline"""
        mock_conn = mock_redis_connection.return_value
        mock_pipeline = Mock()
        mock_conn.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True, True, True]
        
        operations = [
            ('set', 'key1', 'value1'),
            ('set', 'key2', 'value2'),
            ('set', 'key3', 'value3')
        ]
        
        results = redis_client.batch_execute(operations)
        assert len(results) == 3
        assert all(result is True for result in results)

    @pytest.mark.integration
    def test_redis_integration(self):
        """Integration test with real Redis instance"""
        try:
            client = RedisClient(host='localhost', port=6379)
            
            # Test connection
            assert client.health_check() is True
            
            # Test basic operations
            assert client.set('test_integration_key', 'test_value') is True
            assert client.get('test_integration_key') == 'test_value'
            
            # Cleanup
            client.delete('test_integration_key')
            
        except redis.ConnectionError:
            pytest.skip("Redis not available for integration test")


class TestRedisClientConfiguration:
    """Test Redis client configuration options"""

    def test_custom_configuration(self):
        """Test Redis client with custom configuration"""
        config = {
            'host': 'custom-host',
            'port': 6380,
            'db': 1,
            'password': 'secret',
            'socket_timeout': 30
        }
        
        with patch('newsies_clients.redis_client.redis.Redis') as mock_redis:
            client = RedisClient(**config)
            
            mock_redis.assert_called_once_with(
                host='custom-host',
                port=6380,
                db=1,
                password='secret',
                socket_timeout=30,
                decode_responses=True
            )

    def test_connection_pool_configuration(self):
        """Test Redis client with connection pool"""
        with patch('newsies_clients.redis_client.redis.ConnectionPool') as mock_pool:
            with patch('newsies_clients.redis_client.redis.Redis') as mock_redis:
                client = RedisClient(use_connection_pool=True, max_connections=20)
                
                mock_pool.assert_called_once()
                mock_redis.assert_called_once()

    def test_ssl_configuration(self):
        """Test Redis client with SSL configuration"""
        ssl_config = {
            'ssl': True,
            'ssl_cert_reqs': 'required',
            'ssl_ca_certs': '/path/to/ca.pem'
        }
        
        with patch('newsies_clients.redis_client.redis.Redis') as mock_redis:
            client = RedisClient(**ssl_config)
            
            call_args = mock_redis.call_args[1]
            assert call_args['ssl'] is True
            assert call_args['ssl_cert_reqs'] == 'required'


@pytest.mark.unit
class TestRedisClientUtilities:
    """Test utility functions in Redis client"""

    def test_serialize_data(self):
        """Test data serialization"""
        from newsies_clients.redis_client import serialize_data
        
        # Test various data types
        assert serialize_data('string') == 'string'
        assert serialize_data(42) == '42'
        assert serialize_data({'key': 'value'}) == '{"key": "value"}'
        assert serialize_data([1, 2, 3]) == '[1, 2, 3]'

    def test_deserialize_data(self):
        """Test data deserialization"""
        from newsies_clients.redis_client import deserialize_data
        
        # Test JSON deserialization
        json_data = '{"key": "value"}'
        result = deserialize_data(json_data.encode('utf-8'))
        assert result == {"key": "value"}
        
        # Test string deserialization
        string_data = b'simple string'
        result = deserialize_data(string_data)
        assert result == 'simple string'

    def test_build_key(self):
        """Test Redis key building utility"""
        from newsies_clients.redis_client import build_key
        
        # Test key building with prefix
        key = build_key('task', '123', 'status')
        assert key == 'task:123:status'
        
        # Test key building with namespace
        key = build_key('user', 'john', namespace='newsies')
        assert key == 'newsies:user:john'

    def test_validate_key(self):
        """Test Redis key validation"""
        from newsies_clients.redis_client import validate_key
        
        # Valid keys
        assert validate_key('valid_key') is True
        assert validate_key('task:123') is True
        
        # Invalid keys
        assert validate_key('') is False
        assert validate_key(None) is False
        assert validate_key('key with spaces') is False
