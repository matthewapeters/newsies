"""
Redis-based distributed task status system
Replaces in-memory TASK_STATUS for Kubernetes microservices
"""

import json
import redis
import os
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional


class RedisTaskStatus:
    """
    Redis-based distributed task status system
    Replaces the in-memory AppStatus for microservices architecture
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize Redis task status system"""
        if redis_client:
            self.redis = redis_client
        else:
            # Create Redis client with environment variable support
            try:
                user, passwd = os.environ["REDIS_CREDS"].split(":")
            except KeyError:
                # Default values for development/testing
                user, passwd = "default", "password"
            
            self.redis = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                db=int(os.environ.get("REDIS_DB", "0")),
                username=user,
                password=passwd,
                decode_responses=True  # Automatically decode responses to strings
            )
        
        # Key prefix for task status entries
        self.key_prefix = "newsies:task_status:"
        
    def __setitem__(self, task_id: str, value: Union[str, Dict[str, str]]):
        """
        Set task status in Redis
        :param task_id: str task id uuid
        :param value: Union[str,Dict[str,str]]
            task details when queued (Dict[str,str]) or status update (str)
        """
        key = f"{self.key_prefix}{task_id}"
        
        # Handle status updates vs new task creation
        if isinstance(value, str):
            # This is a status update - get existing record and update status
            existing_data = self.redis.get(key)
            if existing_data:
                task_record = json.loads(existing_data)
                task_record["status"] = value
            else:
                # Create minimal record if doesn't exist
                task_record = {
                    "task_id": task_id,
                    "status": value,
                    "session_id": "N/A",
                    "task": "unknown",
                    "username": "system"
                }
        else:
            # This is a new task record
            task_record = dict(value)
            task_record["task_id"] = task_id
        
        # Set timestamp
        task_record["timestamp"] = datetime.now().isoformat()
        
        # Store in Redis with expiration (24 hours)
        self.redis.setex(key, 86400, json.dumps(task_record))
        
        # Also log to file for compatibility
        self._log_status(task_id, task_record)
    
    def __getitem__(self, task_id: str) -> Dict[str, str]:
        """Get task status from Redis"""
        key = f"{self.key_prefix}{task_id}"
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        raise KeyError(f"Task {task_id} not found")
    
    def __contains__(self, task_id: str) -> bool:
        """Check if task exists in Redis"""
        key = f"{self.key_prefix}{task_id}"
        return self.redis.exists(key) > 0
    
    def __delitem__(self, task_id: str):
        """Delete task from Redis"""
        key = f"{self.key_prefix}{task_id}"
        self.redis.delete(key)
    
    def get(self, task_id: str, default=None) -> Optional[Dict[str, str]]:
        """Get task status with default value"""
        try:
            return self[task_id]
        except KeyError:
            return default
    
    def keys(self) -> List[str]:
        """Get all task IDs"""
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        return [key.replace(self.key_prefix, "") for key in keys]
    
    def items(self) -> List[tuple]:
        """Get all task items as (task_id, task_data) tuples"""
        result = []
        for task_id in self.keys():
            try:
                task_data = self[task_id]
                result.append((task_id, task_data))
            except KeyError:
                continue  # Task may have expired between keys() and __getitem__
        return result
    
    def values(self) -> List[Dict[str, str]]:
        """Get all task data values"""
        return [task_data for _, task_data in self.items()]
    
    def sorted(self, complete_retention: timedelta = timedelta(hours=12)) -> List[Dict]:
        """
        Get tasks sorted by timestamp in descending order
        Automatically cleans up old completed tasks
        """
        tasks = []
        threshold = datetime.now() - complete_retention
        
        for task_id, task_data in self.items():
            # Check if task should be cleaned up
            if task_data["status"].startswith("error") or task_data["status"] in ("complete", "failed"):
                task_timestamp = datetime.fromisoformat(task_data["timestamp"])
                if task_timestamp <= threshold:
                    # Remove old completed task
                    del self[task_id]
                    continue
            
            # Add to results
            tasks.append({task_id: task_data})
        
        # Sort by timestamp descending
        tasks.sort(key=lambda t: list(t.values())[0]["timestamp"], reverse=True)
        return tasks
    
    def clear_all_tasks(self):
        """Clear all task status entries (useful for testing)"""
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
    
    def _log_status(self, task_id: str, task_record: Dict[str, str]):
        """Log status to file for compatibility with existing logging"""
        try:
            with open("newsies.log", "a", encoding="utf8") as log:
                status_record = json.dumps({"task_id": task_id, **task_record})
                log.write(f"INFO:\t{status_record}\n")
        except Exception as e:
            # Don't fail task updates due to logging issues
            print(f"Warning: Failed to log task status: {e}")


# Create global Redis-based task status instance
# This replaces the old TASK_STATUS = AppStatus()
def get_task_status() -> RedisTaskStatus:
    """Get the global Redis task status instance"""
    global _redis_task_status
    if '_redis_task_status' not in globals():
        _redis_task_status = RedisTaskStatus()
    return _redis_task_status


# For backward compatibility, provide TASK_STATUS that behaves like the old system
TASK_STATUS = get_task_status()
