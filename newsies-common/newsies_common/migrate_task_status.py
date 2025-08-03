"""
Migration utility to transition from in-memory TASK_STATUS to Redis-based system
"""

import sys
import importlib
from typing import Dict, Any

def migrate_to_redis_task_status():
    """
    Migrate existing code from in-memory TASK_STATUS to Redis-based system
    This function helps with the transition by providing compatibility
    """
    
    print("🚀 Migrating to Redis-based task status system...")
    
    # Import both old and new systems
    try:
        from newsies_common.task_status import TASK_STATUS as OLD_TASK_STATUS
        from newsies_common.redis_task_status import TASK_STATUS as NEW_TASK_STATUS
        
        print("✅ Successfully imported both task status systems")
        
        # If there are any existing tasks in the old system, migrate them
        if hasattr(OLD_TASK_STATUS, 'items') and len(OLD_TASK_STATUS) > 0:
            print(f"📦 Migrating {len(OLD_TASK_STATUS)} existing tasks...")
            
            for task_id, task_data in OLD_TASK_STATUS.items():
                NEW_TASK_STATUS[task_id] = task_data
                print(f"   ✅ Migrated task: {task_id}")
            
            print("🎉 Task migration completed!")
        else:
            print("📝 No existing tasks to migrate")
            
        return NEW_TASK_STATUS
        
    except ImportError as e:
        print(f"⚠️  Import error during migration: {e}")
        print("   This is expected if Redis is not available")
        
        # Fall back to old system if Redis is not available
        try:
            from newsies_common.task_status import TASK_STATUS
            print("📋 Falling back to in-memory task status system")
            return TASK_STATUS
        except ImportError:
            print("❌ Could not import any task status system")
            raise
    
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        raise


def get_compatible_task_status():
    """
    Get task status system with automatic fallback
    Tries Redis first, falls back to in-memory if needed
    """
    try:
        from newsies_common.redis_task_status import TASK_STATUS
        return TASK_STATUS
    except Exception:
        # Fall back to old system
        from newsies_common.task_status import TASK_STATUS
        return TASK_STATUS


if __name__ == "__main__":
    """Run migration when called directly"""
    migrate_to_redis_task_status()
