"""
newsies-common
Shared utilities, data structures, and visitor pattern implementation
"""

__version__ = "0.2.0"

# Re-export commonly used items
# Note: Avoiding star imports to prevent circular dependencies

__all__ = [
    "document_structures",
    "visitor", 
    "utils",
    "task_status",
    "redis_task_status",
    "migrate_task_status",
    "targets",
]
