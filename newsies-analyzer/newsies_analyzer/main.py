"""
Main entry point for newsies-analyzer service
"""

import sys
import uuid
import os
import pwd
from .pipelines import analyze_pipeline


def main():
    """Main entry point for analyzer service"""
    print("Newsies Analyzer Service")
    
    if len(sys.argv) < 2:
        print("Usage: python -m newsies_analyzer.main <command> [archive_date]")
        print("Commands:")
        print("  analyze [YYYY-MM-DD] - Run the content analysis pipeline")
        return
    
    command = sys.argv[1]
    task_id = str(uuid.uuid4())
    user_id = pwd.getpwuid(os.getuid())[0]
    
    if command == "analyze":
        archive_date = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Starting analyze pipeline (task_id: {task_id}, archive: {archive_date})")
        analyze_pipeline(task_id=task_id, archive=archive_date)
        print("Analyze pipeline completed")
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
