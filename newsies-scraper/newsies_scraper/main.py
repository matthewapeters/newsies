"""
Main entry point for newsies-scraper service
"""

import sys
import uuid
import os
import pwd
from .pipelines import get_articles_pipeline


def main():
    """Main entry point for scraper service"""
    print("Newsies Scraper Service")
    
    if len(sys.argv) < 2:
        print("Usage: python -m newsies_scraper.main <command>")
        print("Commands:")
        print("  get-articles - Run the article scraping pipeline")
        return
    
    command = sys.argv[1]
    task_id = str(uuid.uuid4())
    user_id = pwd.getpwuid(os.getuid())[0]
    
    if command == "get-articles":
        print(f"Starting get-articles pipeline (task_id: {task_id})")
        get_articles_pipeline(task_id=task_id)
        print("Get-articles pipeline completed")
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
