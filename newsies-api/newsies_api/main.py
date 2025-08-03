"""
Main entry point for newsies-api service
"""

import uvicorn
from .api.app import app


def main():
    """Main entry point for API service"""
    print("Starting Newsies API Service")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
