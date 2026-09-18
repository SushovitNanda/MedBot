"""
Start the MedRAG FastAPI backend.

From the MedRAG repo root:
    python run.py
"""

import uvicorn

from config.settings import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
