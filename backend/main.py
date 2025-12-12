from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.rag import chat, embedding, retrieval
from backend.api.auth import auth
from backend.api.utils import validation
from backend.utils.error_handlers import setup_error_handlers
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="API for the interactive textbook on Physical AI & Humanoid Robotics",
    version="0.1.0",
)

# Setup error handlers
setup_error_handlers(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router, prefix="/api/rag", tags=["rag-chat"])
app.include_router(embedding.router, prefix="/api/rag", tags=["rag-embedding"])
app.include_router(retrieval.router, prefix="/api/rag", tags=["rag-retrieval"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(validation.router, prefix="/api/utils", tags=["utils"])

@app.get("/")
def read_root():
    return {"message": "Physical AI & Humanoid Robotics Textbook API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "textbook-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )