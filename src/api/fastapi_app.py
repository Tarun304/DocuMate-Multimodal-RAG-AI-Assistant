from src.api.routes import router
from fastapi import FastAPI

# Create the FastAPI app instance
app = FastAPI(
    title="Multimodal RAG API",
    description="AI-powered multimodal document question answering system",
    docs_url="/docs",
    
)

# Include the router
app.include_router(router, prefix='/api')


