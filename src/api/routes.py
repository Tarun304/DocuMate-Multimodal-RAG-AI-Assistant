from src.utils.logger import logger
from src.backend.indexing import Indexer, BUILT_RETRIEVER
from src.backend.qa_pipeline import QAPipeline
from src.api.models import PDFPathRequest, QuestionRequest, QuestionResponse, IndexBuildResponse, HealthResponse, ResetRequest, ResetResponse
from fastapi import APIRouter, HTTPException
from langsmith import traceable

from langgraph.checkpoint.memory import InMemorySaver
import uuid

# Create the router instance
router = APIRouter()

# Initialize the components
indexer = Indexer()  # Object of Indexer Class
global_checkpointer= InMemorySaver()



# Define the health check endpoint
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return HealthResponse(
        status="Healthy",
        message="Multimodal RAG API is running"
    )



# Define the index building endpoint
@router.post("/build-index", response_model=IndexBuildResponse)
@traceable(name="api_build_index", tags=["api", "indexing"])   # Langsmith tracing
async def build_index(request: PDFPathRequest):
    """Build index from PDF file path."""
    try:
        logger.info(f"Index building requested for file: {request.pdf_path}")
        
        # Basic input validation
        if not request.pdf_path.strip():
            raise HTTPException(status_code=400, detail="PDF path cannot be empty")
        
        if not request.pdf_path.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Build the index
        retriever = indexer.build_index(request.pdf_path)
        
        if retriever is None:
            raise HTTPException(status_code=500, detail="Failed to build index")
        
        logger.info("Index built successfully")
        return IndexBuildResponse(
            status="completed",
            message=f"Index built successfully from {request.pdf_path}",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in build_index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# Define the question answering endpoint
@router.post("/ask-question", response_model=QuestionResponse)
@traceable(name="api_ask_question", tags=["api", "qa"])  # Langsmith tracing
async def ask_question(request: QuestionRequest):
    """Main endpoint for question answering."""
    try:
        logger.info("Question answering requested")
        
        # Basic input validation
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        #Use provided thread_id or generate new one
        thread_id= request.thread_id or str(uuid.uuid4())

        # Create QAPipeline with thread_id and shared checkpointer
        qa_pipeline= QAPipeline(thread_id= thread_id, checkpointer= global_checkpointer)
        
        # Process the request through the QA pipeline
        result = qa_pipeline.answer_question(request.question)
        
        # Return the response
        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            context=result["context_for_ui"],
            success=True,
            thread_id= thread_id 
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# Define the reset endpoint to delete the Index
@router.post("/reset", response_model=ResetResponse)
async def reset_index(request:ResetRequest):
    """Forget every previously processed document and clear the embeddings database."""
    try:
        from src.backend import indexing
        
        # Clear Chroma collection data
        if indexing.BUILT_RETRIEVER is not None:
            indexing.BUILT_RETRIEVER.vectorstore.delete_collection()

            logger.info("Chroma collection deleted successfully")
        else:
            logger.info("No retriever to clear - already empty.")
        
        # Clear Python object reference
        indexing.BUILT_RETRIEVER = None

        logger.info("BUILT_RETRIEVER reset to None")

        # Note: We don't need to clear thread memory since frontend generates new thread_id
        # The old thread_id will simply be abandoned and garbage collected eventually
        logger.info(f"Reset completed for thread_id: {request.thread_id}")

        return ResetResponse(success=True, message="Index cleared, new session will use fresh thread_id")

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        # Fallback: clear the reference to prevent crashes
        from src.backend import indexing
        indexing.BUILT_RETRIEVER = None
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")