from pydantic import BaseModel, Field
from typing import Dict, List , Any, Optional


# Define the Request Schemas
class PDFPathRequest(BaseModel):
    """Request model for PDF file path."""
    pdf_path: str = Field(..., description="Path to the PDF file to be indexed")

class QuestionRequest(BaseModel):
    """ Request model for question answering."""
    question: str = Field(..., description= "User's question to be answered")
    thread_id: Optional[str]= None  # Added 

class ResetRequest(BaseModel):
    thread_id: str = Field(..., description="Thread memory to be cleared")



# Define the Response Schemas
class IndexBuildResponse(BaseModel):
    """ Response model for index building. """
    status: str = Field(..., description= "Status of index building")
    message: str= Field(..., description= "Details about index building")
    success: bool = Field (..., description= "Whether the operation was successful or not.")

class QuestionResponse(BaseModel):
    """ Response model for question answering. """
    question: str = Field(..., description="The original question asked")
    answer: str = Field(..., description="AI-generated answer")
    context: Dict[str, List[Any]] = Field(..., description="Retrieved context (texts, tables, images)")
    success: bool = Field(..., description="Whether the operation was successful")
    thread_id: str # Added

class ResetResponse(BaseModel):
    """Response model for reset endpoint."""
    success: bool = Field(..., description="Whether the reset succeeded")
    message: str = Field(..., description="Status message")



# Health check Schema
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API Status")
    message: str = Field(..., description="Health check message")
