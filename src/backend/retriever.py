from typing import List, Any
from src.utils.logger import logger
from langsmith import traceable

class Retriever:
  

    @traceable(name='retrieve_documents', tags=['retrieval', 'search'])
    def retrieve(self, question: str, top_k: int=5):
        """ 
        Retrieve the top-k relevant documents as per the question.
        """
        # Import the global variable at runtime, not module level
        from src.backend.indexing import BUILT_RETRIEVER

        if BUILT_RETRIEVER is None:
            logger.error("Index not built yet - call /build_index first")
            return []

        try:
            logger.debug(f"Fetching top {top_k} docs for: {question!r}")
            return BUILT_RETRIEVER.invoke(question)[:top_k]  # Return the documents retrieved 
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []