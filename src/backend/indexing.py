import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from src.backend.summarizer import Summarizer
from src.utils.logger import logger
from typing import Optional
from langsmith import traceable

# Simple in-process cache
BUILT_RETRIEVER: Optional[MultiVectorRetriever] = None


# Indexer Class
class Indexer:

    def __init__(self, collection_name: str = "multi_modal_rag"):
        """
        Initializes vectorstore, docstore, and retriever.
        """
        # Initialise the Embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # Vectorstore to store summaries
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

        # Docstore to store original docs
        self.docstore = InMemoryStore()
        self.id_key = "doc_id"

        # Retriever object 
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
        )

    
    
    @traceable(name='index_multimodal_documents', tags=['indexing', 'vectorstore'])
    def index_documents(self, texts, text_summaries, tables, table_summaries, images, image_summaries):
        """Index texts, tables, and images with summaries linked to originals."""
        
        try:

            # Texts
            if texts and text_summaries:
                    logger.debug("Indexing texts...")
                    text_ids = [str(uuid.uuid4()) for _ in texts]
                    
                    # Vectorstore: summaries with metadata
                    summary_texts = [
                        Document(page_content=summary, metadata={self.id_key: text_ids[i], "kind": "text"})
                        for i, summary in enumerate(text_summaries)
                    ]
                    self.vectorstore.add_documents(summary_texts)
                    
                    # Docstore: original texts as Document objects with metadata
                    original_text_docs = [
                        Document(page_content=text, metadata={"kind": "text"})
                        for text in texts
                    ]
                    self.docstore.mset(list(zip(text_ids, original_text_docs)))
                    logger.info("Completed indexing texts.")
            else:
                    logger.debug("No texts to index.")


            # Tables  
            if tables and table_summaries:
                    logger.debug("Indexing tables...")
                    table_ids = [str(uuid.uuid4()) for _ in tables]
                    
                    summary_tables = [
                        Document(page_content=summary, metadata={self.id_key: table_ids[i], "kind": "table"})
                        for i, summary in enumerate(table_summaries)
                    ]
                    self.vectorstore.add_documents(summary_tables)
                    
                    # Docstore: original tables as Document objects with metadata
                    original_table_docs = [
                        Document(
                            page_content=table.metadata.text_as_html if hasattr(table.metadata, 'text_as_html') and table.metadata.text_as_html else str(table),
                            metadata={
                                "kind": "table",
                                "page_number": table.metadata.page_number
                            }
                        )
                        for table in tables
                    ]
                    self.docstore.mset(list(zip(table_ids, original_table_docs)))
                    logger.info("Completed indexing tables.")
                
            else:
                    logger.debug("No tables to index.")


            # Images
            if images and image_summaries:
                    logger.debug("Indexing images...")
                    img_ids = [str(uuid.uuid4()) for _ in images]
                    
                    summary_images = [
                        Document(page_content=summary, metadata={self.id_key: img_ids[i], "kind": "image"})
                        for i, summary in enumerate(image_summaries)
                    ]
                    self.vectorstore.add_documents(summary_images)
                    
                    # Docstore: original images as Document objects with metadata
                    original_image_docs = [
                        Document(page_content=img_b64, metadata={"kind": "image"})
                        for img_b64 in images
                    ]
                    self.docstore.mset(list(zip(img_ids, original_image_docs)))
                    logger.info("Completed indexing images.")

            else:
                logger.debug("No images to index.")

            logger.info("Completed Indexing.")


        except Exception:
            logger.exception("Indexing Failed.")
            raise 
            


    @traceable(name='build_index_full_pipleine', tags=['indexing', 'main'])
    def build_index(self, pdf_path):
        """Run the pipeline: Summarize and index documents and push it into RAM cache"""
        
        summarizer = Summarizer()  # Create the object of Summarizer class
        results = summarizer.summarize_all(pdf_path)  # Store the elements and their summaries

        # Index the elements and their summaries
        self.index_documents(
            texts=results.get("text", []),
            text_summaries=results.get("text_summaries", []),
            tables=results.get("tables", []),
            table_summaries=results.get("table_summaries", []),
            images=results.get("images_base64", []),
            image_summaries=results.get("image_summaries", []),
        )

        global BUILT_RETRIEVER   
        BUILT_RETRIEVER=self.retriever   # Store the retriever object

        return self.retriever
