from src.backend.pdf_ingestor import PDFIngestor
from src.utils.config import config
from src.utils.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langsmith import traceable


# Define the Pydantic models for Structured ouptut

# For Text
class TextSummary(BaseModel):
    """Structured ouptut for text chunks."""
    summary: str= Field(..., description="Concise, searchable summary of the text.")

# For Tables
class TableSummary(BaseModel):
    """Structured ouptut for tables."""
    summary: str= Field(..., description="Concise, searchable summary of the table")

# For Images
class ImageSummary(BaseModel):
    """Structured ouptut for Images"""
    summary: str= Field(..., description="Concise description  of the image content")


# Summarizer class 
class Summarizer:

    def __init__(self):
   
        # Initialize the LLM
        self.llm = init_chat_model(config.LLM_MODEL, model_provider= config.LLM_PROVIDER)
    

    @traceable(name='summarize_text_chunks', tags=['summarization', 'text'])
    def summarize_texts(self, texts: List[str]):
        """ Summarize the text chunks. """

        try:

            logger.debug("Summarizing text chunks...")

            # Define the LLM with structured output
            structured_llm= self.llm.with_structured_output(TextSummary)

            # Define the prompt template
            prompt= ChatPromptTemplate.from_template(

                """ You are an expert document analyst creating summaries for semantic search and retrieval.

                    Analyze the following text chunk and create a concise summary that captures:
                    - **Key concepts, entities, and technical terms** with their context
                    - **Main facts, findings, or conclusions** presented
                    - **Quantitative data** (numbers, percentages, measurements, dates)
                    - **Relationships** between different concepts or entities
                    - **Action items, processes, or methodologies** mentioned

                    Focus on making the summary **searchable** - someone should be able to find this content based on your summary when asking related questions.

                    Text to summarize: 
                    {input_text}

                    Create a comprehensive yet concise summary that preserves the core meaning and searchable elements.
                
                """
            )

            # Define the chain
            chain= prompt | structured_llm
            
            # Define the list to store text summaries
            summaries=[]

            for chunk in texts:
                result=chain.invoke({"input_text": chunk})
                summaries.append(result.summary)

            logger.info("Completed summarizing texts.")
            
            return summaries


        except Exception as e:
            logger.error(f"Error summarizing texts: {e}")
            return []
            
          

    @traceable(name='summarize_table', tags=['summarzation', 'tables'])
    def summarize_tables(self, tables: List[str]):
        """ Summarize tables into structured summaries."""

        try:
            
            logger.debug("Summarizing tables...")

            # Define the LLM with structured output
            structured_llm= self.llm.with_structured_output(TableSummary)

            # Define the prompt
            prompt=ChatPromptTemplate.from_template(
                """ You are analyzing a data table for semantic search and retrieval purposes.

                    Create a structured summary that captures:
                    - **Table purpose**: What does this table show or compare?
                    - **Data structure**: Column headers, row categories, and data types
                    - **Key insights**: Notable patterns, trends, or outliers in the data
                    - **Quantitative highlights**: Important numbers, ranges, or calculations
                    - **Comparative elements**: How different rows/columns relate or compare
                    - **Context clues**: Any indicators of time periods, units, or scope

                    Make the summary detailed enough that someone could determine if this table contains relevant data for their query.

                    Table content:
                    {input_table}

                    Provide a comprehensive summary that captures both the structure and insights of this table.
                 
                 """

            )

            # Define the chain
            chain= prompt | structured_llm

            # Define a list to store the table summaries
            summaries=[]

            for tbl in tables:
                table_content = tbl.metadata.text_as_html if hasattr(tbl.metadata, 'text_as_html') and tbl.metadata.text_as_html else str(tbl)
                result = chain.invoke({"input_table": table_content})
                summaries.append(result.summary)

            logger.info("Completed summarizing tables.")  

            return summaries
        
        except Exception as e:
            logger.error(f"Error summarizing tables: {e}")
            return []
    

    @traceable(name='summarize_images', tags=['summarization', 'images'])
    def summarize_images(self, images: List[str]):
        """ Summarize images into structured summaries."""

        try:
            logger.debug("Summarizing images...")

            # Define the LLM with structured output
            structured_llm = self.llm.with_structured_output(ImageSummary)

            # Define the prompt
            prompt = ChatPromptTemplate.from_messages([
                    ("user", [
                        {"type": "text", "text": 
                        
                        """ You are analyzing an image from a document. Create a detailed description for semantic search and retrieval.

                            Describe comprehensively:
                            - **Visual elements**: Charts, graphs, diagrams, photographs, illustrations
                            - **Text content**: All visible text, labels, captions, titles, axis labels
                            - **Data visualization**: If it's a chart/graph, describe the data, trends, and key points
                            - **Spatial relationships**: Layout, positioning, and how elements relate
                            - **Technical details**: Equipment, processes, or concepts shown
                            - **Context indicators**: Any clues about the document's subject matter

                            Focus on creating a description that would help someone find this image when searching for related concepts, data, or visual information.

                            Analyze this image thoroughly:"""
                        },

                        {"type": "image_url", "image_url": {"url": "{image_url}"}},
                    ])
                ])

            # Define the chain
            chain = prompt | structured_llm

            # Define the list to store image summaries
            summaries= []

            for img_b64 in images:
                image_url= f"data:image/jpeg;base64,{img_b64}"
                result = chain.invoke({"image_url": image_url})
                summaries.append(result.summary)

            logger.info("Completed summarizing images.")
            
            return summaries


        except Exception as e:
            logger.error(f"Error summarizing images: {e}")
            return []



    @traceable(name='summarization_full_pipeline', tags=['summarization', 'main'])
    def summarize_all(self, pdf_path: str):
        """ Run the full summarization pipeline (texts, tables, images). """

        try:
            logger.info("Processing pdf and extracting elements...")
            ingestor = PDFIngestor(pdf_path)  # Create an object of the PDFIngestor class
            elements = ingestor.process_pdf()  # Store the extracted elements : Texts, Tables , Images

            # Convert functions into runnables 
            text_runnable = RunnableLambda(
                lambda x: {**x, "text_summaries": self.summarize_texts(x["text"])}
            )

            table_runnable = RunnableLambda(
                lambda x: {**x, "table_summaries": self.summarize_tables(x["tables"])}
            )

            image_runnable = RunnableLambda(
                lambda x: {**x, "image_summaries": self.summarize_images(x["images_base64"])}
            )


            # Compose into sequential pipeline 
            pipeline: RunnableSequence = text_runnable | table_runnable | image_runnable

            # Run pipeline
            results = pipeline.invoke(elements)

            logger.info("Completed summarizing all elements.")
            logger.debug(
                    "Summary counts — text:%s  tables:%s  images:%s",
                    len(results.get("text_summaries", [])),
                    len(results.get("table_summaries", [])),
                    len(results.get("image_summaries", [])),
                )
                
            return results  # returns the elements and their summaries

        except Exception as e:
            logger.error(f"Error in summarize_all: {e}")
            return {}




    
   

