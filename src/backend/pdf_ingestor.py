from pathlib import Path
from typing import Any, List, Dict
from pydantic import BaseModel
from src.utils.logger import logger

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    CompositeElement,
    Table,
    Image,
    Text,
)
from langsmith import traceable


class Element(BaseModel):
    """Uniform element model """
    type: str
    text: Any

# PDF Ingestor Class
class PDFIngestor:
    
    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = Path(pdf_path) # to store the pdf path 
        self.chunks: List[Any] = []  # CompositeElement and/or others
        self.text_chunks: List[str] = []  # to store the text chunks
        self.tables: List[Table] = [] # to store the tables
        self.images_b64: List[str] = []  # to store the columns

    

    @traceable(name='load_pdf_elements', tags=['pdf_processing', 'extraction'])
    def load_elements(self) -> List[Any]:
        """Partition the PDF  (hi_res + by_title + base64 images)."""
        logger.debug("Loading PDF and extracting elements...")

        self.chunks = partition_pdf(
            filename=str(self.pdf_path),
            # --- table detection ---
            infer_table_structure=True,          # Required to get structured tables (HTML/MD in metadata)
            strategy="hi_res",                   # Mandatory for table inference

            # --- image extraction ---
            extract_image_block_types=["Image", "Table"],  # also rasterize tables if you want them as images
            extract_image_block_to_payload=True,           # put base64 in element metadata (API-friendly)
            # image_output_dir_path=str(self.output_path), # leave commented to keep base64 in-memory

            # --- chunking ---
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )

        logger.info(f"Successfully extracted {len(self.chunks)} chunks from {self.pdf_path.name}")

        # Debug: show high-level types
        type_set = {str(type(el)) for el in self.chunks}
        logger.debug(f"Chunk types observed: {type_set}")

        # Preview a few
        for i, ch in enumerate(self.chunks[:5]):
            preview = str(ch)[:100].replace("\n", " ")
            logger.debug(f"[Chunk {i}] {type(ch).__name__}: '{preview}...'")

        return self.chunks



    def _iter_orig_elements(self, chunk: Any):
        """Safely iterate nested orig_elements for CompositeElement chunks."""
        try:
            orig = getattr(chunk.metadata, "orig_elements", None)
        except Exception:
            orig = None
        if not orig:
            return []
        return orig



    @traceable(name='separate_text_and_tables', tags=['pdf_processing', 'separation'])
    def separate_elements(self) -> Dict[str, Any]:
        """
        Separate into:
          - text_chunks: list[str]
          - tables: list[Table] (found top-level or nested in orig_elements)
        """
        texts: List[str] = []
        tables: List[Table] = []

        for chunk in self.chunks:
            # 1) Collect text chunks
            if isinstance(chunk, CompositeElement):
                text_parts: List[str] = []

                # 2) Look inside nested orig_elements for tables (critical fix)
                for el in self._iter_orig_elements(chunk):
                    if isinstance(el, Table):
                        tables.append(el)
                        # DON'T add table to text_parts
                    else:
                        text_parts.append(str(el))

                
                # Add text parts if any exist
                if text_parts:
                    texts.append("\n".join(text_parts))
                elif not list(self._iter_orig_elements(chunk)):
                    # Fallback: if no orig_elements, add the chunk as text
                    texts.append(str(chunk))

            # 3) (Rare with by_title) If Unstructured returned top-level Table elements
            elif isinstance(chunk, Table):
                tables.append(chunk)

        self.text_chunks = texts
        self.tables = tables

        logger.info(f"Separated {len(self.text_chunks)} text chunks and {len(self.tables)} table(s)")
        return {"text": self.text_chunks, "tables": self.tables}



    @traceable(name='extract_images_base64', tags=['pdf_processing', 'images'])
    def get_images_base64(self) -> List[str]:
        """
        Collect base64 images from nested orig_elements.
        
        """
        images_b64: List[str] = []

        for chunk in self.chunks:
            if isinstance(chunk, CompositeElement):
                for el in self._iter_orig_elements(chunk):
                    if isinstance(el, Image):
                        # Available because extract_image_block_to_payload=True
                        b64 = getattr(getattr(el, "metadata", None), "image_base64", None)
                        if b64:
                            images_b64.append(b64)

            # (Rare) If Unstructured returns top-level Image
            elif isinstance(chunk, Image):
                b64 = getattr(getattr(chunk, "metadata", None), "image_base64", None)
                if b64:
                    images_b64.append(b64)

        self.images_b64 = images_b64
        logger.info(f"Extracted {len(self.images_b64)} base64 image(s)")
        return self.images_b64


    
    @traceable(name='process_pdf_full_pipeline', tags=['pdf_processing', 'main'])
    def process_pdf(self) -> Dict[str, Any]:
        """Full pipeline: load → separate → collect images """
        self.load_elements()
        separated = self.separate_elements()
        images = self.get_images_base64()

        # Debug: if we “know” there’s 1 table, log a short preview
        if self.tables:
            tprev = str(self.tables[0])[:200].replace("\n", " ")
            logger.debug(f"[Table preview] {tprev}...")

        return {
            "text": separated["text"],        # list[str]
            "tables": separated["tables"],    # list[Table]
            "images_base64": images,          # list[str]
        }

