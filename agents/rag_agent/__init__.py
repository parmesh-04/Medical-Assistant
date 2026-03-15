import os
import time
import logging
from typing import List, Optional, Dict, Any

# These imports refer to files we will build next in the same folder
from .doc_parser import MedicalDocParser
from .content_processor import ContentProcessor
from .vectorstore_qdrant import VectorStore
from .reranker import Reranker
from .query_expander import QueryExpander
from .response_generator import ResponseGenerator

class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation system that integrates all components.
    """
    def __init__(self, config):
        """
        Initialize the RAG Agent using the Gemini-powered config.
        """
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Medical RAG system with Gemini")
        self.config = config
        
        # Initializing sub-components
        self.doc_parser = MedicalDocParser()
        self.content_processor = ContentProcessor(config)
        self.vector_store = VectorStore(config)
        self.reranker = Reranker(config)
        self.query_expander = QueryExpander(config)
        self.response_generator = ResponseGenerator(config)
        self.parsed_content_dir = self.config.rag.parsed_content_dir
    
   
    
    def ingest_file(self, document_path: str) -> Dict[str, Any]:
        """Ingest a single file using the 5-step pipeline."""
        start_time = time.time()
        try:
            # 1. Parse -> 2. Summarize -> 3. Format -> 4. Chunk -> 5. Store
            parsed_document, images = self.doc_parser.parse_document(document_path, self.parsed_content_dir)
            image_summaries = self.content_processor.summarize_images(images)
            formatted_document = self.content_processor.format_document_with_images(parsed_document, image_summaries)
            document_chunks = self.content_processor.chunk_document(formatted_document)
            
            self.vector_store.create_vectorstore(document_chunks=document_chunks, document_path=document_path)
            
            return {"success": True, "chunks_processed": len(document_chunks), "processing_time": time.time() - start_time}
        except Exception as e:
            return {"success": False, "error": str(e)}
        

    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """Ingest all files in a directory."""
        start_time = time.time()
        self.logger.info(f"Ingesting files from directory: {directory_path}")
        
        try:
            if not os.path.isdir(directory_path):
                raise ValueError(f"Directory not found: {directory_path}")
            
            files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                     if os.path.isfile(os.path.join(directory_path, f))]
            
            total_chunks_processed = 0
            successful_ingestions = 0
            
            for file_path in files:
                result = self.ingest_file(file_path)
                if result["success"]:
                    successful_ingestions += 1
                    total_chunks_processed += result.get("chunks_processed", 0)
            
            return {
                "success": True,
                "documents_ingested": successful_ingestions,
                "chunks_processed": total_chunks_processed,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}