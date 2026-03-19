import re
import logging
import base64
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
from PIL import Image

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI 

class ContentProcessor:
    """
    Processes the parsed content - summarizes images, creates llm based semantic chunks
    """
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model 
        self.chunker_model = config.rag.chunker_model 
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """Summarize images using Gemini Vision."""
        prompt_template = """Describe the image in detail while keeping it concise... 
                             (Keep your original prompt here)"""

        results = []
        for image_path in images:
            try:
                # 1. Encode image to Base64 (Required for Gemini Cloud)
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

                # 2. Prepare Gemini multi-modal message
       
                message = [
                    (
                        "user",
                        [
                            {"type": "text", "text": prompt_template},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                            },
                        ],
                    )
                ]
                
                # 3. Invoke Gemini
                res = self.summarizer_model.invoke(message)
                results.append(res.content)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                results.append("no image summary")
        
        return results
    
  
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        IMAGE_PLACEHOLDER = ""
        PAGE_BREAK_PLACEHOLDER = ""
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        return self._replace_occurrences(formatted_parsed_document, IMAGE_PLACEHOLDER, image_summaries)
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(target, f'picture_counter_{counter} ' + replacement, 1)
                else:
                    result = result.replace(target, '', 1)
            else:
                break
        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        # Split by section boundaries
        SPLIT_PATTERN = "\n#"
        chunks = formatted_document.split(SPLIT_PATTERN)
        
        chunked_text = ""
        for i, chunk in enumerate(chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"
            chunked_text += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"
        
        # LLM-based semantic chunking prompt 
        CHUNKING_PROMPT = "... (Keep your original CHUNKING_PROMPT here) ..."
        
        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)
        chunking_response = self.chunker_model.invoke(formatted_chunking_prompt).content
        
        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        split_after = [] 
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()] 

        if not split_after:
            return [chunked_text]

        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        sections = []
        current_section = [] 
        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = [] 
        
        if current_section:
            sections.append("".join(current_section).strip())

        return sections