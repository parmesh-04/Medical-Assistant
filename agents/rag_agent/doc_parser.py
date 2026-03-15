import os
import logging
from pathlib import Path
from typing import List, Tuple, Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

class MedicalDocParser:
    """
    Handles parsing of medical research documents using Docling.
    It extracts text, tables, and images so the AI can 'see' the whole paper.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Medical Document Parser initialized!")

    def parse_document(
            self,
            document_path: str,
            output_dir: str,
            image_resolution_scale: float = 2.0,
            do_ocr: bool = True,
            do_tables: bool = True,
            do_formulas: bool = True,
            do_picture_desc: bool = False
        ) -> Tuple[Any, List[str]]:
        
        # 1. Ensure the output directory exists for extracted images
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 2. Configure Docling's 'Advanced' parsing options
        pipeline_options = PdfPipelineOptions(
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=image_resolution_scale,
            do_ocr=do_ocr,
            do_table_structure=do_tables,
            do_formula_enrichment=do_formulas,
            do_picture_description=do_picture_desc
        )
        
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        # 3. Initialize the Converter
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        # 4. Convert PDF to structured Docling document
        conversion_res = converter.convert(document_path)
        doc_filename = conversion_res.input.file.stem
        
        image_paths = []
        picture_counter = 0

        # 5. Iterate through the document to find and save Pictures
        for element, _level in conversion_res.document.iterate_items():
            # In this version, we focus on Pictures for the Image Analysis Agent
            if isinstance(element, PictureItem):
                picture_filename = f"{doc_filename}-picture-{picture_counter}.png"
                element_image_path = output_dir_path / picture_filename
                
               
                with element_image_path.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")
                
                image_paths.append(str(element_image_path))
                picture_counter += 1
        
        
        return conversion_res.document, image_paths