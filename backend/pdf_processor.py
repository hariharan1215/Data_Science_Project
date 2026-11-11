"""
CareerMate - PDF Document Processor
Extracts, validates, and chunks text from uploaded career PDFs.
Uses PyMuPDF (fitz) with transparent fallback to pypdf for resilient cross-platform compatibility.
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from backend.config import Config
from backend.domain_validator import DomainValidator

logger = logging.getLogger(__name__)

# Check PyMuPDF availability
HAS_PYMUPDF = False
try:
    import fitz
    HAS_PYMUPDF = True
    logger.info("PyMuPDF (fitz) engine initialized.")
except Exception as e:
    logger.warning(f"PyMuPDF could not be loaded ({e}). Pure-python pypdf engine will be used.")

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

class PDFProcessor:
    """
    Handles PDF validation, text extraction (PyMuPDF / pypdf), scanned PDF detection,
    career-domain document verification, and semantic text chunking.
    """

    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        """Checks if file has an allowed extension."""
        return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @classmethod
    def extract_text_by_pages(cls, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text from each page using PyMuPDF (or pypdf).
        Returns a list of dicts: [{'page': 1, 'text': '...', 'word_count': ...}]
        Raises PDFProcessingError if the PDF is corrupt or scanned/empty.
        """
        if not file_path.exists():
            raise PDFProcessingError("Uploaded file was not found on server.")

        pages_data = []
        total_chars = 0
        total_words = 0

        # Attempt extraction using PyMuPDF
        if HAS_PYMUPDF:
            try:
                doc = fitz.open(str(file_path))
                if len(doc) == 0:
                    doc.close()
                    raise PDFProcessingError("The uploaded PDF has 0 pages.")

                for page_index in range(len(doc)):
                    page = doc[page_index]
                    text = page.get_text("text") or ""
                    cleaned_text = re.sub(r"[ \t]+", " ", text).strip()
                    words = len(cleaned_text.split())
                    total_chars += len(cleaned_text)
                    total_words += words

                    pages_data.append({
                        "page": page_index + 1,
                        "text": cleaned_text,
                        "word_count": words,
                        "char_count": len(cleaned_text)
                    })
                doc.close()
            except PDFProcessingError:
                raise
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}. Attempting pypdf fallback.")
                pages_data = []

        # If PyMuPDF wasn't available or failed, use pypdf
        if not pages_data:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                if len(reader.pages) == 0:
                    raise PDFProcessingError("The uploaded PDF has 0 pages.")

                total_chars = 0
                total_words = 0
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    cleaned_text = re.sub(r"[ \t]+", " ", text).strip()
                    words = len(cleaned_text.split())
                    total_chars += len(cleaned_text)
                    total_words += words

                    pages_data.append({
                        "page": idx + 1,
                        "text": cleaned_text,
                        "word_count": words,
                        "char_count": len(cleaned_text)
                    })
            except PDFProcessingError:
                raise
            except Exception as e:
                raise PDFProcessingError(f"Could not open or parse PDF file: {str(e)}")

        # Scanned PDF detection
        if total_chars < 50 or total_words < 15:
            raise PDFProcessingError(
                "The uploaded PDF appears to be a scanned document or contains only images. "
                "CareerMate requires a searchable text-based PDF to extract career and educational information."
            )

        # Career domain validation of PDF content
        combined_text_sample = " ".join(p["text"] for p in pages_data[:3])
        if not cls.validate_pdf_domain(combined_text_sample):
            raise PDFProcessingError(
                "The uploaded document does not appear to be related to careers, education, "
                "resumes, or technical skills. CareerMate only processes career-relevant documents."
            )

        return pages_data

    @classmethod
    def validate_pdf_domain(cls, text: str) -> bool:
        """
        Verifies that the PDF text contains indicators of career, education,
        skills, resume, interview, or technical domains.
        """
        tokens = set(re.findall(r"\b[a-z0-9+#.-]+\b", text.lower()))
        matched_keywords = tokens & DomainValidator.CAREER_KEYWORDS
        # If at least 3 career-relevant keywords appear in the document sample
        return len(matched_keywords) >= 3

    @classmethod
    def chunk_document(
        cls,
        pages_data: List[Dict[str, Any]],
        filename: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Splits text from pages into overlapping chunks with metadata.
        Preserves page reference and source filename.
        """
        chunk_size = chunk_size or Config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        chunks = []
        global_chunk_id = 0

        for page_info in pages_data:
            page_num = page_info["page"]
            text = page_info["text"]

            if not text:
                continue

            # Split text by paragraphs or double newlines to respect structure
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text]

            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) + 1 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{para}".strip()
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_id": f"{filename}_p{page_num}_c{global_chunk_id}",
                            "source": filename,
                            "page": page_num,
                            "text": current_chunk
                        })
                        global_chunk_id += 1

                    if len(para) > chunk_size:
                        start = 0
                        while start < len(para):
                            end = start + chunk_size
                            slice_text = para[start:end]
                            chunks.append({
                                "chunk_id": f"{filename}_p{page_num}_c{global_chunk_id}",
                                "source": filename,
                                "page": page_num,
                                "text": slice_text
                            })
                            global_chunk_id += 1
                            start += (chunk_size - chunk_overlap)
                        current_chunk = ""
                    else:
                        overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                        current_chunk = f"{overlap_text}\n\n{para}".strip()

            if current_chunk:
                chunks.append({
                    "chunk_id": f"{filename}_p{page_num}_c{global_chunk_id}",
                    "source": filename,
                    "page": page_num,
                    "text": current_chunk
                })
                global_chunk_id += 1

        return chunks
