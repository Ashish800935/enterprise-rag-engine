from typing import List
from src.config import settings

def recursive_character_chunking(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[str]:
    """
    Splits text into chunks respecting natural sentence/paragraph boundaries.
    
    Why Chunk Overlap Matters:
    If a key thought or sentence is cut exactly at the chunk boundary (e.g., chunk 1 ends with
    'The server crashed because...' and chunk 2 starts with '...the memory ran out'), neither chunk
    contains the complete context. Overlapping by 50-100 characters bridges this gap!
    """
    if not text or not text.strip():
        return []

    # Clean whitespace while preserving paragraphs
    cleaned_text = " ".join(text.split())
    chunks: List[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size
        
        # If we haven't reached the end, look for a natural boundary (period, newline, space)
        if end < text_length:
            boundary = -1
            # Search backwards for sentence end
            for punct in [". ", "? ", "! ", "\n"]:
                pos = cleaned_text.rfind(punct, start, end)
                if pos != -1 and pos > boundary:
                    boundary = pos + len(punct)
            
            # If no sentence boundary found, break on word boundary (space)
            if boundary == -1:
                space_pos = cleaned_text.rfind(" ", start, end)
                if space_pos != -1:
                    boundary = space_pos + 1
            
            if boundary != -1 and boundary > start:
                end = boundary

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - chunk_overlap)
        start = end - chunk_overlap
        if start >= end:
            start = end

    return chunks
