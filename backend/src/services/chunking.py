from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

def recursive_character_chunking(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[str]:
    """
    Splits text into semantically cohesive chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Why Recursive Splitting Matters:
    LangChain's RecursiveCharacterTextSplitter attempts to split on paragraphs ('\\n\\n') first,
    then newlines ('\\n'), sentences ('. ', '? ', '! '), and finally spaces. This keeps related
    sentences together rather than cutting arbitrarily.
    
    Why Chunk Overlap Matters:
    If a causal explanation or key sentence is split across chunks, overlapping preserves the
    surrounding context in both chunks, preventing boundary information loss during retrieval.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]
