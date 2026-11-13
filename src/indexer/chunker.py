from abc import ABC, abstractmethod
from typing import Any, Dict, List
import ast
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        pass

class PythonChunker(BaseChunker):
    def chunk(self, raw_text: str) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            language=Language.PYTHON
            chunk_size=self.max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
            )

        return splitter.create_documents

class MarkdownChunker(BaseChunker):
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            language=Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
        )

        return splitter.create_documents

def chunk_file(file_path: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    with open(file_path, encoding="utf-8", newline="") as f:
        raw_text = f.read()

    if file_path.endswith('.py'):
        chunker = PythonChunker(max_chunk_size)
    else:
        chunker = MarkdownChunker(max_chunk_size)

    chunks = chunker.chunk(raw_text)
    for chunk in chunks:
        chunk["file_path"] = file_path
        chunk["last_character_index"] = chunk.metadata["start_index"] + len(chunk.page_content)

    return chunks
