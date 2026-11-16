from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        pass

class PythonChunker(BaseChunker):
    def chunk(self, raw_text: str) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
            )

        return splitter.create_documents([raw_text])

class MarkdownChunker(BaseChunker):
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
        )

        return splitter.create_documents([text])

def chunk_file(file_path: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    with open(file_path, encoding="utf-8", newline="") as f:
        raw_text = f.read()

    if file_path.endswith('.py'):
        chunker = PythonChunker(max_chunk_size)
    else:
        chunker = MarkdownChunker(max_chunk_size)

    docs = chunker.chunk(raw_text)
    chunks = []
    for doc in docs:
        chunk = {}
        chunk['text'] = doc.page_content
        chunk['first_character_index'] = doc.metadata['start_index']
        chunk["last_character_index"] = doc.metadata["start_index"] + len(doc.page_content)
        chunk["file_path"] = file_path
        chunks.append(chunk)

    return chunks
