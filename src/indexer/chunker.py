from abc import ABC, abstractmethod
from typing import Any, Dict, List
import ast
from rich import print

class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        pass

class PythonChunker(BaseChunker):
    def chunk(self, raw_text: str) -> List[Dict[str, Any]]:

        tree = ast.parse(raw_text)
        lines = raw_text.splitlines(keepends=True)
        max_chunk_size = self.max_chunk_size
    
        current_pos = 0
        line_starts = []
    
        for line in lines:
            line_starts.append(current_pos)
            current_pos += len(line)

        i = 0
        chunks = [{}]
    
        for node in tree.body:
            start_line = node.lineno
            end_line = node.end_lineno
    
            node_lines = lines[start_line-1: end_line]
            node_text = ''.join(node_lines)
    
            first_character_index = line_starts[node.lineno - 1]
            last_character_index = first_character_index + len(node_text)
    
            current_chunk = chunks[i]
            node_text = raw_text[first_character_index : last_character_index] 
    
            if len(node_text) + len(current_chunk.get('text', "")) <= max_chunk_size:
                if current_chunk.get('first_character_index') is None:
                    current_chunk['first_character_index'] = first_character_index
    
                current_chunk['last_character_index'] = last_character_index
                current_chunk['text'] = raw_text[current_chunk['first_character_index'] : last_character_index]
            else:
                if len(node_text) <= max_chunk_size:
                    i+= 1
                    chunks.append({'text': node_text, 
                                'first_character_index': first_character_index,
                                'last_character_index': last_character_index
                                })
                else:
                    sub_chunks = self._split_oversized_node(node_lines, first_character_index)
                    chunks.extend(sub_chunks)
                    i = len(chunks) - 1
    
        chunks = [c for c in chunks if c.get('text')]

        return chunks

    def _split_oversized_node(
            self,
            node_lines: List[str], 
            start_character_index: int
    ) -> List[Dict[str, Any]]:


        sub_chunks = []
        total_len = sum(len(s) for s in node_lines)
        curr_len = 0
        max_chunk_size = self.max_chunk_size

        i = 0
        chunk = dict()
        chunk['first_character_index'] = start_character_index
        while (curr_len < total_len and i < len(node_lines)):
            line = node_lines[i]
            if len(chunk.get('text', '')) + len(line) <= max_chunk_size:
                chunk['text'] = chunk.get('text', '') + line
                curr_len += len(line)
                chunk['last_character_index'] = start_character_index + curr_len
                i += 1
            else:
                sub_chunks.append(chunk)
                chunk = dict()
                chunk['first_character_index'] = start_character_index + curr_len
        else:
            sub_chunks.append(chunk)

        return sub_chunks

class MarkdownChunker(BaseChunker):
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        pass

def chunk_text(target_files: list[str], max_chunk_size: int) -> List[Dict[str, Any]]:
    chunks = []
    for file in target_files:
        if file.endswith('.md'):
            pass
        else:
            chunks.chunk_python_code(file, max_chunk_size)

    return chunks