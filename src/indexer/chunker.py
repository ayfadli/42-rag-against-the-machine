from abc import ABC, abstractmethod
from typing import Any, Dict, List
import ast

class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        pass

class PythonChunker(BaseChunker):
    def chunk(self, raw_text: str) -> List[Dict[str, Any]]:


        lines = raw_text.splitlines(keepends=True)
        try:
            tree = ast.parse(raw_text)
        except (SyntaxError, ValueError):
            return self._split_oversized_node(lines, 0)
        max_chunk_size = self.max_chunk_size
    
        current_pos = 0
        line_starts = []
    
        for line in lines:
            line_starts.append(current_pos)
            current_pos += len(line)

        i = 0
        chunks = [{}]
    
        for node in tree.body:
            decorators = getattr(node, 'decorator_list', [])
            start_line = min([node.lineno] + [d.lineno for d in decorators])
            end_line = node.end_lineno
    
            node_lines = lines[start_line-1: end_line]
            node_text = ''.join(node_lines)
    
            first_character_index = line_starts[start_line - 1]
            last_character_index = first_character_index + len(node_text)
    
            current_chunk = chunks[i]
            node_text = raw_text[first_character_index : last_character_index]

            start = current_chunk.get('first_character_index', first_character_index)
            if last_character_index - start <= max_chunk_size:
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
        curr_len = 0
        max_chunk_size = self.max_chunk_size

        chunk = {'first_character_index': start_character_index, 'text': ''}

        for line in node_lines:
            line_start = start_character_index + curr_len

            if len(line) > max_chunk_size:
                if chunk['text']:
                    sub_chunks.append(chunk)

                for k in range(0, len(line), max_chunk_size):
                    piece = line[k: k + max_chunk_size]
                    sub_chunks.append({
                        'text': piece,
                        'first_character_index': line_start + k,
                        'last_character_index': line_start + k + len(piece) 
                    })

                chunk = {'first_character_index': line_start + len(line), 'text': ''}
            elif len(chunk['text']) + len(line) <= max_chunk_size:
                chunk['text'] += line
                chunk['last_character_index'] = line_start + len(line)
            else:
                sub_chunks.append(chunk)
                chunk = {
                    'text': line,
                    'first_character_index': line_start,
                    'last_character_index': line_start + len(line)
                }
            curr_len += len(line)

        if chunk['text']:
            sub_chunks.append(chunk)
        return sub_chunks

class MarkdownChunker(BaseChunker):
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        lines = text.splitlines(keepends=True)
        sections = []
        section_start = 0
        pos = 0
        in_code_block = False

        for line in lines:
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
            elif not in_code_block and line.startswith('#') and pos != section_start:
                sections.append((section_start), pos)
                section_start = pos
            pos += len(line)

        sections.append((section_start), pos)

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

    return chunks