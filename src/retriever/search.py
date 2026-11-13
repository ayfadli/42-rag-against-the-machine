from typing import List, Dict, Any
from src.indexer import index
import bm25s
from src.models import MinimalSource
from rich import print

def retrieve_sources(query: str, top_k: int, processed_dir: str) -> List[Dict[str, Any]]:
    try:
        retriever = bm25s.BM25.load(processed_dir, load_corpus=True)
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        return []

    sources = []

    query_tokens = bm25s.tokenize([query], stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=top_k)

    for chunk, score in zip(results[0], scores[0]):
        if score == 0:
            continue
        minimal_source = MinimalSource(file_path=chunk['file_path'],
                                       first_character_index = chunk['first_character_index'],
                                       last_character_index = chunk['last_character_index'])
        sources.append(minimal_source)

    return sources

