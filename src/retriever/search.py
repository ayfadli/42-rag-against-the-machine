from typing import List, Dict, Any
import bm25s
from src.models import MinimalSource
from rich import print
from src.indexer.text_prep import prepare_query

def load_retriever(processed_dir: str) -> bm25s.BM25 | None:
    """Load the BM25 index and its corpus once. Return None if missing."""
    try:
        return bm25s.BM25.load(processed_dir ,load_corpus=True)
    except (OSError, ValueError) as e:
        print(f"Error: cannot load index from '{processed_dir}' ({e}). "
            f"Run 'index' first.")
        return None
        
def retrieve_sources(retriever: bm25s.BM25, query: str,
                     k: int) -> List[MinimalSource]:
    k = min(k, len(retriever.corpus))

    sources = []
    query_tokens = bm25s.tokenize([prepare_query(query)], stopwords="en", show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)

    for chunk, score in zip(results[0], scores[0]):
        if score == 0:
            continue
        minimal_source = MinimalSource(file_path=chunk['file_path'],
                                       first_character_index = chunk['first_character_index'],
                                       last_character_index = chunk['last_character_index'])
        sources.append(minimal_source)

    return sources

