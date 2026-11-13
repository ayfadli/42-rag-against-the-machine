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
        return

    query_tokens = bm25s.tokenize([query], stopwords="en")

    results, scores = retriever.retrieve(query_tokens, k=top_k)

    print(results[0])
    print("\n\n\n")
    print(scores[0])

