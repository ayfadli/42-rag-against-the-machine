from typing import List, Dict, Any 
import indexing


def search(query: str, database: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    corpus = [chunk['text'] for chunk in database]

    for chunk in database:
        chunk['score'] = indexing.score_document(query, chunk['text'], corpus)
        