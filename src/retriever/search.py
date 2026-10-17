from typing import List, Dict, Any 
from src.indexer import index

def search(query: str, database: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    corpus = [chunk['text'] for chunk in database]

    for chunk in database:
        chunk['text'] = index.score_document(query, chunk['score'], corpus)

    sorted_database = sorted(database, key=lambda x: x['score'], reverse=True)

    return sorted_database[:top_k]