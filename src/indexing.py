from typing import Dict, List, Any
from collections import Counter
import math

def compute_tf(text: str) -> Dict[str, float]:
    tokens = text.lower().split()
    if not tokens:
        return {}

    word_counts = Counter(tokens)
    return {word: count / len(tokens) for word, count in word_counts.items()}


def compute_idf(corpus: List[str]) -> Dict[str, float]:
    corpus_df = {}
    corpus_idf = {}

    for doc in corpus:
        for token in set(doc.lower().split()):
            corpus_df[token] = corpus_df.get(token, 0) + 1
    
    for token, df in corpus_df.items():
        corpus_idf[token] = math.log10(len(corpus) / df)

    return corpus_idf


def score_document(query: str, document: str, corpus: List[str]) -> float:
    doc_tf = compute_tf(document)
    corpus_idf = compute_idf(corpus)
    
    total_score = 0
    tokens = query.lower().split()

    for token in tokens:
        total_score += doc_tf.get(token, 0) * corpus_idf.get(token, 0)

    return total_score

def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("The overlap value cannot be greater or equal to the chunk size.")
    
    i = 0
    chunks = []

    while i < len(text):
        end_idx = min(chunk_size + i, len(text))
        chunks.append({
            'text': text[i:end_idx],
            'start': i,
            'end': end_idx
        })
        
        if end_idx == len(text):
            break
            
        i += chunk_size - overlap

    return chunks