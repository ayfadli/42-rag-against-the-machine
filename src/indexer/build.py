from src.indexer import chunker
from src.indexer.text_prep import prepare_for_index
from tqdm import tqdm
from typing import List
import bm25s
import sys

def build_database(
        file_names: List[str],
        chunk_size: int = 2000,
        processed_dir: str = None
        ):

    all_chunks = []

    for path in tqdm(file_names, desc="Chunking...", unit="file"):
        try:
            chunks = chunker.chunk_file(path, chunk_size)
            all_chunks.extend(chunks)
        except (UnicodeDecodeError, OSError) as e:
            tqdm.write(f"SKIPPING: '{path}' error: {e}")

    if not all_chunks:
        print("No chunks available.", file=sys.stderr)
        return

    texts = [prepare_for_index(chunk['text'], chunk['file_path']) for chunk in all_chunks]
    tokens = bm25s.tokenize(texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(tokens)

    retriever.save(processed_dir, corpus=all_chunks)

    print(f"Success! Generated {len(all_chunks)} total chunks.")


    print(f"Database saved to '{processed_dir}'.")
