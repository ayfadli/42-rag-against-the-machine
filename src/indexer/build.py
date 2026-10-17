from src.walker import find_target_files 
from src.indexer.index import chunk_text_with_overlap
from pathlib import Path
import json
from typing import List

def build_database(
        output_path: str = "data/processed/database.json",
        file_names: List[str] = None,
        chunk_size: int = 1200,
        overlap: int = 200
        ):

    # file_names = find_target_files("data/raw/vllm-0.10.1/", [".py", ".md"])
    # print(f"Found {len(file_names)} files. Chunking...")
    all_chunks = []

    for file in file_names:
        try:
            with open (file, encoding="utf-8", newline="") as f:  
                raw_text = Path(f).read_text()

            file_chunks = chunk_text_with_overlap(raw_text, chunk_size, overlap)

            for chunk in file_chunks:
                chunk['filepath'] = file

            all_chunks.extend(file_chunks)

        except Exception as e:
            print(f"Error reading {file}: {e}")


    print(f"Success! Generated {len(all_chunks)} total chunks.")
    with open("database.json", 'w', encoding="utf-8") as output:
        json.dump(all_chunks, output, indent=2)

    print(f"Database saved to '{output_path}'.")
