import walker, indexing
from index import chunk_text_with_overlap
from walker import find_target_files
from retriever.search import search

from pathlib import Path
import json

def load_database(filepath: str = "data/processed/database.json"):
    try:
        with open (filepath, 'r', encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Run 'python3 -m src index' first!")


def main():
    pass

if __name__ == "__main__":
    main()