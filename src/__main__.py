import argparse
import json
from pathlib import Path

import walker, indexing
from index import chunk_text_with_overlap
from walker import find_target_files
from retriever.search import search

def load_database(filepath: str = "data/processed/database.json"):
    try:
        with open (filepath, 'r', encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Run 'python3 -m src index' first!")


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")

    subparsers = parser.add_subparsers(dest="command ", required=True)


if __name__ == "__main__":
    main()