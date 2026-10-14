import argparse
import json
from pathlib import Path

from src.indexer.index import chunk_text_with_overlap
from src.indexer.build import build_database
from src.walker import find_target_files
from src.retriever.search import search
from src.generator.llm_client import ask_llm
from src.generator.prompt_builder import build_prompt

class RAGCli:
    def __init__(self,
                raw_dir: str = "data/raw",
                porcessed_dir: str = "data/processed") -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(self.processed_dir)

    def _validate_query_and_k(self, query: str, k: int):
        if not isinstance(query, str) or not query:
            print("Error: Quey must be a non-empty string.")
            return False
        if not isinstance(k, int) or k <= 0:
            print("Error: k must be greater than or equal to 0.")
            return False

        return True

    def index(
            self,
            max_chunk_size: int = 2000,
            raw_dir: str | None = None,
            processed_dir: str | None = None
            ) -> None:
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("Error: max_chunk_size must be between 1 and 2000 characters.")
            return

        target_raw = Path(raw_dir) if raw_dir else self.raw_dir
        target_proc = Path(processed_dir) if processed_dir else self.processed_dir

        try:
            if not Path(target_raw).is_file:
                print(f"The file '{target_raw}' not found.")
                return
            target_files = find_target_files(target_raw, [".md", ".py"])
            
        except Exception as e:
            print(f"Error during indexing: {e}")


def load_database(filepath: str = "data/processed/database.json"):
    try:
        with open (filepath, 'r', encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Run 'python3 -m src index' first!")


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Indexing the codebase.")
    index_parser.add_argument("--source", default="data/raw")
    index_parser.add_argument("--output", default="data/processed/database.json")

    chat_parser = subparsers.add_parser("chat", help="Return the top-k sources for a single query.")
    chat_parser.add_argument()

if __name__ == "__main__":
    main()