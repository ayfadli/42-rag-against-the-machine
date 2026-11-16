from pathlib import Path
import fire
from pydantic import ValidationError
from tqdm import tqdm
from typing import List

from src.models import (MinimalSearchResults, StudentSearchResults)
from src.indexer.build import build_database
from src.walker import find_target_files
from src.retriever.search import (retrieve_sources, load_retriever)
from src.io_utils import load_dataset
from src.io_utils import save_json

class RAGCli:
    def __init__(self,
                raw_dir: str = "data/raw",
                processed_dir: str = "data/processed") -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def _validate_query_and_k(self, query: str, k: int):
        query = str(query).strip()
        if not isinstance(query, str) or not query:
            print("Error: Query must be a non-empty string.")
            return False
        if not isinstance(k, int) or k <= 0 or isinstance(k, bool):
            print("Error: k must be a positive number greater than or equal to 1.")
            return False

        return True
    def _validate_k(self, k: int) -> bool:
        if not isinstance(k, int) or k <= 0:
            print("Error: k must be a positive integer (>= 1).")
            return False
        return True

    def index(
            self,
            max_chunk_size: int = 2000,
            raw_dir: str | None = None,
            processed_dir: str | None = None
            ) -> None:

        if isinstance(max_chunk_size, bool) or not isinstance(max_chunk_size, int):
            print("The max_chunk_size must be an integer.")
            return

        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("Error: max_chunk_size must be between 1 and 2000 characters.")
            return

        target_raw = Path(raw_dir) if raw_dir else self.raw_dir
        target_proc = Path(processed_dir) if processed_dir else self.processed_dir

        try:
            if not Path(target_raw).is_dir():
                print(f"The file '{target_raw}' not found.")
                return
            target_files = find_target_files(target_raw, [".md", ".py", ".txt"])
            build_database(target_files, max_chunk_size, str(target_proc))

        except Exception as e:
            print(f"Error during indexing: {e}")

    def search(self, query: str, k: int = 10) -> None:
        if not self._validate_query_and_k(query, k):
            return
        retriver = load_retriever(str(self.processed_dir))
        if not retriver:
            return

        sources = retrieve_sources( retriver, str(query).strip(), k)
        for source in sources:
            print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")

    def search_dataset(self, dataset_path: str = "", k: int = 10, save_directory: str = "data/output/search_results") -> None:
        """Search every question of a dataset and save StudentSearchResults."""
        if not self._validate_k(k):
            return
        dataset = load_dataset(dataset_path)
        if dataset is None:
            return None
        retriver = load_retriever(str(self.processed_dir))
        if retriver is None:
            return

        results: List[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching", unit='q'):
            results.append(MinimalSearchResults(
                question_id=q.question_id,
                question = q.question,
                retrieved_sources=retrieve_sources(retriver, q.question, k)
            ))

        output = StudentSearchResults(search_results=results, k=k)
        out_path = Path(save_directory) / Path(dataset_path).name
        save_json(out_path, output.model_dump_json(indent=2))
        print(f"Saved student_search_results to {out_path}")

def main():
  fire.Fire(RAGCli)

if __name__ == "__main__":
    main()
