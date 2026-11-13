from pathlib import Path
import fire

from src.indexer.build import build_database
from src.walker import find_target_files
from src.retriever.search import retrieve_sources


class RAGCli:
    def __init__(self,
                raw_dir: str = "data/raw",
                processed_dir: str = "data/processed") -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def _validate_query_and_k(self, query: str, k: int):
        query = str(query).strip()
        if not isinstance(query, str) or not query:
            print("Error: Quey must be a non-empty string.")
            return False
        if not isinstance(k, int) or k <= 0 or isinstance(k, bool):
            print("Error: k must be a positive number greater than or equal to 1.")
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
            target_files = find_target_files(target_raw, [".md", ".py"])
            build_database(target_files, max_chunk_size, str(target_proc))

        except Exception as e:
            print(f"Error during indexing: {e}")

    def search(self, query: str, k: int = 10) -> None:
        if not self._validate_query_and_k(query, k):
            return

        retrieve_sources(query, k, str(self.processed_dir))



def main():
  fire.Fire(RAGCli)

if __name__ == "__main__":
    main()
