from src.models import RagDataset
from pydantic import ValidationError
from pathlib import Path

def load_dataset(path: str)  -> RagDataset | None:
    """Read and Validate a RagDataset JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return RagDataset.model_validate_json(f.read())
    except FileNotFoundError:
        print(f"Error: dataset '{path}' not found.")
    except IsADirectoryError:
        print(f"Error: '{path}' is a directory, not a file.")
    except ValidationError:
        print(f"Error: '{path}' not a valid dataset.")

    return None

def save_json(path: Path, content: str) -> bool:
    """Write text to path, creating parent folders."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding="utf-8") as f:
            f.write(content)
        return True

    except OSError as e:
        print(f"Error: cannot write '{path}' ({e})")
        return False
