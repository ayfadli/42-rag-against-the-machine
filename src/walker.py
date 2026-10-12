from pathlib import Path
from typing import List

IGNORED_DIRS: Set[str] = {".git", "__pycache__", ".venv", "venv", "build", "dist", "egg-info"}

def find_target_files(directory_path: str, allowed_extensions: List[str]) -> List[str]:
    base_path = Path(directory_path)
    allowed_set = {ext.lower() for ext in allowed_extensions}
    file_names = []

    for item in base_path.rglob("*"):
        if any(part.startswith(".") or part in IGNORED_DIRS for part in item.parts):
            continue

        if item.is_file() and item.suffix.lower() in allowed_set:
            file_names.append(str(item))

    return file_names

find_target_files("data/", [".md", ".py"])