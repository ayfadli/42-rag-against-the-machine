from pathlib import Path

def find_target_files(directory_path: str, allowed_extension: List[str]) -> List[str]:
    file_names = []
    for item in Path(directory_path).rglob("*"):
        if item.is_file() and item.suffix in allowed_extension:
            file_names.append(str(item))
            # print(str(item))

    return file_names

find_target_files("data/", [".md", ".py"])