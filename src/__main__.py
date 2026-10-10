import walker, indexing
from pathlib import Path
import json

def main():
    file_names = walker.find_target_files("data/raw/vllm-0.10.1/", [".py", ".md"])
    print(f"Found {len(file_names)} files. Chunking...")
    all_chunks = []

    for file in file_names:
        try:
            raw_text = Path(file).read_text()
            file_chunks = indexing.chunk_text_with_overlap(raw_text, 1200, 200)

            for chunk in file_chunks:
                chunk['filepath'] = file

            all_chunks.extend(file_chunks)

        except Exception as e:
            print(f"Error reading {file}: {e}")


    print(f"Success! Generated {len(all_chunks)} total chunks.")
    with open("database.json", 'w', encoding="utf-8") as output:
        json.dump(all_chunks, output, indent=2)

    print("Database saved to 'database.json'.")


if __name__ == "__main__":
    main()