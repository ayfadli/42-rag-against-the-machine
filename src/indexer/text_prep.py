"""Text preparation shared by indexing and search."""
import re


def split_identifiers(text: str) -> str:
    """Split snake_case, camelCase and paths into separate words.

    Example:
        'load_lora_adapter' -> 'load lora adapter'
        'LoadLoRAAdapter'   -> 'Load Lo RAAdapter'  (good enough for BM25)
        'docs/features/lora.md' -> 'docs features lora md'
    """

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return text.replace("_", " ").replace("/", " ").replace(".", " ")

def prepare_for_index(text: str, file_path: str) -> str:
    """Returns what BM25 indexes for one chunk (NOT what is saved)."""
    full = file_path + "\n" + text
    return full + "\n" + split_identifiers(full)

def prepare_query(query: str) -> str:
    """Prepare a query exactly like the indexed text."""
    return query + "\n" + split_identifiers(query)


