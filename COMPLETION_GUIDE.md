# RAG against the machine: completion guide

Everything that's left to reach a complete mandatory part, in order.
Each task has: **Where**, **Problem**, **Fix** (hints and examples), and **Check**.

> Work top to bottom. Each step builds on the previous one.
> After each step, run its **Check** before moving on.

---

## Status (audited 2026-11-15)

| Area | Status |
|---|---|
| Python chunking (langchain `Language.PYTHON`) | ✅ 0 chunks over the limit, 0 wrong offsets, sizes 2000 and 50 |
| Markdown/text chunking (langchain `Language.MARKDOWN`) | ✅ same |
| `.txt` files indexed | ✅ done in `__main__.py:49` |
| Datasets in `data/datasets/` | ✅ copied |
| flake8 and mypy installed (dev) | ✅ |
| Indexing time | ✅ about 3 s (limit is 300 s) |
| `requires-python` | ❌ missing |
| Recall@5 docs ≥ 0.80 | ❌ 0.78 → needs Task 2 |
| Recall@5 code ≥ 0.50 | ⚠️ 0.545 (just above the limit) → Task 2 gives 0.74 |
| `search --k 99999999` | ❌ crashes with a traceback |
| `search_dataset` | ❌ not implemented |
| `evaluate` | ❌ not implemented |
| `answer` / `answer_dataset` (local Qwen) | ❌ `llm_client.py` calls the Groq API, which isn't allowed |
| `make lint` | ❌ 48 flake8 errors and 18 mypy errors in `src/` |
| Makefile | ❌ 2 bugs |
| README | ❌ only 2 lines |

---

## Task 1: `requires-python` (2 min)

**Where:** `pyproject.toml`

**Problem:** without it, uv picks `>=3.14`. If the evaluator's machine has
Python 3.12, `uv sync` can fail, and then **nothing runs**.

**Fix:**
```toml
[project]
name = "rag-against-machine"
version = "0.1.0"
requires-python = ">=3.10"
```
then:
```bash
uv lock
```

**Check:** `uv sync` doesn't print the
`No requires-python value found` warning anymore.

---

## Task 2: improve retrieval (30 min)

I measured these on the public datasets (R@5):

| change | docs | code |
|---|---|---|
| before | 0.78 ❌ | 0.545 |
| + `.txt` (✅ already done) | 0.81 | 0.556 |
| + file path in the indexed text | 0.85 | 0.606 |
| + identifier splitting | 0.82 | 0.667 |
| **+ all three** | **0.83** | **0.737** |

### Why each one helps

**File path:** the question *"How do I use **LoRA** adapters?"* has its answer in
`docs/features/lora.md`. The path itself contains the right word, but BM25
only ever sees the chunk text, so it can't use that clue.

**Identifier splitting:** BM25's tokenizer keeps `load_lora_adapter` as **one**
token. A question that says *"load a LoRA adapter"* produces 3 tokens:
`load`, `lora`, `adapter`. None of them equals `load_lora_adapter`, so you
get **zero** matches. If you split the identifiers, they match.

```
code:     "def load_lora_adapter(self, request: LoadLoRAAdapterRequest)"
tokens:   ["def", "load_lora_adapter", "self", "request", "loadloraadapterrequest"]
after split, also:  ["load", "lora", "adapter", "Load", "Lo", "RAAdapter", ...]
question: "How do I load a LoRA adapter?"
tokens:   ["load", "lora", "adapter"]   ← now they match ✅
```

We keep **both** the original and the split version. The original still
matches when the question quotes an identifier exactly, and the split version
matches when it paraphrases.

### Fix

**2a.** Create `src/indexer/text_prep.py`:
```python
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
    """Return what BM25 indexes for one chunk (NOT what is saved)."""
    full = file_path + "\n" + text
    return full + "\n" + split_identifiers(full)


def prepare_query(query: str) -> str:
    """Prepare a query exactly like the indexed text."""
    return query + "\n" + split_identifiers(query)
```

**2b.** `src/indexer/build.py`, line 26:
```python
texts = [prepare_for_index(c["text"], c["file_path"]) for c in all_chunks]
```
⚠️ Keep `corpus=all_chunks` in `retriever.save(...)` exactly as it is.
Only the **search text** changes. The saved chunks and their offsets stay the same.

**2c.** `src/retriever/search.py`, line 15:
```python
query_tokens = bm25s.tokenize([prepare_query(query)], stopwords="en",
                              show_progress=False)
```
`show_progress=False` removes the `Split strings: 0%|...` noise from your
output. Do the same for `retriever.retrieve(..., show_progress=False)`.

**Golden rule:** index and query must go through **the same** preparation.
If you change one, change the other.

**Check:**
```bash
uv run python -m src index --max_chunk_size 2000
```
```bash
uv run python -m src search "How do I load a LoRA adapter dynamically?" --k 5
```
`docs/features/lora.md [4695:6098]` should appear in the top 5.
After Task 4 (`evaluate`), you should see about 0.83 for docs and 0.74 for code.

---

## Task 3: refactor search and add `search_dataset` (45 min)

### 3a. Load the index once

**Where:** `src/retriever/search.py`

**Problem 1:** `retrieve_sources` calls `bm25s.BM25.load(...)` on **every**
query. For 200 questions, that's 200 index loads. The subject's limit is 200
questions in 90 s.

**Problem 2:** `--k 99999999` crashes:
```
ValueError: k of 99999999 is larger than the number of available scores, which is 14369
```

**Problem 3:** the return type says `List[Dict[str, Any]]`, but it returns
`MinimalSource` objects. mypy complains about this in `__main__.py:61`.

**Fix:** split it into two functions:
```python
def load_retriever(processed_dir: str) -> bm25s.BM25 | None:
    """Load the BM25 index and its corpus once. Return None if missing."""
    try:
        return bm25s.BM25.load(processed_dir, load_corpus=True)
    except (OSError, ValueError) as e:
        print(f"Error: cannot load index from '{processed_dir}' ({e}). "
              f"Run 'index' first.")
        return None


def retrieve_sources(retriever: bm25s.BM25, query: str,
                     k: int) -> List[MinimalSource]:
    """Return the top-k sources for one query."""
    k = min(k, len(retriever.corpus))       # ← fixes the crash
    ...
```

**Example of the k fix:**
```
corpus = 14369 chunks, user asks k = 99999999
min(99999999, 14369) = 14369  → no crash
```

### 3b. Update `search` in `__main__.py`
```python
def search(self, query: str, k: int = 10) -> None:
    """Print the top-k sources for a single query."""
    if not self._validate_query_and_k(query, k):
        return
    retriever = load_retriever(str(self.processed_dir))
    if retriever is None:
        return
    for s in retrieve_sources(retriever, str(query), k):
        print(f"{s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")
```

### 3c. The `search_dataset` command

The evaluator runs exactly this command:
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```
and expects:
```
data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```
The output file has **the same name as the input dataset**.

**Skeleton:**
```python
from pydantic import ValidationError
from tqdm import tqdm
from src.models import (RagDataset, MinimalSearchResults,
                        StudentSearchResults)

def search_dataset(self, dataset_path: str = "", k: int = 10,
                   save_directory: str = "data/output/search_results"
                   ) -> None:
    """Search every question of a dataset and save StudentSearchResults."""
    if not self._validate_k(k):           # validate k (see Task 6)
        return
    dataset = load_dataset(dataset_path)   # returns None on error
    if dataset is None:
        return
    retriever = load_retriever(str(self.processed_dir))
    if retriever is None:
        return

    results: List[MinimalSearchResults] = []
    for q in tqdm(dataset.rag_questions, desc="Searching", unit="q"):
        results.append(MinimalSearchResults(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=retrieve_sources(retriever, q.question, k),
        ))

    output = StudentSearchResults(search_results=results, k=k)
    out_path = Path(save_directory) / Path(dataset_path).name
    save_json(out_path, output.model_dump_json(indent=2))
    print(f"Saved student_search_results to {out_path}")
```

**Helper for reading a dataset (put it in a small `src/io_utils.py`):**
```python
def load_dataset(path: str) -> RagDataset | None:
    """Read and validate a RagDataset JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return RagDataset.model_validate_json(f.read())
    except FileNotFoundError:
        print(f"Error: dataset '{path}' not found.")
    except IsADirectoryError:
        print(f"Error: '{path}' is a directory, not a file.")
    except ValidationError as e:
        print(f"Error: '{path}' is not a valid dataset "
              f"({e.error_count()} problems).")
    return None


def save_json(path: Path, content: str) -> bool:
    """Write text to path, creating parent folders."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except OSError as e:
        print(f"Error: cannot write '{path}' ({e}).")
        return False
```

💡 **Why `model_validate_json`?** It parses the JSON **and** checks the
structure in one step. Broken JSON (`{bad`) and the wrong structure
(`{"foo": 1}`) both raise `ValidationError`, so one `except` covers both.

💡 **Empty question in a dataset?** `question: ""` is valid pydantic but
gives no useful search. Return `retrieved_sources=[]` for that question
instead of crashing.

**Check (time it, the limit is 90 s for 200 questions):**
```bash
time uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```
```bash
time uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```
Then open the JSON. Each `retrieved_sources` should have up to 10 items,
and each one is ≤ 2000 characters wide.

---

## Task 4: the `evaluate` command (30 min)

**Where:** new file `src/evaluation.py` and a new command in `__main__.py`

**The metric, from the subject:**
> recall@k is the share of a question's correct sources that you retrieve
> in your top-k. A correct source counts as found when one of your results
> is in **the same file** and **overlaps** its range (IoU ≥ 0.05).

### IoU (intersection over union)

```
ground truth:  [1000 ─────────── 2000]
your result:          [1500 ─────────── 2500]

intersection = 2000 - 1500 = 500
union        = 2500 - 1000 = 1500
IoU          = 500 / 1500  = 0.33  ≥ 0.05 → FOUND ✅
```
```
ground truth:  [1000 ── 1100]
your result:                    [3000 ── 5000]
intersection = 0 → IoU = 0 → NOT FOUND ❌
```

### Code
```python
def overlaps(gt: MinimalSource, res: MinimalSource,
             threshold: float = 0.05) -> bool:
    """True if res is in the same file and overlaps gt with IoU >= threshold."""
    if gt.file_path != res.file_path:
        return False
    inter = max(0, min(gt.last_character_index, res.last_character_index)
                - max(gt.first_character_index, res.first_character_index))
    union = (max(gt.last_character_index, res.last_character_index)
             - min(gt.first_character_index, res.first_character_index))
    return union > 0 and inter / union >= threshold


def recall_at_k(gt_sources: List[MinimalSource],
                retrieved: List[MinimalSource], k: int) -> float:
    """Share of gt_sources found in the first k retrieved sources."""
    if not gt_sources:
        return 0.0
    top = retrieved[:k]
    found = sum(1 for gt in gt_sources if any(overlaps(gt, r) for r in top))
    return found / len(gt_sources)
```

### Example with 2 ground-truth sources
```
gt = [A, B], your top-5 contains something overlapping A but not B
recall@5 for this question = 1/2 = 0.5
final recall@5 = average over all questions
```

### The command
```python
def evaluate(self, student_search_results_path: str = "",
             dataset_path: str = "") -> None:
    """Print recall@1/3/5/10 of search results against an answered dataset."""
    # 1. load StudentSearchResults (model_validate_json, same try/except style)
    # 2. load the RagDataset; keep only AnsweredQuestion (they have .sources)
    # 3. build dict: question_id -> AnsweredQuestion
    # 4. for each k in (1, 3, 5, 10) with k <= results.k:
    #        average recall_at_k over questions whose id is in the dict
    # 5. print like the moulinette:
    #    Recall@1: 0.580 Recall@3: 0.750 Recall@5: 0.830 Recall@10: 0.880
```

⚠️ **Match by `question_id`**, not by position. Never assume both files
are in the same order.

⚠️ The subject says: *"Your solution must never import or call the
moulinette."* This is your own implementation.

**Check:**
```bash
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
Expect Recall@5 ≈ **0.83** for docs and ≈ **0.74** for code (after Task 2).

---

## Task 5: local Qwen, `answer` and `answer_dataset` (2 h)

### 5a. Rewrite `src/generator/llm_client.py` completely

**Problem:** it calls the **Groq cloud API** (`api.groq.com`) with an API key.
The subject requires the **local** `Qwen/Qwen3-0.6B`. It also has a bug:
`[{'role': "user"}, {"content": "prompt"}]` sends the literal word
`"prompt"`, not your prompt. Remove `requests`, `dotenv`, and the `.env` key.

**Fix:**
```python
"""Local Qwen client using Hugging Face transformers."""
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenClient:
    """Load Qwen once and generate answers locally on CPU/GPU."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate an answer for prompt (greedy, no thinking mode)."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens,
                                     skip_special_tokens=True).strip()
```

**Line by line:**
- `apply_chat_template`: wraps your prompt in Qwen's chat format
  (`<|im_start|>user ... <|im_end|><|im_start|>assistant`).
- `enable_thinking=False`: Qwen3 otherwise writes a long `<think>...</think>`
  block first, which is slow and ends up inside your answer.
- `do_sample=False`: greedy decoding, so the same question always gives the
  same answer. That's reproducible, and it's good for grounding.
- `output[0][len(input):]`: the model returns **prompt + answer**, so we
  keep only the new tokens.

⚠️ **Load the model once.** For `answer_dataset`, create `QwenClient()`
**before** the loop, never inside it. Loading takes several seconds.

💡 The first run downloads about 1.2 GB into `~/.cache/huggingface`.
Run it once before the defense so the download doesn't happen live.

### 5b. Fix `src/generator/prompt_builder.py`

**Problems:**
- It reads `chunk['text']` and `chunk.get('filepath')`. Search results are
  `MinimalSource`, which has `file_path` and offsets but **no text**.
- There's no size limit, so 10 chunks × 2000 characters = 20,000 characters.
  That's slow on CPU and confuses a 0.6B model.

**Fix:** read the text from the file using the offsets, and stop at a budget:
```python
def read_source_text(src: MinimalSource) -> str:
    """Return the exact text a source points to ('' on error)."""
    try:
        with open(src.file_path, encoding="utf-8", newline="") as f:
            return f.read()[src.first_character_index:
                            src.last_character_index]
    except OSError:
        return ""


def build_prompt(query: str, sources: List[MinimalSource],
                 max_context_chars: int = 6000) -> str:
    """Build a grounded prompt from the best sources, within a budget."""
    blocks: List[str] = []
    used = 0
    for i, src in enumerate(sources, start=1):
        text = read_source_text(src)
        if not text or used + len(text) > max_context_chars:
            break
        blocks.append(f"[Source {i}: {src.file_path}]\n{text}")
        used += len(text)
    context = "\n\n".join(blocks)
    return (
        "You answer questions about the vLLM codebase.\n"
        "Use ONLY the sources below. If they do not contain the answer, "
        "say you don't know.\n"
        "Answer in 2-5 sentences. Mention function or file names "
        "when useful.\n\n"
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {query}\n"
        "ANSWER:"
    )
```

**Why 6000 characters?** That's roughly 1,500 tokens: the top 3 or so chunks.
Your best chunks come first, so cutting the tail loses little. You can talk
about this trade-off at the defense: more context means slower answers and
more noise.

### 5c. The `answer` command (single query)
```python
def answer(self, query: str, k: int = 5) -> None:
    """Retrieve top-k sources for query and print a generated answer."""
    # validate query and k → load_retriever → retrieve_sources
    # → build_prompt → QwenClient().generate → print answer + sources
```

### 5d. The `answer_dataset` command
```python
def answer_dataset(self, student_search_results_path: str = "",
                   save_directory: str = "data/output/search_results_and_answer"
                   ) -> None:
    """Generate answers for saved search results."""
    # 1. load StudentSearchResults with model_validate_json (try/except)
    # 2. client = QwenClient()                     ← ONCE
    # 3. for r in tqdm(results.search_results, desc="Answering"):
    #        prompt = build_prompt(r.question, r.retrieved_sources)
    #        answers.append(MinimalAnswer(**r.model_dump(),
    #                                     answer=client.generate(prompt)))
    # 4. out = StudentSearchResultsAndAnswer(search_results=answers, k=results.k)
    # 5. save to Path(save_directory) / Path(student_search_results_path).name
    # 6. print(f"Saved student_search_results_and_answer to {path}")
```

💡 `MinimalAnswer(**r.model_dump(), answer=...)` copies `question_id`,
`question` and `retrieved_sources` from the search result and adds `answer`.

💡 If one generation fails, catch it and store
`answer="Error: generation failed."`. Never lose the whole run.

**Check:**
```bash
uv run python -m src answer "How do I load a LoRA adapter dynamically?" --k 5
```
```bash
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
The answer should mention `/v1/load_lora_adapter`. Expect the dataset run
to take a while on CPU (about 3–10 s per question).

---

## Task 6: input validation, never a traceback (30 min)

The subject says: *"The CLI is tested with such edge cases and must
never crash with an unhandled traceback."*

### Current results
| command | now | expected |
|---|---|---|
| `search "" --k 5` | ✅ clean error | |
| `search hello --k 0` | ✅ clean error | |
| `search hello --k abc` | ✅ clean error | |
| `search hello --k 99999999` | ❌ **traceback** | fixed by Task 3a (`min(k, len(corpus))`) |
| `search_dataset --dataset_path nope.json` | ❌ Fire help screen | fixed once the command exists |

### Small fixes
- Typo in `__main__.py:19`: `"Quey"` → `"Query"`.
- `_validate_query_and_k`: `str(query).strip()` runs **before**
  `isinstance(query, str)`, so the `isinstance` check can never fail. Fire can pass
  numbers (`search 123`). Just keep `query = str(query).strip()` and check
  that it isn't empty.
- Split it into `_validate_k(k)` and `_validate_query(q)`, because
  `search_dataset` needs only `k`.

### Safety net in `main()`
```python
def main() -> None:
    """Entry point: run the Fire CLI without ever showing a traceback."""
    try:
        fire.Fire(RAGCli)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:  # last-resort guard required by the subject
        print(f"Error: {e}")
```
⚠️ This is a **safety net**, not a replacement. Each command should still
handle its own expected errors with clear messages.

### Full edge-case test list
Run every line. None should show `Traceback`:
```bash
uv run python -m src search "" --k 5
```
```bash
uv run python -m src search "hello" --k 0
```
```bash
uv run python -m src search "hello" --k -3
```
```bash
uv run python -m src search "hello" --k abc
```
```bash
uv run python -m src search "hello" --k 2.5
```
```bash
uv run python -m src search "hello" --k 99999999
```
```bash
uv run python -m src search "zzqqxx blorp" --k 5
```
```bash
uv run python -m src search_dataset --dataset_path nope.json --k 5
```
```bash
uv run python -m src search_dataset --dataset_path data --k 5
```
```bash
echo "{bad" > /tmp/bad.json && uv run python -m src search_dataset --dataset_path /tmp/bad.json --k 5
```
```bash
echo '{"foo": 1}' > /tmp/wrong.json && uv run python -m src search_dataset --dataset_path /tmp/wrong.json --k 5
```
```bash
uv run python -m src index --max_chunk_size 0
```
```bash
uv run python -m src index --max_chunk_size 5000
```
```bash
uv run python -m src index --raw_dir does/not/exist
```
```bash
uv run python -m src evaluate --student_search_results_path nope.json --dataset_path nope.json
```
```bash
uv run python -m src answer_dataset --student_search_results_path nope.json
```
Also test **search before index**: `rm -rf data/processed/*`, then run
`search`. It should print *"Run 'index' first."*, then you re-index.

---

## Task 7: Makefile, flake8, mypy (1 h)

### 7a. Makefile bugs
```makefile
debug:
	uv run python -m pdb -m src          # was "-m src/" → wrong module name

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	#                ^^^^^^^ was "--ignore-missing-import" (missing s)
```

### 7b. Exclude venvs and data
`flake8 .` and `mypy .` would otherwise scan `venv/` and `.venv/`
(thousands of errors that aren't yours).

Create `.flake8`:
```ini
[flake8]
exclude = .venv,venv,data,__pycache__,.git
```
Add to `pyproject.toml`:
```toml
[tool.mypy]
exclude = ["venv", ".venv", "data"]
```

### 7c. Your current errors in `src/` (48 flake8, 18 mypy)

| code | count | meaning | fix |
|---|---|---|---|
| E501 | 16 | line > 79 chars | break inside `()` or split f-strings |
| E302 | 9 | need 2 blank lines before def/class | add a blank line |
| W293 | 5 | blank line contains spaces | delete the spaces |
| E251 | 4 | spaces around `=` in kwargs | `f(a=1)` not `f(a = 1)` |
| W391 | 3 | blank line at end of file | delete it |
| F401 | 3 | unused import | delete it |
| W291 | 2 | trailing spaces | delete them |
| E128 | 2 | bad continuation indent | align with the `(` |

**mypy examples from your code:**
```python
# build.py:7-10 ❌
def build_database(file_names: List[str], chunk_size: int = 2000,
                   processed_dir: str = None):
# ✅
def build_database(file_names: List[str], chunk_size: int = 2000,
                   processed_dir: str = "data/processed") -> None:
```
```python
# __main__.py:49 ❌ passes a Path where str is expected
find_target_files(target_raw, ...)
# ✅
find_target_files(str(target_raw), ...)
```
```python
# __main__.py:16 ❌ missing return type
def _validate_query_and_k(self, query: str, k: int):
# ✅
def _validate_query_and_k(self, query: str, k: int) -> bool:
```

💡 **Speed tip:** most E501/W293/W291/E302 errors can be fixed quickly by
your editor (VS Code: install the "autopep8" extension → Format Document).
Fix the rest by hand.

### 7d. Docstrings
The subject requires PEP 257 docstrings on **all** functions and classes.
Google style:
```python
def recall_at_k(gt: List[MinimalSource], res: List[MinimalSource],
                k: int) -> float:
    """Compute recall@k for one question.

    Args:
        gt: The ground-truth sources.
        res: The retrieved sources, best first.
        k: How many retrieved sources to consider.

    Returns:
        The share of ground-truth sources found, between 0.0 and 1.0.
    """
```

### 7e. Leftovers to clean up
- `src/__pycache__/evaluation.cpython-314.pyc` belongs to a deleted file. `make clean` removes it.
- `from rich import print` in `search.py`: it works (rich is a dependency
  now), but it's not needed. Use the plain `print` and remove `rich` from the
  dependencies.
- `requirements.txt` is outdated (it only lists fire). Delete it; uv uses
  `pyproject.toml` and `uv.lock`.
- `.env` holds a Groq API key. It's ignored by git (good), but delete it once
  Task 5 is done. You won't need it.
- `venv/` (old pip venv) is no longer needed; uv uses `.venv/`.

**Check:**
```bash
make lint
```
→ **0 errors**. Then try `make lint-strict` (optional; it's recommended).

---

## Task 8: README.md (1 h)

**The first line must be exactly this** (italic, your login):
```markdown
*This project has been created as part of the 42 curriculum by <your_login>.*
```

### Required sections, with what to write
| Section | What to write |
|---|---|
| **Description** | RAG over vLLM 0.10.1: index, retrieve with BM25, answer with local Qwen3-0.6B, measure recall@k. |
| **Instructions** | `uv sync` / `make install`, where to put `data/raw`, and every CLI command. |
| **Resources** | BM25 (Robertson & Zaragoza, *The Probabilistic Relevance Framework*), the bm25s docs, the Hugging Face Qwen3 model card, the langchain text splitters docs, and **how AI was used** (be honest: for example, "used Claude for reviewing bugs and explaining concepts; all code written and understood by me"). |
| **System architecture** | A diagram: `walker → chunker → text_prep → bm25s index (data/processed)` → `search → prompt_builder → Qwen → JSON`. |
| **Chunking strategy** | Python: separators `class` → `def` → blank line → line → char. Markdown: headings → code fences → paragraphs → lines. Max 2000, overlap 0, exact offsets via `add_start_index`. |
| **Retrieval method** | BM25 (explain TF saturation `k1` and length normalisation `b`), file path in the indexed text, identifier splitting, English stopwords. |
| **Performance analysis** | A recall table at k = 1/3/5/10 for docs and code, indexing time, search time for 200 questions, and **the effect of chunk size** (the subject asks for it: run with 2000, 1000 and 500). |
| **Design decisions** | Why BM25 over TF-IDF, why bm25s, why langchain splitters, why a 6000-character context budget, why greedy decoding and `enable_thinking=False`. |
| **Challenges faced** | Chunks over 2000 because of comments between AST nodes; decorators cut off; the infinite loop on long lines; identifiers not matching paraphrased questions; offsets with `newline=""`. |
| **Example usage** | The 4 pipeline commands from subject §VI.7.2, with sample output. |

### BM25 in one paragraph (for the README and the defense)
> For each query word, BM25 scores a chunk higher when the word appears
> often in it (TF), but with diminishing returns (`k1`, about 1.5); higher when
> the word is rare in the corpus (IDF); and lower when the chunk is very
> long (`b`, about 0.75). It's TF-IDF with saturation and length normalisation,
> which is why it beats plain TF-IDF on chunks of uneven length.

---

## Task 9: the full evaluator run (30 min)

Run what the evaluator runs, from a clean state:
```bash
make clean
```
```bash
uv sync
```
```bash
uv run python -m src index --max_chunk_size 2000
```
```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```
```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```
```bash
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
```bash
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
```
```bash
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
```bash
make lint
```
If you have the moulinette binary, score with it as well:
```bash
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
```

**Final checklist:**
- [ ] `uv sync` works with no warning
- [ ] `index` takes under 5 min
- [ ] `search_dataset` on 200 questions takes under 90 s
- [ ] docs R@5 ≥ 0.80, code R@5 ≥ 0.50
- [ ] every retrieved source is ≤ 2000 characters
- [ ] `file_path` starts with `data/raw/vllm-0.10.1/`
- [ ] `answer` and `answer_dataset` produce valid JSON with real answers
- [ ] every edge case in Task 6 prints an error, never a traceback
- [ ] `make lint` gives 0 errors
- [ ] the README has all the required sections and the italic first line
- [ ] git has no data, no model weights, no outputs, no `.env`

---

## Task 10: defense prep (keep for later)

### Questions you should be able to answer
1. **Why must offsets be exact?** The moulinette compares `file_path`
   verbatim and checks overlap on character ranges.
2. **Why `newline=""` when reading files?** Without it, Python turns
   `\r\n` into `\n`, so your offsets drift away from the grader's.
3. **What is IoU 0.05?** Intersection over union of the two ranges must be
   ≥ 5%. That's a low bar: covering the right area of the right file is enough.
4. **Why BM25 and not TF-IDF?** Saturation (`k1`) and length normalisation (`b`).
5. **Why split identifiers?** Questions paraphrase (`load LoRA adapter`),
   while code uses `load_lora_adapter`.
6. **Why put the path in the indexed text?** File names carry the topic
   (`lora.md`, `quantization/`).
7. **How do the chunkers differ?** See Task 8: the separator lists.
8. **What happens with `--max_chunk_size 50`?** It still works:
   545k chunks, 0 over the limit, no hang. Recall drops because chunks are
   too small to hold an answer.
9. **Why `enable_thinking=False`?** Qwen3's thinking mode is slow and
   puts `<think>` text into the answer.

### Recode practice: likely "small change" requests
- "Add `.rst` files to the index." → the extension list in `__main__.py`.
- "Return only results with score > X." → the filter in `retrieve_sources`.
- "Add the score to the search output." → extend the model, then print it.
- "Change the prompt to answer in one sentence." → `build_prompt`.
- "Make Python chunks also split on `async def`." → pass your own
  `separators=[...]` to the splitter.

### ⚠️ Langchain decision (decide before the defense)
The subject says you must **implement** two chunking strategies, and
libraries are allowed. A strict reviewer might still want your own code. Your
own AST chunker (all bugs fixed, fully tested) is in git history:
```bash
git show 8922f4f:src/indexer/chunker.py
```
Bringing it back for Python removes that risk. Either way, be able to explain
every separator the splitter uses.
