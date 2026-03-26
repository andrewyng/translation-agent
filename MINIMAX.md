# Translation Agent — MiniMax Backend

This variant replaces the default OpenAI backend with [MiniMax](https://www.minimaxi.com/), using the same three-step reflection workflow: **Initial translation → Reflection → Improved translation**.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements-minimax.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root (copy from the example below):

```bash
MINIMAX_API_KEY=your_api_key_here
MINIMAX_BASE_URL=https://api.minimaxi.com/v1   # default, can be omitted
TRANS_MODEL=MiniMax-M2.7                        # default, can be omitted
MAX_TOKENS=1000                                 # max tokens per chunk, default 1000
TEMPERATURE=1.0                                 # must be in (0, 1] for MiniMax
```

> Get your MiniMax API key at the [MiniMax open platform](https://www.minimaxi.com/).

---

## CLI Usage

```
python translate.py -t <target_language> [options] [text]
```

### Options

| Flag | Description |
|------|-------------|
| `-s`, `--source` | Source language (default: `English`) |
| `-t`, `--target` | **Required.** Target language (e.g. `Chinese`, `Spanish`) |
| `-c`, `--country` | Country for regional dialect (e.g. `Mexico`, `Taiwan`) |
| `-f`, `--file` | Input file path (UTF-8) |
| `-o`, `--output` | Output file path (default: stdout) |
| `-m`, `--max-tokens` | Max tokens per chunk (default: `1000`) |
| `-v`, `--verbose` | Show progress info (model, chunk count, etc.) |

### Examples

```bash
# Translate a string directly
python translate.py -s English -t Chinese "Hello, world!"

# Translate a file
python translate.py -s English -t "Simplified Chinese" -f input.txt

# Translate from stdin
echo "Hello" | python translate.py -s English -t Chinese

# With regional dialect
python translate.py -s English -t Spanish -c Mexico "Hello, how are you?"

# Save output to file
python translate.py -s English -t Chinese -f input.txt -o output.txt

# Verbose mode (shows chunk info and model)
python translate.py -s English -t Chinese -v "Hello world"
```

---

## Python API

You can also import `translate` directly in your own scripts:

```python
from translate import translate

result = translate(
    source_lang="English",
    target_lang="Chinese",
    source_text="Your text here.",
    country="",        # optional: e.g. "Taiwan" for regional style
    max_tokens=1000,   # optional: chunk size in tokens
    verbose=False,     # optional: print progress to stderr
)
print(result)
```

---

## How It Works

For each chunk of text, the agent runs three LLM calls:

1. **Initial translation** — produce a first-pass translation.
2. **Reflection** — critique the translation for accuracy, fluency, style, and terminology.
3. **Improved translation** — revise the translation using the reflection as guidance.

Long texts are automatically split into chunks and translated in context (each chunk sees the surrounding text as context).

---

## Differences from the Default Backend

| | Default (OpenAI) | This version (MiniMax) |
|---|---|---|
| Client | `openai` SDK | `openai` SDK (MiniMax-compatible endpoint) |
| Default model | `gpt-4-turbo` | `MiniMax-M2.7` |
| Temperature range | `[0, 2]` | `(0, 1]` (MiniMax requirement) |
| Extra dependencies | `langchain` | None (custom splitter built-in) |
| API key env var | `OPENAI_API_KEY` | `MINIMAX_API_KEY` |
