#!/usr/bin/env python3
"""
Translation Agent - Powered by MiniMax
Three-step reflection workflow: Initial → Reflect → Improve
"""

import os
import sys
import argparse
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
MINIMAX_API_KEY  = os.getenv("MINIMAX_API_KEY", "")
MODEL            = os.getenv("TRANS_MODEL", "MiniMax-M2.7")
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS", "1000"))
TEMPERATURE      = float(os.getenv("TEMPERATURE", "1.0"))  # MiniMax requires (0, 1]

client = OpenAI(
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
)


# ── Token counting (no tiktoken needed — simple char estimate) ─────────────────

def num_tokens(text: str) -> int:
    """Estimate token count. ~4 chars per token is a reasonable approximation."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


# ── Simple recursive text splitter (replaces langchain dependency) ─────────────

def split_text(text: str, chunk_size: int) -> List[str]:
    """
    Recursively split text by paragraph, sentence, then character boundaries.
    Mimics RecursiveCharacterTextSplitter with chunk_overlap=0.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(t: str, seps: List[str]) -> List[str]:
        if not seps:
            # Split by character count
            return [t[i:i+chunk_size] for i in range(0, len(t), chunk_size)]

        sep = seps[0]
        if sep == "":
            return _split(t, [])

        parts = t.split(sep)
        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if num_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single part is too big, recurse with next separator
                if num_tokens(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    return _split(text, separators)


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    if token_count <= token_limit:
        return token_count
    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks
    remaining = token_count % token_limit
    if remaining > 0:
        chunk_size += remaining // num_chunks
    return chunk_size


# ── LLM call ───────────────────────────────────────────────────────────────────

def chat(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Single-chunk translation (3 steps) ─────────────────────────────────────────

def initial_translation(source_lang: str, target_lang: str, text: str) -> str:
    system = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    prompt = (
        f"This is an {source_lang} to {target_lang} translation, please provide the "
        f"{target_lang} translation for this text. "
        f"Do not provide any explanations or text apart from the translation.\n"
        f"{source_lang}: {text}\n\n"
        f"{target_lang}:"
    )
    return chat(system, prompt)


def reflect_on_translation(
    source_lang: str, target_lang: str, source_text: str,
    translation: str, country: str = ""
) -> str:
    system = (
        f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. "
        f"You will be provided with a source text and its translation and your goal is to improve the translation."
    )
    country_clause = (
        f"The final style and tone of the translation should match the style of "
        f"{target_lang} colloquially spoken in {country}.\n\n"
        if country else ""
    )
    prompt = (
        f"Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, "
        f"and then give constructive criticism and helpful suggestions to improve the translation. "
        f"{country_clause}"
        f"\nThe source text and initial translation, delimited by XML tags "
        f"<SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:\n\n"
        f"<SOURCE_TEXT>\n{source_text}\n</SOURCE_TEXT>\n\n"
        f"<TRANSLATION>\n{translation}\n</TRANSLATION>\n\n"
        f"When writing suggestions, pay attention to whether there are ways to improve the translation's\n"
        f"(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n"
        f"(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, "
        f"and ensuring there are no unnecessary repetitions),\n"
        f"(iii) style (by ensuring the translations reflect the style of the source text "
        f"and take into account any cultural context),\n"
        f"(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; "
        f"and by only ensuring you use equivalent idioms {target_lang}).\n\n"
        f"Write a list of specific, helpful and constructive suggestions for improving the translation.\n"
        f"Each suggestion should address one specific part of the translation.\n"
        f"Output only the suggestions and nothing else."
    )
    return chat(system, prompt)


def improve_translation(
    source_lang: str, target_lang: str,
    source_text: str, translation: str, reflection: str
) -> str:
    system = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
    prompt = (
        f"Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, "
        f"taking into account a list of expert suggestions and constructive criticisms.\n\n"
        f"The source text, the initial translation, and the expert linguist suggestions are delimited by "
        f"XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and "
        f"<EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> as follows:\n\n"
        f"<SOURCE_TEXT>\n{source_text}\n</SOURCE_TEXT>\n\n"
        f"<TRANSLATION>\n{translation}\n</TRANSLATION>\n\n"
        f"<EXPERT_SUGGESTIONS>\n{reflection}\n</EXPERT_SUGGESTIONS>\n\n"
        f"Please take into account the expert suggestions when editing the translation. "
        f"Edit the translation by ensuring:\n"
        f"(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n"
        f"(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules "
        f"and ensuring there are no unnecessary repetitions),\n"
        f"(iii) style (by ensuring the translations reflect the style of the source text),\n"
        f"(iv) terminology (inappropriate for context, inconsistent use), or\n"
        f"(v) other errors.\n\n"
        f"Output only the new translation and nothing else."
    )
    return chat(system, prompt)


def translate_chunk(
    source_lang: str, target_lang: str, text: str, country: str = ""
) -> str:
    t1 = initial_translation(source_lang, target_lang, text)
    ref = reflect_on_translation(source_lang, target_lang, text, t1, country)
    t2 = improve_translation(source_lang, target_lang, text, t1, ref)
    return t2


# ── Multi-chunk translation ────────────────────────────────────────────────────

def multichunk_initial_translation(
    source_lang: str, target_lang: str, chunks: List[str]
) -> List[str]:
    system = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    prompt_tpl = (
        "Your task is to provide a professional translation from {source_lang} to {target_lang} of PART of a text.\n\n"
        "The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. "
        "Translate only the part within the source text delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. "
        "You can use the rest of the source text as context, but do not translate any of the other text. "
        "Do not output anything other than the translation of the indicated part of the text.\n\n"
        "<SOURCE_TEXT>\n{tagged_text}\n</SOURCE_TEXT>\n\n"
        "To reiterate, you should translate only this part of the text, shown here again between "
        "<TRANSLATE_THIS> and </TRANSLATE_THIS>:\n"
        "<TRANSLATE_THIS>\n{chunk}\n</TRANSLATE_THIS>\n\n"
        "Output only the translation of the portion you are asked to translate, and nothing else."
    )
    results = []
    for i, chunk in enumerate(chunks):
        tagged = (
            "".join(chunks[:i])
            + "<TRANSLATE_THIS>" + chunk + "</TRANSLATE_THIS>"
            + "".join(chunks[i+1:])
        )
        prompt = prompt_tpl.format(
            source_lang=source_lang, target_lang=target_lang,
            tagged_text=tagged, chunk=chunk
        )
        results.append(chat(system, prompt))
    return results


def multichunk_reflect(
    source_lang: str, target_lang: str,
    chunks: List[str], translations: List[str], country: str = ""
) -> List[str]:
    system = (
        f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. "
        f"You will be provided with a source text and its translation and your goal is to improve the translation."
    )
    country_clause = (
        f"The final style and tone of the translation should match the style of "
        f"{target_lang} colloquially spoken in {country}.\n\n"
        if country else ""
    )
    prompt_tpl = (
        "Your task is to carefully read a source text and part of a translation of that text from "
        "{source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions "
        "for improving the translation.\n"
        + country_clause +
        "\nThe source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, "
        "and the part that has been translated is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> "
        "within the source text. You can use the rest of the source text as context for critiquing "
        "the translated part.\n\n"
        "<SOURCE_TEXT>\n{tagged_text}\n</SOURCE_TEXT>\n\n"
        "To reiterate, only part of the text is being translated, shown here again between "
        "<TRANSLATE_THIS> and </TRANSLATE_THIS>:\n"
        "<TRANSLATE_THIS>\n{chunk}\n</TRANSLATE_THIS>\n\n"
        "The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is:\n"
        "<TRANSLATION>\n{translation}\n</TRANSLATION>\n\n"
        "When writing suggestions, pay attention to whether there are ways to improve the translation's:\n"
        "(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n"
        "(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, "
        "and ensuring there are no unnecessary repetitions),\n"
        "(iii) style (by ensuring the translations reflect the style of the source text "
        "and take into account any cultural context),\n"
        "(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; "
        "and by only ensuring you use equivalent idioms {target_lang}).\n\n"
        "Write a list of specific, helpful and constructive suggestions for improving the translation.\n"
        "Each suggestion should address one specific part of the translation.\n"
        "Output only the suggestions and nothing else."
    )
    results = []
    for i, (chunk, trans) in enumerate(zip(chunks, translations)):
        tagged = (
            "".join(chunks[:i])
            + "<TRANSLATE_THIS>" + chunk + "</TRANSLATE_THIS>"
            + "".join(chunks[i+1:])
        )
        prompt = prompt_tpl.format(
            source_lang=source_lang, target_lang=target_lang,
            tagged_text=tagged, chunk=chunk, translation=trans
        )
        results.append(chat(system, prompt))
    return results


def multichunk_improve(
    source_lang: str, target_lang: str,
    chunks: List[str], translations: List[str], reflections: List[str]
) -> List[str]:
    system = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
    prompt_tpl = (
        "Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, "
        "taking into account a set of expert suggestions and constructive criticisms. "
        "Below, the source text, initial translation, and expert suggestions are provided.\n\n"
        "The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, "
        "and the part that has been translated is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> "
        "within the source text. You can use the rest of the source text as context, but need to provide "
        "a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.\n\n"
        "<SOURCE_TEXT>\n{tagged_text}\n</SOURCE_TEXT>\n\n"
        "To reiterate, only part of the text is being translated, shown here again between "
        "<TRANSLATE_THIS> and </TRANSLATE_THIS>:\n"
        "<TRANSLATE_THIS>\n{chunk}\n</TRANSLATE_THIS>\n\n"
        "The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is:\n"
        "<TRANSLATION>\n{translation}\n</TRANSLATION>\n\n"
        "The expert suggestions, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, are:\n"
        "<EXPERT_SUGGESTIONS>\n{reflection}\n</EXPERT_SUGGESTIONS>\n\n"
        "Taking into account the expert suggestions rewrite the translation to improve it, paying attention to:\n"
        "(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n"
        "(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules "
        "and ensuring there are no unnecessary repetitions),\n"
        "(iii) style (by ensuring the translations reflect the style of the source text),\n"
        "(iv) terminology (inappropriate for context, inconsistent use), or\n"
        "(v) other errors.\n\n"
        "Output only the new translation of the indicated part and nothing else."
    )
    results = []
    for i, (chunk, trans, ref) in enumerate(zip(chunks, translations, reflections)):
        tagged = (
            "".join(chunks[:i])
            + "<TRANSLATE_THIS>" + chunk + "</TRANSLATE_THIS>"
            + "".join(chunks[i+1:])
        )
        prompt = prompt_tpl.format(
            source_lang=source_lang, target_lang=target_lang,
            tagged_text=tagged, chunk=chunk, translation=trans, reflection=ref
        )
        results.append(chat(system, prompt))
    return results


# ── Main translate function ────────────────────────────────────────────────────

def translate(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str = "",
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
    verbose: bool = False,
) -> str:
    """
    Translate source_text from source_lang to target_lang using a 3-step reflection workflow.
    """
    token_count = num_tokens(source_text)

    if verbose:
        print(f"[info] tokens in text: {token_count}", file=sys.stderr)

    if token_count < max_tokens:
        if verbose:
            print("[info] translating as single chunk", file=sys.stderr)
        return translate_chunk(source_lang, target_lang, source_text, country)
    else:
        chunk_size = calculate_chunk_size(token_count, max_tokens)
        chunks = split_text(source_text, chunk_size)
        if verbose:
            print(f"[info] translating as {len(chunks)} chunks (chunk_size≈{chunk_size} tokens)", file=sys.stderr)

        t1_chunks  = multichunk_initial_translation(source_lang, target_lang, chunks)
        ref_chunks = multichunk_reflect(source_lang, target_lang, chunks, t1_chunks, country)
        t2_chunks  = multichunk_improve(source_lang, target_lang, chunks, t1_chunks, ref_chunks)
        return "".join(t2_chunks)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Translation Agent (MiniMax-powered, 3-step reflection workflow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate a string directly
  python translate.py -s English -t Chinese "Hello, world!"

  # Translate a file
  python translate.py -s English -t "Simplified Chinese" -f input.txt

  # Translate from stdin
  echo "Hello" | python translate.py -s English -t Chinese

  # With regional dialect
  python translate.py -s English -t Spanish -c Mexico "Hello, how are you?"

  # Output to file
  python translate.py -s English -t Chinese -f input.txt -o output.txt

  # Verbose mode (shows progress)
  python translate.py -s English -t Chinese -v "Hello world"
        """,
    )
    parser.add_argument("-s", "--source", default="English", help="Source language (default: English)")
    parser.add_argument("-t", "--target", required=True, help="Target language (e.g. Chinese, Spanish)")
    parser.add_argument("-c", "--country", default="", help="Country for regional dialect (e.g. Mexico)")
    parser.add_argument("-f", "--file", help="Input file path (UTF-8)")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("-m", "--max-tokens", type=int, default=MAX_TOKENS_PER_CHUNK,
                        help=f"Max tokens per chunk (default: {MAX_TOKENS_PER_CHUNK})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress info")
    parser.add_argument("text", nargs="?", help="Text to translate (alternatively use -f or stdin)")

    args = parser.parse_args()

    # Get source text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            source_text = f.read()
    elif args.text:
        source_text = args.text
    elif not sys.stdin.isatty():
        source_text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    if not MINIMAX_API_KEY or MINIMAX_API_KEY == "your_api_key_here":
        print("Error: MINIMAX_API_KEY is not set. Copy .env.sample to .env and fill in your key.", file=sys.stderr)
        sys.exit(1)

    if not source_text.strip():
        print("Error: empty input text.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"[info] model: {MODEL}", file=sys.stderr)
        print(f"[info] {args.source} → {args.target}" + (f" ({args.country})" if args.country else ""), file=sys.stderr)

    result = translate(
        source_lang=args.source,
        target_lang=args.target,
        source_text=source_text,
        country=args.country,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        if args.verbose:
            print(f"[info] saved to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
