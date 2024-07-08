from difflib import Differ

import docx
import gradio as gr
import pymupdf
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from patch import (
    calculate_chunk_size,
    model_load,
    multichunk_improve_translation,
    multichunk_initial_translation,
    multichunk_reflect_on_translation,
    num_tokens_in_string,
    one_chunk_improve_translation,
    one_chunk_initial_translation,
    one_chunk_reflect_on_translation,
)
from simplemma import simple_tokenizer


progress = gr.Progress()


def extract_text(path):
    with open(path) as f:
        file_text = f.read()
    return file_text


def extract_pdf(path):
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_docx(path):
    doc = docx.Document(path)
    data = []
    for paragraph in doc.paragraphs:
        data.append(paragraph.text)
    content = "\n\n".join(data)
    return content


def tokenize(text):
    # Use nltk to tokenize the text
    words = simple_tokenizer(text)
    # Check if the text contains spaces
    if " " in text:
        # Create a list of words and spaces
        tokens = []
        for word in words:
            tokens.append(word)
            if not word.startswith("'") and not word.endswith(
                "'"
            ):  # Avoid adding space after punctuation
                tokens.append(" ")  # Add space after each word
        return tokens[:-1]  # Remove the last space
    else:
        return words


def diff_texts(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    d = Differ()
    diff_result = list(d.compare(tokens1, tokens2))

    highlighted_text = []
    for token in diff_result:
        word = token[2:]
        category = None
        if token[0] == "+":
            category = "added"
        elif token[0] == "-":
            category = "removed"
        elif token[0] == "?":
            continue  # Ignore the hints line

        highlighted_text.append((word, category))

    return highlighted_text


# modified from src.translaation-agent.utils.tranlsate
def translator(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int = 1000,
):
    """Translate the source_text from source_lang to target_lang."""
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        progress((1, 3), desc="First translation...")
        init_translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        progress((2, 3), desc="Reflection...")
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, init_translation, country
        )

        progress((3, 3), desc="Second translation...")
        final_translation = one_chunk_improve_translation(
            source_lang, target_lang, source_text, init_translation, reflection
        )

        return init_translation, reflection, final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        progress((1, 3), desc="First translation...")
        translation_1_chunks = multichunk_initial_translation(
            source_lang, target_lang, source_text_chunks
        )

        init_translation = "".join(translation_1_chunks)

        progress((2, 3), desc="Reflection...")
        reflection_chunks = multichunk_reflect_on_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            country,
        )

        reflection = "".join(reflection_chunks)

        progress((3, 3), desc="Second translation...")
        translation_2_chunks = multichunk_improve_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            reflection_chunks,
        )

        final_translation = "".join(translation_2_chunks)

        return init_translation, reflection, final_translation


def translator_sec(
    endpoint2: str,
    base2: str,
    model2: str,
    api_key2: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int = 1000,
):
    """Translate the source_text from source_lang to target_lang."""
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        progress((1, 3), desc="First translation...")
        init_translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        try:
            model_load(endpoint2, base2, model2, api_key2)
        except Exception as e:
            raise gr.Error(f"An unexpected error occurred: {e}") from e

        progress((2, 3), desc="Reflection...")
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, init_translation, country
        )

        progress((3, 3), desc="Second translation...")
        final_translation = one_chunk_improve_translation(
            source_lang, target_lang, source_text, init_translation, reflection
        )

        return init_translation, reflection, final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        progress((1, 3), desc="First translation...")
        translation_1_chunks = multichunk_initial_translation(
            source_lang, target_lang, source_text_chunks
        )

        init_translation = "".join(translation_1_chunks)

        try:
            model_load(endpoint2, base2, model2, api_key2)
        except Exception as e:
            raise gr.Error(f"An unexpected error occurred: {e}") from e

        progress((2, 3), desc="Reflection...")
        reflection_chunks = multichunk_reflect_on_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            country,
        )

        reflection = "".join(reflection_chunks)

        progress((3, 3), desc="Second translation...")
        translation_2_chunks = multichunk_improve_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            reflection_chunks,
        )

        final_translation = "".join(translation_2_chunks)

        return init_translation, reflection, final_translation
