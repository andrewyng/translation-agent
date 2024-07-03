import gradio as gr
from simplemma import simple_tokenizer
from difflib import Differ
from icecream import ic
from app.webui.patch import model_load,num_tokens_in_string,one_chunk_initial_translation, one_chunk_reflect_on_translation, one_chunk_improve_translation
from app.webui.patch import calculate_chunk_size, multichunk_initial_translation, multichunk_reflect_on_translation, multichunk_improve_translation

from llama_index.core.node_parser import SentenceSplitter

progress=gr.Progress()

def tokenize(text):
    # Use nltk to tokenize the text
    words = simple_tokenizer(text)
    # Check if the text contains spaces
    if ' ' in text:
        # Create a list of words and spaces
        tokens = []
        for word in words:
            tokens.append(word)
            if not word.startswith("'") and not word.endswith("'"):  # Avoid adding space after punctuation
                tokens.append(' ')  # Add space after each word
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
        if token[0] == '+':
            category = 'added'
        elif token[0] == '-':
            category = 'removed'
        elif token[0] == '?':
            continue  # Ignore the hints line

        highlighted_text.append((word, category))

    return highlighted_text

#modified from src.translaation-agent.utils.tranlsate
def translator(
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
        max_tokens:int = 1000,
):
    """Translate the source_text from source_lang to target_lang."""
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        progress((1,3), desc="First translation...")
        init_translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        progress((2,3), desc="Reflecton...")
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, init_translation, country
        )

        progress((3,3), desc="Second translation...")
        final_translation = one_chunk_improve_translation(
            source_lang, target_lang, source_text, init_translation, reflection
        )

        return init_translation, reflection, final_translation

    else:
        ic("Translating text as multiple chunks")

        progress((1,5), desc="Calculate chunk size...")
        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        #using sentence splitter
        text_parser = SentenceSplitter(
           chunk_size=token_size,
        )

        progress((2,5), desc="Spilt source text...")
        source_text_chunks = text_parser.split_text(source_text)

        progress((3,5), desc="First translation...")
        translation_1_chunks = multichunk_initial_translation(
            source_lang, target_lang, source_text_chunks
        )

        init_translation = "".join(translation_1_chunks)

        progress((4,5), desc="Reflection...")
        reflection_chunks = multichunk_reflect_on_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            country,
        )

        reflection = "".join(reflection_chunks)

        progress((5,5), desc="Second translation...")
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
        model2: str,
        api_key2: str,
        context_window: int,
        num_output: int,
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

        progress((1,3), desc="First translation...")
        init_translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        try:
            model_load(endpoint2, model2, api_key2, context_window, num_output)
        except Exception as e:
            raise gr.Error(f"An unexpected error occurred: {e}")

        progress((2,3), desc="Reflecton...")
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, init_translation, country
        )

        progress((3,3), desc="Second translation...")
        final_translation = one_chunk_improve_translation(
            source_lang, target_lang, source_text, init_translation, reflection
        )

        return init_translation, reflection, final_translation

    else:
        ic("Translating text as multiple chunks")

        progress((1,5), desc="Calculate chunk size...")
        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        #using sentence splitter
        text_parser = SentenceSplitter(
           chunk_size=token_size,
        )

        progress((2,5), desc="Spilt source text...")
        source_text_chunks = text_parser.split_text(source_text)

        progress((3,5), desc="First translation...")
        translation_1_chunks = multichunk_initial_translation(
            source_lang, target_lang, source_text_chunks
        )

        init_translation = "".join(translation_1_chunks)

        try:
            model_load(endpoint2, model2, api_key2, context_window, num_output)
        except Exception as e:
            raise gr.Error(f"An unexpected error occurred: {e}")

        progress((4,5), desc="Reflection...")
        reflection_chunks = multichunk_reflect_on_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            country,
        )

        reflection = "".join(reflection_chunks)

        progress((5,5), desc="Second translation...")
        translation_2_chunks = multichunk_improve_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            reflection_chunks,
        )

        final_translation = "".join(translation_2_chunks)

        return init_translation, reflection, final_translation