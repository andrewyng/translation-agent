import os
from typing import List
from typing import Union
import re
import openai
import tiktoken
from dotenv import load_dotenv
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


load_dotenv()  # read local .env file
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MAX_TOKENS_PER_CHUNK = (
    int(os.getenv("MAX_TOKENS_PER_CHUNK"))  # if text is more than this many tokens, we'll break it up into
)
TOTAL_MAX_TOKENS = int(os.getenv("TOTAL_MAX_TOKENS"))  # if text is more than this many tokens, we'll break it up into
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS")) # 单次模型最大生成token数
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
DEFAULT_CHUNK_MODEL = os.getenv("DEFAULT_CHUNK_MODEL")
# discrete chunks to translate one chunk at a time
# load default tokenizer
if DEFAULT_CHUNK_MODEL in tiktoken.model.MODEL_TO_ENCODING:
    map_model_name = tiktoken.model.MODEL_TO_ENCODING[DEFAULT_CHUNK_MODEL]
    DEFAULT_TOKENIZER = tiktoken.get_encoding(map_model_name)
else:
    DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_CHUNK_MODEL)


def get_completion(
    prompt: str,
    model: str,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str): The name of the OpenAI model to use for generating the completion.
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    if json_mode:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            timeout=60,
        )
        return response.choices[0].message.content


def one_chunk_initial_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    model: str,
    text_type: str,
    identity_description: str = "You are an overseas study expert",
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.
    Returns:
        str: The translated text.
    """

    system_message = f"{identity_description}, specializing in translation from {source_lang} to {target_lang}."
    # 临时替换, 解决{}问题
    source_text = source_text.replace('{', '{{').replace('}', '}}')
    translation_prompt = f"""This is an {source_lang} to {target_lang} translation for {text_type}, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""
    prompt = translation_prompt.format(source_text=source_text)
    prompt = prompt.replace('{{', '{').replace('}}', '}')

    translation = get_completion(prompt, model=model, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    model: str,
    text_type: str,
    identity_description: str = "You are an overseas study expert",
    country: str = "",
) -> str:
    source_text = source_text.replace('{', '{{').replace('}', '}}')
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.
        country (str): Country specified for target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"{identity_description} in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text for {text_type} and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text for {text_type} and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
    )
    prompt = prompt.replace('{{', '{').replace('}}', '}')
    reflection = get_completion(prompt, model=model, system_message=system_message)
    return reflection


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
    model: str,
    text_type: str,
    identity_description: str = "You are an overseas study expert",
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = f"{identity_description}, specializing in translation editing from {source_lang} to {target_lang}."

    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang} for {text_type}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    translation_2 = get_completion(prompt, model=model, system_message=system_message)

    return translation_2


def one_chunk_translate_text(
    source_lang: str,
    target_lang: str,
    source_text: str,
    model: str,
    text_type: str,
    identity_description: str = "You are an overseas study expert",
    country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.
        country (str): Country specified for target language.
    Returns:
        str: The improved translation of the source text.
    """
    translation_1 = one_chunk_initial_translation(
        source_lang,
        target_lang,
        source_text,
        model,
        text_type=text_type,
        identity_description=identity_description,
    )

    reflection = one_chunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text,
        translation_1,
        model,
        text_type=text_type,
        identity_description=identity_description,
        country=country
    )
    translation_2 = one_chunk_improve_translation(
        source_lang,
        target_lang,
        source_text,
        translation_1,
        reflection,
        model,
        text_type=text_type,
        identity_description=identity_description,
    )

    return translation_2


def num_tokens_in_string(
    input_str: str, encoding_name: str = "cl100k_base", tokenizer: Union[AutoTokenizer, tiktoken.core.Encoding] = None
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        input_str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by GPT-4).
        tokenizer (AutoTokenizer | tiktoken.core.Encoding): for string tokenize

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding(encoding_name)
    num_tokens = len(tokenizer.encode(input_str))
    return num_tokens


def multichunk_initial_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    model: str,
    text_type: str,
    identity_description: str = "You are an overseas study expert",
) -> List[str]:
    """
    Translate a text in multiple chunks from the source language to the target language.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): A list of text chunks to be translated.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.

    Returns:
        List[str]: A list of translated text chunks.
    """

    system_message = f"{identity_description}, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = """Your task is provide a professional translation from {source_lang} to {target_lang} of PART of a text for {text_type}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other than the translation of the indicated part of the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Output only the translation of the portion you are asked to translate, and nothing else.
"""

    translation_chunks = []
    for i in range(len(source_text_chunks)):
        ic("initial translation Translating chunk", i)
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text_type=text_type,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
        )

        translation = get_completion(prompt, model=model, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks


def multichunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    model: str,
    text_type: str,
    identity_description: str = "You are an overseas study expert",
    country: str = "",
) -> List[str]:
    """
    Provides constructive criticism and suggestions for improving a partial translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The translated chunks corresponding to the source text chunks.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.
        country (str): Country specified for target language.
    Returns:
        List[str]: A list of reflections containing suggestions for improving each translated chunk.
    """

    system_message = f"{identity_description} specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang} for {text_type}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang} for {text_type}, and then give constructive criticism and helpful suggestions for improving the translation.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        ic("reflecting on translation chunk", i)
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )
        if country != "":
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                text_type=text_type,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
                country=country,
            )
        else:
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                text_type=text_type,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
            )

        reflection = get_completion(prompt, model=model, system_message=system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks


def multichunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    reflection_chunks: List[str],
    model: str, 
    text_type: str,
    identity_description: str = "You are an overseas study expert",
) -> List[str]:
    """
    Improves the translation of a text from source language to target language by considering expert suggestions.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The initial translation of each chunk.
        reflection_chunks (List[str]): Expert suggestions for improving each translated chunk.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.
    Returns:
        List[str]: The improved translation of each chunk.
    """

    system_message = f"{identity_description}, specializing in translation editing from {source_lang} to {target_lang}."

    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang} for {text_type}, taking into
account a set of expert suggestions and constructive critisms. Below, the source text, initial translation, and expert suggestions are provided.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, is as follows:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation of the indicated part and nothing else."""

    translation_2_chunks = []
    for i in range(len(source_text_chunks)):
        ic("improving translation chunk", i)
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text_type=text_type,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        translation_2 = get_completion(prompt, model=model, system_message=system_message)
        translation_2_chunks.append(translation_2)

    return translation_2_chunks


def multichunk_translation(
    source_lang,
    target_lang,
    source_text_chunks,
    model,
    text_type,
    identity_description: str = "You are an overseas study expert",
    country: str = ""
):
    """
    Improves the translation of multiple text chunks based on the initial translation and reflection.

    Args:
        source_lang (str): The source language of the text chunks.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The list of source text chunks to be translated.
        model (str): The name of the OpenAI model to use for generating the completion.
        text_type (str): The type of text to be translated.
        country (str): Country specified for target language
    Returns:
        List[str]: The list of improved translations for each source text chunk.
    """

    translation_1_chunks = multichunk_initial_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        model,
        text_type=text_type,
        identity_description=identity_description,
    )

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        model,
        text_type=text_type,
        identity_description=identity_description,
        country=country,
    )

    translation_2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
        model,
        text_type=text_type,
        identity_description=identity_description,
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count and token limit.
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.
        The chunk size is determined by dividing the token limit by the number of chunks.
        If there are remaining tokens after dividing the token count by the token limit,
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def translate(
    source_lang,
    target_lang,
    source_text,
    text_type,
    identity_description,
    country,
    model=DEFAULT_MODEL,
    chunk_model=DEFAULT_CHUNK_MODEL,
    max_tokens=MAX_TOKENS_PER_CHUNK,
    total_max_tokens=TOTAL_MAX_TOKENS,
):
    """Translate the source_text from source_lang to target_lang."""
    # remove (top_up) in source_ text
    p = re.compile("(\(top[ -]up\))", flags=re.IGNORECASE)
    source_text = p.sub("", source_text)
    if chunk_model == DEFAULT_CHUNK_MODEL:
        tokenizer = DEFAULT_TOKENIZER
    elif chunk_model in tiktoken.model.MODEL_TO_ENCODING:
        map_name = tiktoken.model.MODEL_TO_ENCODING[DEFAULT_CHUNK_MODEL]
        tokenizer = tiktoken.get_encoding(map_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(chunk_model)

    num_tokens_in_text = num_tokens_in_string(source_text, tokenizer=tokenizer)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        final_translation = one_chunk_translate_text(
            source_lang,
            target_lang,
            source_text,
            model,
            text_type=text_type,
            identity_description=identity_description,
            country=country
        )

        return final_translation
    elif num_tokens_in_text < total_max_tokens:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)
        if chunk_model in tiktoken.model.MODEL_TO_ENCODING:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name=chunk_model,
                chunk_size=token_size,
                chunk_overlap=0,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=token_size,
                chunk_overlap=0,
            )

        source_text_chunks = text_splitter.split_text(source_text)

        translation_2_chunks = multichunk_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            model,
            text_type=text_type,
            identity_description=identity_description,
            country=country
        )

        return "".join(translation_2_chunks)
    else:
        # text is too long, we need to split it into smaller chunks
        ic("Translating long text as multiple chunks")
        split_text_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=total_max_tokens
        )
        ic(split_text_size)
        if chunk_model in tiktoken.model.MODEL_TO_ENCODING:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name=chunk_model,
                chunk_size=split_text_size,
                chunk_overlap=0,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=split_text_size,
                chunk_overlap=0,
            )
        source_text_chunks = text_splitter.split_text(source_text)
        translate_list = []
        for i, chunk in enumerate(source_text_chunks):
            ic(f"Chunk {i + 1}: {chunk}")
            # 再次判断每个chunk的长度
            num_tokens_in_chunk = num_tokens_in_string(chunk, tokenizer=tokenizer)
            ic(num_tokens_in_chunk)
            if num_tokens_in_chunk < max_tokens:
                ic("Translating chunk as single chunk")
                temp_translation = one_chunk_translate_text(
                    source_lang,
                    target_lang,
                    chunk,
                    model,
                    text_type=text_type,
                    identity_description=identity_description,
                    country=country
                )
            else:
                ic("Translating chunk as multiple chunks")
                chunk_size = calculate_chunk_size(
                    token_count=num_tokens_in_chunk, token_limit=max_tokens
                )

                ic(chunk_size)
                if chunk_model in tiktoken.model.MODEL_TO_ENCODING:
                    chunk_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        model_name=chunk_model,
                        chunk_size=chunk_size,
                        chunk_overlap=0,
                    )
                else:
                    chunk_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                        tokenizer=tokenizer,
                        chunk_size=chunk_size,
                        chunk_overlap=0,
                    )

                split_chunks = chunk_splitter.split_text(chunk)

                translation_2_chunks = multichunk_translation(
                    source_lang,
                    target_lang,
                    split_chunks,
                    model,
                    text_type=text_type,
                    identity_description=identity_description,
                    country=country
                )

                temp_translation = "".join(translation_2_chunks)
            translate_list.append(temp_translation)
        return "".join(translate_list)


if __name__ == "__main__":
    # with open("text.txt") as f:
    #     text = f.read()  # noqa: ERA001
    identity_description: str = "Overseas study consulting expert"
    text_list = ["English - Master", "Forestry - Master", "History - Master", "Indigenous Learning"]
    for text in text_list:
        translated_text = translate("English", "Chinese", text, "a major", identity_description,"China")
        print(text, "-->", translated_text)
