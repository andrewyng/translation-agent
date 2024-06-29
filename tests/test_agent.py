import json
import os
from unittest.mock import patch

import openai
import pytest
from dotenv import load_dotenv

# from translation_agent.utils import find_sentence_starts
from translation_agent.utils import get_completion
from translation_agent.utils import num_tokens_in_string
from translation_agent.utils import one_chunk_improve_translation
from translation_agent.utils import one_chunk_initial_translation
from translation_agent.utils import one_chunk_reflect_on_translation
from translation_agent.utils import one_chunk_translate_text


load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_get_completion_json_mode_api_call():
    # Set up the test data
    prompt = "What is the capital of France in json?"
    system_message = "You are a helpful assistant."
    model = "gpt-4-turbo"
    temperature = 0.3
    json_mode = True

    # Call the function with JSON_mode=True
    result = get_completion(
        prompt, system_message, model, temperature, json_mode
    )

    # Assert that the result is not None
    assert result is not None

    # Assert that it can be transformed to dictionary (json)
    assert isinstance(json.loads(result), dict)


def test_get_completion_non_json_mode_api_call():
    # Set up the test data
    prompt = "What is the capital of France?"
    system_message = "You are a helpful assistant."
    model = "gpt-4-turbo"
    temperature = 0.3
    json_mode = False

    # Call the function with JSON_mode=False
    result = get_completion(
        prompt, system_message, model, temperature, json_mode
    )

    # Assert that the result is not None
    assert result is not None

    # Assert that the result has the expected response format
    assert isinstance(result, str)


def test_one_chunk_initial_translation():
    # Define test data
    source_lang = "English"
    target_lang = "Spanish"
    source_text = "Hello, how are you?"
    expected_translation = "Hola, ¿cómo estás?"

    # Mock the get_completion_content function
    with patch(
        "translation_agent.utils.get_completion"
    ) as mock_get_completion:
        mock_get_completion.return_value = expected_translation

        # Call the function with test data
        translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        # Assert the expected translation is returned
        assert translation == expected_translation

        # Assert the get_completion_content function was called with the correct arguments
        expected_system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        expected_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

        mock_get_completion.assert_called_once_with(
            expected_prompt, system_message=expected_system_message
        )


def test_one_chunk_reflect_on_translation():
    # Define test data
    source_lang = "English"
    target_lang = "Spanish"
    country = "Mexico"
    source_text = "This is a sample source text."
    translation_1 = "Este es un texto de origen de muestra."

    # Define the expected reflection
    expected_reflection = "The translation is accurate and conveys the meaning of the source text well. However, here are a few suggestions for improvement:\n\n1. Consider using 'texto fuente' instead of 'texto de origen' for a more natural translation of 'source text'.\n2. Add a definite article before 'texto fuente' to improve fluency: 'Este es un texto fuente de muestra.'\n3. If the context allows, you could also use 'texto de ejemplo' as an alternative translation for 'sample text'."

    # Mock the get_completion_content function
    with patch(
        "translation_agent.utils.get_completion"
    ) as mock_get_completion:
        mock_get_completion.return_value = expected_reflection

        # Call the function with test data
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, translation_1, country
        )

        # Assert that the reflection matches the expected reflection
        assert reflection == expected_reflection

        # Assert that the get_completion_content function was called with the correct arguments
        expected_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
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
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
        expected_system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
        mock_get_completion.assert_called_once_with(
            expected_prompt, system_message=expected_system_message
        )


@pytest.fixture
def example_data():
    return {
        "source_lang": "English",
        "target_lang": "Spanish",
        "source_text": "This is a sample source text.",
        "translation_1": "Esta es una traducción de ejemplo.",
        "reflection": "The translation is accurate but could be more fluent.",
    }


@patch("translation_agent.utils.get_completion")
def test_one_chunk_improve_translation(mock_get_completion, example_data):
    # Set up the mock return value for get_completion_content
    mock_get_completion.return_value = (
        "Esta es una traducción de ejemplo mejorada."
    )

    # Call the function with the example data
    result = one_chunk_improve_translation(
        example_data["source_lang"],
        example_data["target_lang"],
        example_data["source_text"],
        example_data["translation_1"],
        example_data["reflection"],
    )

    # Assert that the function returns the expected translation
    assert result == "Esta es una traducción de ejemplo mejorada."

    # Assert that get_completion was called with the expected arguments
    expected_prompt = f"""Your task is to carefully read, then edit, a translation from {example_data["source_lang"]} to {example_data["target_lang"]}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{example_data["source_text"]}
</SOURCE_TEXT>

<TRANSLATION>
{example_data["translation_1"]}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{example_data["reflection"]}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying Spanish grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    expected_system_message = f"You are an expert linguist, specializing in translation editing from English to Spanish."

    mock_get_completion.assert_called_once_with(
        expected_prompt, expected_system_message
    )


def test_one_chunk_translate_text(mocker):
    # Define test data
    source_lang = "English"
    target_lang = "Spanish"
    country = "Mexico"
    source_text = "Hello, how are you?"
    translation_1 = "Hola, ¿cómo estás?"
    reflection = "The translation looks good, but it could be more formal."
    translation2 = "Hola, ¿cómo está usted?"

    # Mock the helper functions
    mock_initial_translation = mocker.patch(
        "translation_agent.utils.one_chunk_initial_translation",
        return_value=translation_1,
    )
    mock_reflect_on_translation = mocker.patch(
        "translation_agent.utils.one_chunk_reflect_on_translation",
        return_value=reflection,
    )
    mock_improve_translation = mocker.patch(
        "translation_agent.utils.one_chunk_improve_translation",
        return_value=translation2,
    )

    # Call the function being tested
    result = one_chunk_translate_text(
        source_lang, target_lang, source_text, country
    )

    # Assert the expected result
    assert result == translation2

    # Assert that the helper functions were called with the correct arguments
    mock_initial_translation.assert_called_once_with(
        source_lang, target_lang, source_text
    )
    mock_reflect_on_translation.assert_called_once_with(
        source_lang, target_lang, source_text, translation_1, country
    )
    mock_improve_translation.assert_called_once_with(
        source_lang, target_lang, source_text, translation_1, reflection
    )


def test_num_tokens_in_string():
    # Test case 1: Empty string
    assert num_tokens_in_string("") == 0

    # Test case 2: Simple string
    assert num_tokens_in_string("Hello, world!") == 4

    # Test case 3: String with special characters
    assert (
        num_tokens_in_string(
            "This is a test string with special characters: !@#$%^&*()"
        )
        == 16
    )

    # Test case 4: String with non-ASCII characters
    assert num_tokens_in_string("Héllò, wörld! 你好，世界！") == 17

    # Test case 5: Long string
    long_string = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
    )
    assert num_tokens_in_string(long_string) == 101

    # Test case 6: Different encoding
    assert (
        num_tokens_in_string("Hello, world!", encoding_name="p50k_base") == 4
    )
