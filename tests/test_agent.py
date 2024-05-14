import pytest
from translation_agent.util import (
    get_completion,
    one_chunk_initial_translation,
    one_chunk_reflect_on_translation,
    one_chunk_improve_translation,
)
from dotenv import load_dotenv
import openai
import os
from icecream import ic
from unittest.mock import MagicMock, patch
import json


load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_get_completion_json_mode_api_call():
    # Set up the test data
    prompt = "What is the capital of France in json?"
    system_message = "You are a helpful assistant."
    model = "gpt-4-turbo"
    temperature = 0
    JSON_mode = True

    # Call the function with JSON_mode=True
    result = get_completion(prompt, system_message, model, temperature, JSON_mode)

    # Assert that the result is not None
    assert result is not None

    # Assert that the result has the expected keys
    assert "id" in dir(result)
    assert "object" in dir(result)
    assert "created" in dir(result)
    assert "model" in dir(result)
    assert "choices" in dir(result)

    # Assert that the result has the expected response format
    assert result.object == "chat.completion"
    assert result.choices[0].message.role == "assistant"
    assert isinstance(json.loads(result.choices[0].message.content), dict)


def test_get_completion_non_json_mode_api_call():
    # Set up the test data
    prompt = "What is the capital of France?"
    system_message = "You are a helpful assistant."
    model = "gpt-4-turbo"
    temperature = 0
    JSON_mode = False

    # Call the function with JSON_mode=False
    result = get_completion(prompt, system_message, model, temperature, JSON_mode)

    # Assert that the result is not None
    assert result is not None

    # Assert that the result has the expected keys
    assert "id" in dir(result)
    assert "created" in dir(result)
    assert "model" in dir(result)
    assert "choices" in dir(result)

    # Assert that the result has the expected response format
    assert result.object == "chat.completion"
    assert result.choices[0].message.role == "assistant"
    assert isinstance(result.choices[0].message.content, str)


def test_one_chunk_initial_translation():
    # Define test data
    source_lang = "English"
    target_lang = "Spanish"
    source_text = "Hello, how are you?"
    expected_translation = "Hola, ¿cómo estás?"

    # Mock the get_completion_content function
    with patch("translation_agent.util.get_completion_content") as mock_get_completion_content:
        mock_get_completion_content.return_value = expected_translation

        # Call the function with test data
        translation = one_chunk_initial_translation(source_lang, target_lang, source_text)

        # Assert the expected translation is returned
        assert translation == expected_translation

        # Assert the get_completion_content function was called with the correct arguments
        expected_system_message = (
            f"You are an expert language translator, specializing in {source_lang} to {target_lang} translation."
        )
        expected_prompt = f"""You task is provide a professional translation of a text from {source_lang} to {target_lang}.

Translate the text below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT> and output the translation.
Do not output anything other the translation.

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>
"""
        mock_get_completion_content.assert_called_once_with(expected_prompt, system_message=expected_system_message)


def test_one_chunk_reflect_on_translation():
    # Define test data
    source_lang = "English"
    target_lang = "Spanish"
    source_text = "This is a sample source text."
    translation1 = "Este es un texto de origen de muestra."

    # Define the expected reflection
    expected_reflection = "The translation is accurate and conveys the meaning of the source text well. However, here are a few suggestions for improvement:\n\n1. Consider using 'texto fuente' instead of 'texto de origen' for a more natural translation of 'source text'.\n2. Add a definite article before 'texto fuente' to improve fluency: 'Este es un texto fuente de muestra.'\n3. If the context allows, you could also use 'texto de ejemplo' as an alternative translation for 'sample text'."

    # Mock the get_completion_content function
    with patch("translation_agent.util.get_completion_content") as mock_get_completion_content:
        mock_get_completion_content.return_value = expected_reflection

        # Call the function with test data
        reflection = one_chunk_reflect_on_translation(source_lang, target_lang, source_text, translation1)

        # Assert that the reflection matches the expected reflection
        assert reflection == expected_reflection

        # Assert that the get_completion_content function was called with the correct arguments
        expected_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for the translation.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT> and <TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \
(i) accuracy (by correcting errors of addition, mistranslation, omission, untranslated text),
(ii) fluency (grammar, inconsistency, punctuation, register, spelling), \
(iii) style (fix awkward wording),
(iv) terminology (inappropriate for context, inconsistent use), or \
(v) other errors.

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation."""
        expected_system_message = f"You are an expert language translator and mentor, specializing in {source_lang} to {target_lang} translation."
        mock_get_completion_content.assert_called_once_with(expected_prompt, system_message=expected_system_message)


@pytest.fixture
def example_data():
    return {
        "source_lang": "English",
        "target_lang": "Spanish",
        "source_text": "This is a sample source text.",
        "translation1": "Esta es una traducción de ejemplo.",
        "reflection": "The translation is accurate but could be more fluent.",
    }


@patch("translation_agent.util.get_completion_content")
def test_one_chunk_improve_translation(mock_get_completion_content, example_data):
    # Set up the mock return value for get_completion_content
    mock_get_completion_content.return_value = "Esta es una traducción de ejemplo mejorada."

    # Call the function with the example data
    result = one_chunk_improve_translation(
        example_data["source_lang"],
        example_data["target_lang"],
        example_data["source_text"],
        example_data["translation1"],
        example_data["reflection"],
    )

    # Assert that the function returns the expected translation
    assert result == "Esta es una traducción de ejemplo mejorada."

    # Assert that get_completion_content was called with the expected arguments
    expected_prompt = """Your task is to carefully read, then improve, a translation from English to Spanish, taking into
account a set of expert suggestions and constructive critisms.

The source text, initial translation, and expert suggestions, delimited by XML tags <SOURCE_TEXT>, <TRANSLATION> and <EXPERT_SUGGESTIONS> are as follows:

<SOURCE_TEXT>
This is a sample source text.
</SOURCE_TEXT>

<TRANSLATION>
Esta es una traducción de ejemplo.
</TRANSLATION>

<EXPERT_SUGGESTIONS>
The translation is accurate but could be more fluent.
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's \
(i) accuracy (by correcting errors of addition, mistranslation, omission, untranslated text),
(ii) fluency (grammar, inconsistency, punctuation, register, spelling), \
(iii) style (fix awkward wording),
(iv) terminology (inappropriate for context, inconsistent use), or \
(v) other errors. Output the list of suggestions in JSON, using the key "suggestions".

Output the new translation, and nothing else."""

    expected_system_message = "You are an expert language translator, specializing in English to Spanish translation."

    mock_get_completion_content.assert_called_once_with(expected_prompt, expected_system_message)


# result = get_completion(prompt="this is a test, reply with success")
# breakpoint()
