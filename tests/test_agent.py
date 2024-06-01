import json
import os
from unittest.mock import patch

import openai
import pytest
import spacy
from dotenv import load_dotenv
from translation_agent.util import find_sentence_starts
from translation_agent.util import get_completion
from translation_agent.util import num_tokens_in_string
from translation_agent.util import one_chunk_improve_translation
from translation_agent.util import one_chunk_initial_translation
from translation_agent.util import one_chunk_reflect_on_translation
from translation_agent.util import one_chunk_translate_text


load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_get_completion_json_mode_api_call():
    # Set up the test data
    prompt = "What is the capital of France in json?"
    system_message = "You are a helpful assistant."
    model = "gpt-4-turbo"
    temperature = 0
    json_mode = True

    # Call the function with JSON_mode=True
    result = get_completion(
        prompt, system_message, model, temperature, json_mode
    )

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
    json_mode = False

    # Call the function with JSON_mode=False
    result = get_completion(
        prompt, system_message, model, temperature, json_mode
    )

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
    with patch(
        "translation_agent.util.get_completion_content"
    ) as mock_get_completion_content:
        mock_get_completion_content.return_value = expected_translation

        # Call the function with test data
        translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        # Assert the expected translation is returned
        assert translation == expected_translation

        # Assert the get_completion_content function was called with the correct arguments
        expected_system_message = f"You are an expert language translator, specializing in {source_lang} to {target_lang} translation."
        expected_prompt = f"""You task is provide a professional translation of a text from {source_lang} to {target_lang}.

Translate the text below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT> and output the translation.
Do not output anything other the translation.

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>
"""
        mock_get_completion_content.assert_called_once_with(
            expected_prompt, system_message=expected_system_message
        )


def test_one_chunk_reflect_on_translation():
    # Define test data
    source_lang = "English"
    target_lang = "Spanish"
    source_text = "This is a sample source text."
    translation1 = "Este es un texto de origen de muestra."

    # Define the expected reflection
    expected_reflection = "The translation is accurate and conveys the meaning of the source text well. However, here are a few suggestions for improvement:\n\n1. Consider using 'texto fuente' instead of 'texto de origen' for a more natural translation of 'source text'.\n2. Add a definite article before 'texto fuente' to improve fluency: 'Este es un texto fuente de muestra.'\n3. If the context allows, you could also use 'texto de ejemplo' as an alternative translation for 'sample text'."

    # Mock the get_completion_content function
    with patch(
        "translation_agent.util.get_completion_content"
    ) as mock_get_completion_content:
        mock_get_completion_content.return_value = expected_reflection

        # Call the function with test data
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, translation1
        )

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
        mock_get_completion_content.assert_called_once_with(
            expected_prompt, system_message=expected_system_message
        )


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
def test_one_chunk_improve_translation(
    mock_get_completion_content, example_data
):
    # Set up the mock return value for get_completion_content
    mock_get_completion_content.return_value = (
        "Esta es una traducción de ejemplo mejorada."
    )

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

    mock_get_completion_content.assert_called_once_with(
        expected_prompt, expected_system_message
    )


def test_one_chunk_translate_text(mocker):
    # Define test data
    source_lang = "en"
    target_lang = "es"
    source_text = "Hello, how are you?"
    translation1 = "Hola, ¿cómo estás?"
    reflection = "The translation looks good, but it could be more formal."
    translation2 = "Hola, ¿cómo está usted?"

    # Mock the helper functions
    mock_initial_translation = mocker.patch(
        "translation_agent.util.one_chunk_initial_translation",
        return_value=translation1,
    )
    mock_reflect_on_translation = mocker.patch(
        "translation_agent.util.one_chunk_reflect_on_translation",
        return_value=reflection,
    )
    mock_improve_translation = mocker.patch(
        "translation_agent.util.one_chunk_improve_translation",
        return_value=translation2,
    )

    # Call the function being tested
    result = one_chunk_translate_text(source_lang, target_lang, source_text)

    # Assert the expected result
    assert result == translation2

    # Assert that the helper functions were called with the correct arguments
    mock_initial_translation.assert_called_once_with(
        source_lang, target_lang, source_text
    )
    mock_reflect_on_translation.assert_called_once_with(
        source_lang, target_lang, source_text, translation1
    )
    mock_improve_translation.assert_called_once_with(
        source_lang, target_lang, source_text, translation1, reflection
    )


@pytest.fixture(scope="module")
def english_model():
    return spacy.load("en_core_web_sm")


def test_find_sentence_starts(english_model):
    # Test case 1: Single sentence
    text1 = "This is a single sentence."
    expected_output1 = [0]
    assert find_sentence_starts(text1) == expected_output1

    # Test case 2: Multiple sentences
    text2 = "This is the first sentence. This is the second sentence. And this is the third sentence."
    expected_output2 = [0, 28, 57]
    assert find_sentence_starts(text2) == expected_output2

    # Test case 3: Empty string
    text3 = ""
    expected_output3 = []
    assert find_sentence_starts(text3) == expected_output3

    # Test case 4: Text with punctuation
    text4 = "Hello, how are you? I'm doing fine! Thanks for asking."
    expected_output4 = [0, 20, 36]
    assert find_sentence_starts(text4) == expected_output4

    # Test case 5: Text with multiple spaces and newlines
    text5 = "  This is a sentence.   \n\nAnother sentence here. \n One more sentence.  "
    expected_output5 = [0, 26, 51]
    assert find_sentence_starts(text5) == expected_output5


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


# def test_invalid_encoding():
#     # Test case 7: Invalid encoding name
#     with pytest.raises(KeyError):
#         num_tokens_in_string("Hello, world!", encoding_name="invalid_encoding")


# result = get_completion(prompt="this is a test, reply with success")
# breakpoint()
