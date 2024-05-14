#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, warnings, re, itertools, json

import pysrt
import spacy
import openai

# from joblib import Memory
import tiktoken
import spacy
import math

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

# cachedir = "./cache_dir"  # Directory to store joblib cache
# memory = Memory(cachedir, verbose=0)


openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()


# @memory.cache
def get_completion(
    prompt,
    system_message="You are a helpful assistant.",
    model="gpt-4-turbo",
    temperature=0,
    JSON_mode=False,
):
    if JSON_mode:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )

        return response
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )

        return response


def get_completion_content(
    prompt,
    system_message="You are a helpful assistant.",
    model="gpt-4-0125-preview",
    temperature=0,
    JSON_mode=False,
):
    completion = get_completion(prompt, system_message, model, temperature, JSON_mode)
    return completion.choices[0].message.content


def one_chunk_initial_translation(source_lang, target_lang, source_text):
    """Use an LLM to translate the entire text as one chunk."""

    system_message = (
        f"You are an expert language translator, specializing in {source_lang} to {target_lang} translation."
    )

    translation_prompt = f"""You task is provide a professional translation of a text from {source_lang} to {target_lang}.

Translate the text below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT> and output the translation.
Do not output anything other the translation.

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>
"""
    prompt = translation_prompt.format(source_text=source_text)

    translation = get_completion_content(prompt, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(source_lang, target_lang, source_text, translation1):
    """Use an LLM to reflect on the translation, treating the entire text as one chunk."""

    system_message = (
        f"You are an expert language translator and mentor, specializing in {source_lang} to {target_lang} translation."
    )

    reflection_prompt = """Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for the translation.

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

    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation1=translation1,
    )
    reflection = get_completion_content(prompt, system_message=system_message)

    return reflection


def one_chunk_improve_translation(source_lang, target_lang, source_text, translation1, reflection):
    """Use the reflection to improve the translation, treating the entire text as one chunk."""

    system_message = (
        f"You are an expert language translator, specializing in {source_lang} to {target_lang} translation."
    )

    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, taking into
account a set of expert suggestions and constructive critisms.

The source text, initial translation, and expert suggestions, delimited by XML tags <SOURCE_TEXT>, <TRANSLATION> and <EXPERT_SUGGESTIONS> are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's \
(i) accuracy (by correcting errors of addition, mistranslation, omission, untranslated text),
(ii) fluency (grammar, inconsistency, punctuation, register, spelling), \
(iii) style (fix awkward wording),
(iv) terminology (inappropriate for context, inconsistent use), or \
(v) other errors. Output the list of suggestions in JSON, using the key "suggestions".

Output the new translation, and nothing else."""

    prompt = improvement_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation1=translation1,
        reflection=reflection,
    )
    translation2 = get_completion_content(prompt, system_message)

    return translation2


def one_chunk_translate_text(source_lang, target_lang, source_text):
    """Get a first translation, reflect on it, then output an improved translation. Treat the text to be translated as one chunk."""
    translation1 = one_chunk_initial_translation(source_lang, target_lang, source_text)
    # print(f"-------\nTranslation1: {translation1}\n")
    reflection = one_chunk_reflect_on_translation(source_lang, target_lang, source_text, translation1)
    # print(f"-------\nReflection: {reflection}\n")
    translation2 = one_chunk_improve_translation(source_lang, target_lang, source_text, translation1, reflection)
    # print(f"-------\nTranslation2: {translation2}\n")

    return translation2


english_model = None  # spacy.load("en_core_web_sm")


def find_sentence_starts(text):
    global english_model
    if english_model is None:
        english_model = spacy.load("en_core_web_sm")  # load the english model

    doc = english_model(text)

    # To store the indices of the first character of each sentence
    sentence_starts = []

    # Iterate over the sentences
    for sent in doc.sents:
        # Find the start index of the first character of each sentence
        start_index = sent.start_char
        sentence_starts.append(start_index)

    return sentence_starts


# In[5]:


def num_tokens_in_string(str, encoding_name="cl100k_base"):
    """Number of tokens. Default to using most commonly used encoder, cl100k_base (used by GPT-4)"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(str))
    return num_tokens


# Horribly inefficient linear search, but shouldn't matter
def index_of_closest_value(values, target_value):
    """Given a list of values and a specific target_value, find the index of the closest value in the list. (Inefficient implementation using linear search.)"""

    closest_value = values[0]
    index_of_closest_value = 0
    min_diff = abs(target_value - closest_value)

    for i in range(1, len(values)):  # Start from the second element
        val = values[i]
        diff = abs(target_value - val)
        if diff < min_diff:
            min_diff = diff
            closest_value = val
            index_of_closest_value = i

    return index_of_closest_value


# In[6]:


def multichunk_initial_translation(source_lang, target_lang, source_text_chunks):
    system_message = (
        f"You are an expert language translator, specializing in {source_lang} to {target_lang} translation."
    )

    translation_prompt = """You task is provide a professional translation from {source_lang} to {target_lang} of PART of a text.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other the translation of the indicated part of the text.

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
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
        )
        # print(f"-------------\n{prompt}")

        translation = get_completion_content(prompt, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks


def multichunk_reflect_on_translation(source_lang, target_lang, source_text_chunks, translation1_chunks):
    system_message = (
        f"You are an expert language translator and mentor, specializing in {source_lang} to {target_lang} translation."
    )

    reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.

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
{translation1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:
(i) accuracy (by correcting errors of addition, mistranslation, omission, untranslated text),
(ii) fluency (grammar, inconsistency, punctuation, register, spelling),
(iii) style (fix awkward wording),
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation."""

    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = reflection_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation1_chunk=translation1_chunks[i],
        )
        # print(f"-------------\n{prompt}")

        reflection = get_completion_content(prompt, system_message=system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks


def multichunk_improve_translation(
    source_lang,
    target_lang,
    source_text_chunks,
    translation1_chunks,
    reflection_chunks,
):
    system_message = (
        f"You are an expert language translator, specializing in {source_lang} to {target_lang} translation."
    )

    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, taking into
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
{translation1_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, is as follows:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's
(i) accuracy (by correcting errors of addition, mistranslation, omission, untranslated text),
(ii) fluency (grammar, inconsistency, punctuation, register, spelling),
(iii) style (fix awkward wording),
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors. Output the list of suggestions in JSON, using the key "suggestions".

Output the new translation of the indicated part, and nothing else."""

    translation2_chunks = []
    for i in range(len(source_text_chunks)):
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
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation1_chunk=translation1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )
        # print(f"-------------\n{prompt}")

        translation2 = get_completion_content(prompt, system_message=system_message)
        translation2_chunks.append(translation2)

    return translation2_chunks


def multichunk_translation(source_lang, target_lang, source_text_chunks):
    translation1_chunks = multichunk_initial_translation("English", "Spanish", source_text_chunks)
    # for t in translation1_chunks:
    #    print(t + "\n----\n")

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang, target_lang, source_text_chunks, translation1_chunks
    )
    # for t in reflection_chunks:
    #    print(t + "\n----\n")

    translation2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation1_chunks,
        reflection_chunks,
    )

    return translation2_chunks


# In[7]:


MAX_TOKENS_PER_CHUNK = 1000  # if text is more than this many tokens, we'll break it up into
# discrete chunks to translate one chunk at a time


def translate(source_lang, target_lang, source_text):
    """Translate the source_text from source_lang to target_lang."""

    global MAX_TOKENS_PER_CHUNK

    num_tokens_in_text = num_tokens_in_string(source_text)

    if num_tokens_in_text < MAX_TOKENS_PER_CHUNK:
        print(f"Translating text as single chunk")

        return one_chunk_translate_text(source_lang, target_lang, source_text)

    # We've implemented a sentence splitter only for English, so if doing multi-chunk,
    # make sure the source language is English.
    if source_lang != "English":
        sys.error(
            "Sorry, only English source language supported for now for " "translation for long (multi-chunk) texts."
        )

    potential_breakpoints = find_sentence_starts(
        source_text
    )  # use start of sentences as potential places to break up the text into chunks
    num_sentences = len(potential_breakpoints)
    potential_breakpoints.append(len(source_text))

    tokens_in_sentence = []  # tokens_in_sentence[i] is the number of tokens in the i-th sentence
    for i in range(num_sentences):
        start_index = potential_breakpoints[i]
        end_index = potential_breakpoints[i + 1]
        tokens_in_sentence.append(num_tokens_in_string(source_text[start_index:end_index]))

    # Look at the total number of tokens, and MAX_TOKENS_PER_CHUNK to figure out how many chunks we need
    total_tokens = sum(tokens_in_sentence)  # should be similar to num_tokens_in_text above
    num_chunks = math.ceil(float(total_tokens) / MAX_TOKENS_PER_CHUNK)
    print(f"Translating text as {num_chunks} chunks")

    # The location of the breakpoints if we chopped the text into num_chunks equal-size chunks (equal number of tokens)
    desired_length_per_chunk = float(total_tokens) / num_chunks
    desired_breakpoints = [i * desired_length_per_chunk for i in range(num_chunks)]
    desired_breakpoints.append(total_tokens)

    # Pick the specific places where we'll break up the text into num_chunks
    cum_tokens_count = [0] + list(itertools.accumulate(tokens_in_sentence))
    actual_breakpoints = []
    for i in desired_breakpoints:
        actual_breakpoints.append(index_of_closest_value(cum_tokens_count, i))

    # print(actual_breakpoints)
    # print([cum_tokens_count[i] for i in actual_breakpoints])

    source_text_chunks = []
    for i in range(num_chunks):
        source_text_chunks.append(
            source_text[potential_breakpoints[actual_breakpoints[i]] : potential_breakpoints[actual_breakpoints[i + 1]]]
        )

    translation2_chunks = multichunk_translation(source_lang, target_lang, source_text_chunks)

    return "".join(translation2_chunks)
