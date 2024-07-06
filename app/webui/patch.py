# a monkey patch for completion
import os
import time
from functools import wraps
from threading import Lock
from typing import Union
import translation_agent.utils as Utils
from groq import Groq
from together import Together
from ollama import Client
import gradio as gr

RPM = 60
MODEL = ""
TEMPERATURE = 0.3
JS_MODE = False
ENDPOINT = ""
client = Utils.client

# Add your LLMs here
def model_load(
        endpoint: str,
        model: str,
        api_key: str = None,
        temperature: float = TEMPERATURE,
        rpm: int = RPM,
        js_mode: bool = JS_MODE,

):
    global client, RPM, MODEL, TEMPERATURE, JS_MODE, ENDPOINT
    ENDPOINT = endpoint
    RPM = rpm
    MODEL = model
    TEMPERATURE = temperature
    JS_MODE = js_mode

    if endpoint == "Groq":
        client = Groq(api_key=api_key if api_key else os.getenv("GROQ_API_KEY"))
    elif endpoint == "TogetherAI":
        client = Together(api_key=api_key if api_key else os.getenv("TOGETHER_API_KEY"))
    elif endpoint == "Ollama":
        client = Client(host='http://localhost:11434')
    else:
        client = Utils.openai.OpenAI(api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"))

def rate_limit(get_max_per_minute):
    def decorator(func):
        lock = Lock()
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                max_per_minute = get_max_per_minute()
                min_interval = 60.0 / max_per_minute
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed

                if left_to_wait > 0:
                    time.sleep(left_to_wait)

                ret = func(*args, **kwargs)
                last_called[0] = time.time()
                return ret
        return wrapper
    return decorator

@rate_limit(lambda: RPM)
def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-4-turbo",
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gpt-4-turbo".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    model = MODEL
    temperature = TEMPERATURE
    json_mode = JS_MODE

    if ENDPOINT == "Ollama":
        if json_mode:
            try:
                response = client.chat(
                    model=model,
                    format="json",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    options = {
                        "temperature": TEMPERATURE,
                        "top_p": 1.0
                    },
                )
                print(response)
                return response['message']['content']
            except Exception as e:
                raise gr.Error(f"An unexpected error occurred: {e}")
        else:
            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    options = {
                        "temperature": TEMPERATURE,
                        "top_p": 1.0
                    },
                )
                return response['message']['content']
            except Exception as e:
                raise gr.Error(f"An unexpected error occurred: {e}")
    else:
        if json_mode:
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    top_p=1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                raise gr.Error(f"An unexpected error occurred: {e}")
        else:
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    top_p=1,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                raise gr.Error(f"An unexpected error occurred: {e}")

Utils.get_completion = get_completion

one_chunk_initial_translation = Utils.one_chunk_initial_translation
one_chunk_reflect_on_translation = Utils.one_chunk_reflect_on_translation
one_chunk_improve_translation = Utils.one_chunk_improve_translation
one_chunk_translate_text = Utils.one_chunk_translate_text
num_tokens_in_string = Utils.num_tokens_in_string
multichunk_initial_translation = Utils.multichunk_initial_translation
multichunk_reflect_on_translation = Utils.multichunk_reflect_on_translation
multichunk_improve_translation = Utils.multichunk_improve_translation
multichunk_translation = Utils.multichunk_translation
calculate_chunk_size =Utils.calculate_chunk_size