# a monkey patch to use llama-index completion
import os
import time
from functools import wraps
from threading import Lock
from typing import Union
import src.translation_agent.utils as utils

from llama_index.llms.groq import Groq
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

RPM = 60

# Add your LLMs here
def model_load(
        endpoint: str,
        model: str,
        api_key: str = None,
        context_window: int = 4096,
        num_output: int = 512,
        rpm: int = RPM,
):
    if endpoint == "Groq":
        llm = Groq(
            model=model,
            api_key=api_key if api_key else os.getenv("GROQ_API_KEY"),
        )
    elif endpoint == "Cohere":
        llm = Cohere(
            model=model,
            api_key=api_key if api_key else os.getenv("COHERE_API_KEY"),
        )
    elif endpoint == "OpenAI":
        llm = OpenAI(
            model=model,
            api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"),
        )
    elif endpoint == "TogetherAI":
        llm = TogetherLLM(
            model=model,
            api_key=api_key if api_key else os.getenv("TOGETHER_API_KEY"),
        )
    elif endpoint == "Ollama":
        llm = Ollama(
            model=model,
            request_timeout=120.0)
    elif endpoint == "Huggingface":
        llm = HuggingFaceInferenceAPI(
            model_name=model,
            token=api_key if api_key else os.getenv("HF_TOKEN"),
            task="text-generation",
        )

    global RPM
    RPM = rpm

    Settings.llm = llm
    # maximum input size to the LLM
    Settings.context_window = context_window

    # number of tokens reserved for text generation.
    Settings.num_output = num_output

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
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> Union[str, dict]:
        """
            Generate a completion using the OpenAI API.

        Args:
            prompt (str): The user's prompt or query.
            system_message (str, optional): The system message to set the context for the assistant.
                Defaults to "You are a helpful assistant.".
            temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
                Defaults to 0.3.
            json_mode (bool, optional): Whether to return the response in JSON format.
                Defaults to False.

        Returns:
            Union[str, dict]: The generated completion.
                If json_mode is True, returns the complete API response as a dictionary.
                If json_mode is False, returns the generated text as a string.
        """
        llm = Settings.llm
        if llm.class_name() == "HuggingFaceInferenceAPI":
            llm.system_prompt = system_message
            messages = [
                ChatMessage(
                    role="user", content=prompt),
            ]

            response = llm.chat(
                messages=messages,
                temperature=temperature,
            )
            return response.message.content
        else:
            messages = [
                ChatMessage(
                    role="system", content=system_message),
                ChatMessage(
                    role="user", content=prompt),
            ]

            if json_mode:
                response = llm.chat(
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                return response.message.content
            else:
                response = llm.chat(
                    temperature=temperature,
                    messages=messages,
                )
                return response.message.content

utils.get_completion = get_completion

one_chunk_initial_translation = utils.one_chunk_initial_translation
one_chunk_reflect_on_translation = utils.one_chunk_reflect_on_translation
one_chunk_improve_translation = utils.one_chunk_improve_translation
one_chunk_translate_text = utils.one_chunk_translate_text
num_tokens_in_string = utils.num_tokens_in_string
multichunk_initial_translation = utils.multichunk_initial_translation
multichunk_reflect_on_translation = utils.multichunk_reflect_on_translation
multichunk_improve_translation = utils.multichunk_improve_translation
multichunk_translation = utils.multichunk_translation
calculate_chunk_size =utils.calculate_chunk_size