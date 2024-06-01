# Translation Agent: Agentic translation using reflection workflow

Translation Agent is a Python-based project that leverages an agentic workflow for machine translation tasks. The repository contains code that utilizes the power of Reflection to enhance the translation process and improve the quality of the generated translations.

## Features

- Agentic Workflow: Translation Agent employs an agentic workflow, which allows for a more intelligent and context-aware approach to machine translation. By incorporating Reflection, the system can analyze and understand the source text more effectively, resulting in more accurate and fluent translations.
- Reflection-based Translation: The core of Translation Agent lies in its use of Reflection, a technique that enables the system to introspect and reason about its own translation process. By reflecting on the intermediate steps and considering the context and meaning of the source text, the system can make informed decisions and generate translations that better capture the intended meaning.
- Language Support: Translation Agent supports a wide range of languages, making it a versatile tool for translating text across different linguistic boundaries. Whether you need to translate between commonly spoken languages or handle less-resourced language pairs, Translation Agent has you covered.
- Customizable Models: The repository provides a flexible framework that allows you to customize and fine-tune the translation models according to your specific requirements. You can experiment with different architectures, training data, and hyperparameters to optimize the translation quality for your use case.
- Easy Integration: Translation Agent is designed to be easily integrated into existing projects and workflows. With a simple and intuitive API, you can seamlessly incorporate machine translation capabilities into your applications, websites, or data pipelines.

## Getting Started

To get started with Translation Agent, follow these steps:

### Installation:

```bash
git clone https://github.com/andrewyng/translation-agent.git
```

- Poetry package manager is required (and recommended)
- A .env file with a OPENAI_API_KEY is required to run the workflow. See the .env.sample file as an example.

Once you are in the repository directory:

```python
poetry install
```

### Usage:

```python
import translation_agent as ta

source_lang, target_lang, country = "English", "Spanish", "Mexico"

translation = ta.translate(source_lang, target_lang, source_text, country)
```

See examples/example_script.py for an example script to try out.

## License

Translation Agent is released under the **MIT License**. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.
