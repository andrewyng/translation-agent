# Translation Agent: Agentic translation using reflection workflow

This is Python-based demonstration of a reflection agentic workflow for machine translation. The main steps are:
1. Prompt an LLM to translate a text from {source_language} to {target_language} 
2. Reflect on the translation to come up with constructive suggestions for improving it
3. Use the suggestions to improve the translation 


## Customizability 

By using an LLM as the heart of the translation engine, this system is highly steerable. For example, by changing the prompts, 
you can modify its output's style (formal/informal; how to handle special terms like names and acronyms; have it use specific regional
dialects; etc.) more easily than traditional machine translation systems. 

This is not mature software. But we think agentic translation is a promising direction for machine translation, and hope that
this open source implementation will spur further work. Comments and suggestions for how to improve this are also very welcome! 

## Performance 

[[To be added,]] 

## Getting Started

To get started with Translation Agent, follow these steps:

### Installation:

- The Poetry package manager is required (and recommended)
- A .env file with a OPENAI_API_KEY is required to run the workflow. See the .env.sample file as an example.

```bash
git clone https://github.com/andrewyng/translation-agent.git
pip install poetry
cd translation-agent
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

Translation Agent is released under the **MIT License**. You are free to use, modify, and distribute the code 
for both commercial and non-commercial purposes.


