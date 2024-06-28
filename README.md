# Translation Agent: Agentic translation using reflection workflow

This is a Python demonstration of a reflection agentic workflow for machine translation. The main steps are:
1. Prompt an LLM to translate a text from `source_language` to `target_language`;
2. Have the LLM reflect on the translation to come up with constructive suggestions for improving it;
3. Use the suggestions to improve the translation.

## Customizability

By using an LLM as the heart of the translation engine, this system is highly steerable. For example, by changing the prompts, it is easier using this workflow than a traditional machine translation (MT) system to:
- Modify the output's style, such as formal/informal.
- Specify how to handle idioms and special terms like names, technical terms, and acronyms. For example, including a glossary in the prompt lets you make sure particular terms (such as open source, H100 or GPU) are translated consistently.
- Specify specific regional use of the language, or specific dialects, to serve a target audience. For example, Spanish spoken in Latin America is different from Spanish spoken in Spain; French spoken in Canada is different from how it is spoken in France.

**This is not mature software**, and is the result of Andrew playing around with translations on weekends the past few months, plus collaborators (Joaquin Dominguez, Nedelina Teneva, John Santerre) helping refactor the code.

According to our evaluations using BLEU score on traditional translation datasets, this workflow is sometimes competitive with, but also sometimes worse than, leading commercial offerings. However, we’ve also occasionally gotten fantastic results (superior to commercial offerings) with this approach. We think this is just a starting point for agentic translations, and that this is a promising direction for translation, with significant headroom for further improvement, which is why we’re releasing this demonstration to encourage more discussion, experimentation, research and open-source contributions.

If agentic translations can generate better results than traditional architectures (such as an end-to-end transformer that inputs a text and directly outputs a translation) -- which are often faster/cheaper to run than our approach here -- this also provides a mechanism to automatically generate training data (parallel text corpora) that can be used to further train and improve traditional algorithms. (See also [this article in The Batch](https://www.deeplearning.ai/the-batch/building-models-that-learn-from-themselves/) on using LLMs to generate training data.)

Comments and suggestions for how to improve this are very welcome!


## Getting Started

To get started with `translation-agent`, follow these steps:

### Installation:
- The Poetry package manager is required for installation. [Poetry Installation](https://python-poetry.org/docs/#installation) Depending on your environment, this might work:

```bash
pip install poetry
```

- A .env file with a OPENAI_API_KEY is required to run the workflow. See the .env.sample file as an example.
```bash
git clone https://github.com/andrewyng/translation-agent.git
cd translation-agent
poetry install
poetry shell # activates virtual environment
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

## Ideas for extensions

Here are ideas we haven’t had time to experiment with but that we hope the open-source community will:
- **Try other LLMs.** We prototyped this primarily using gpt-4-turbo. We would love for others to experiment with other LLMs as well as other hyperparameter choices and see if some do better than others for particular language pairs.
- **Glossary Creation.** What’s the best way to efficiently build a glossary -- perhaps using an LLM -- of the most important terms that we want translated consistently? For example, many businesses use specialized terms that are not widely used on the internet and that LLMs thus don’t know about, and there are also many terms that can be translated in multiple ways. For example, ”open source” in Spanish can be “Código abierto” or “Fuente abierta”; both are fine, but it’d better to pick one and stick with it for a single document.
- **Glossary Usage and Implementation.** Given a glossary, what’s the best way to include it in the prompt?
- **Evaluations on different languages.** How does its performance vary in different languages? Are there changes that make it work better for particular source or target languages? (Note that for very high levels of performance, which MT systems are approaching, we’re not sure if BLEU is a great metric.) Also, its performance on lower resource languages needs further study.
- **Error analysis.** We’ve found that specifying a language and a country/region (e.g., “Spanish as colloquially spoken in Mexico”) does a pretty good job for our applications. Where does the current approach fall short? We’re also particularly interested in understanding its performance on specialized topics (like law, medicine) or special types of text (like movie subtitles) to understand its limitations.
- **Better evals.** Finally, we think better evaluations (evals) is a huge and important research topic. As with other LLM applications that generate free text, current evaluation metrics appear to fall short. For example, we found that even on documents where our agentic workflow captures context and terminology better, resulting in translations that our human raters prefer over current commercial offerings, evaluation at the sentence level (using the [FLORES](https://github.com/facebookresearch/flores) dataset) resulted in the agentic system scoring lower on BLEU. Can we design better metrics (perhaps using an LLM to evaluate translations?) that capture translation quality at a document level that correlates better with human preferences?

## Related work

A few academic research groups are also starting to look at LLM-based and agentic translation. We think it’s early days for this field!
- *ChatGPT MT: Competitive for High- (but not Low-) Resource Languages*, Robinson et al. (2023), https://arxiv.org/pdf/2309.07423
- *How to Design Translation Prompts for ChatGPT: An Empirical Study*, Gao et al. (2023), https://arxiv.org/pdf/2304.02182v2
- *Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts*, Wu et al. (2024),  https://arxiv.org/pdf/2405.11804
