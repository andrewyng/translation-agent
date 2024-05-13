# Agentic translation using reflection workflow 

Usage: 

Download spaCy (natural language processing package)'s English model:

```bash
python -m spacy download en_core_web_sm
```

Use as follows: 

```python
import translation_agent as ta 

source_lang, target_lang = "English", "Spanish"

translation = ta.translate(source_lang, target_lang, source_text) 
```

