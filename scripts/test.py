#!env python 

import translation_agent as ta

source_lang, target_lang = "English", "Spanish"

filename = "sample-texts/sample-long1.txt"
with open(filename, 'r', encoding='utf-8') as file:
    source_text = file.read()

print(f"Source text:\n\n{source_text}\n------------\n")

translation = ta.translate(source_lang, target_lang, source_text) 

print(f"Translation:\n\n{translation}")

