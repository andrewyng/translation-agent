import translation_agent as ta
source_text = "Today is a good day. Sunny, shine, breeze, blue sky"
source_lang, target_lang, country,max_tokens = "English", "Simplified Chinese", "China", 100
translation = ta.translate(source_lang, target_lang, source_text, country,max_tokens=max_tokens)
