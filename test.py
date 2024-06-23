import translation_agent as ta
source_text = "Today is a good day. Sunny, shine, breeze, blue sky"
source_lang, target_lang, country = "English", "Simplified Chinese", "China"
translation = ta.translate(source_lang, target_lang, source_text, country)
