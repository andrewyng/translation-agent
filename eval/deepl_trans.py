import json
import os

import deepl
import sacrebleu
from icecream import ic


def deepl_translate(text, lang):
    auth_key = os.getenv("DEEPL_API_KEY")
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(text, target_lang=lang)

    return result.text


if __name__ == "__main__":
    with open("./data/floresp-v2.0-rc.3/devtest/devtest.eng_Latn") as f:
        source_text = f.readlines()

    with open("./data/floresp-v2.0-rc.3/devtest/devtest.cmn_Hans") as f:
        reference_text = f.readlines()

    reference_text = [x.replace("\n", "") for x in reference_text]
    source_text = [x.replace("\n", "") for x in source_text]

    translations = [deepl_translate(text=x, lang="ZH") for x in source_text]

    translation_list = []
    for idx, val in enumerate(translations):
        new_dict = {"source_text": source_text[idx], "translation": val}
        translation_list.append(new_dict)

    with open("./translations/deepl_en_man.json", "w") as f:
        json.dump(translation_list, f)

    bleu = sacrebleu.corpus_bleu(
        translations, [reference_text], tokenize="flores200"
    )
    chrf = sacrebleu.corpus_chrf(translations, [reference_text])

    ic(bleu)
    ic(chrf)
