import json

import sacrebleu
from icecream import ic
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import pipeline


def load_model(model):
    model = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model)
    return model, tokenizer


def translation(translator, text):
    output = translator(text)
    return output[0]["translation_text"]


if __name__ == "__main__":
    # model, tokenizer = load_model("facebook/nllb-200-3.3B")
    with open("./data/floresp-v2.0-rc.3/devtest/devtest.eng_Latn") as f:
        source_text = f.readlines()

    with open("./data/floresp-v2.0-rc.3/devtest/devtest.jpn_Jpan") as f:
        reference_text = f.readlines()

    reference_text = [x.replace("\n", "") for x in reference_text]
    source_text = [x.replace("\n", "") for x in source_text]

    translator = pipeline(
        "translation",
        model="facebook/nllb-200-3.3B",
        tokenizer="facebook/nllb-200-3.3B",
        src_lang="eng_Latn",
        tgt_lang="jpn_Jpan",
        device="cuda",
    )

    translations = [
        translation(
            translator,
            text=text,
        )
        for text in source_text
    ]
    translations_list = []
    for idx, val in enumerate(translations):
        new_dict = {"source_text": source_text[idx], "translation": val}
        translations_list.append(new_dict)

    with open("./translations/translations_nllb_en_jap.json", "w") as f:
        json.dump(translations_list, f)

    bleu = sacrebleu.corpus_bleu(
        translations, [reference_text], tokenize="flores200"
    )
    chrf = sacrebleu.corpus_chrf(translations, [reference_text])

    ic(bleu)
    ic(chrf)
