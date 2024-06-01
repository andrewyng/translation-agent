import json
from concurrent.futures import ThreadPoolExecutor

import sacrebleu
import translation_agent as ta
from icecream import ic


def run_trans(example):
    translation = ta.translate(
        source_lang="English", target_lang="German", source_text=example
    )
    return translation


if __name__ == "__main__":
    with open("./data/floresp-v2.0-rc.3/devtest/devtest.eng_Latn") as f:
        source_text = f.readlines()

    with open("./data/floresp-v2.0-rc.3/devtest/devtest.deu_Latn") as f:
        reference_text = f.readlines()

    reference_text = [x.replace("\n", "") for x in reference_text]

    max_workers = 16

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_trans, text) for text in source_text]

        translations = [future.result() for future in futures]

    translations_list = []
    for idx, val in enumerate(translations):
        new_dict = {"source_text": source_text[idx], "translation": val}
        translations_list.append(new_dict)

    with open("translations_en_deu.json", "w") as f:
        json.dump(translations_list, f)

    bleu = sacrebleu.corpus_bleu(
        translations, [reference_text], tokenize="flores200"
    )
    chrf = sacrebleu.corpus_chrf(translations, [reference_text])

    ic(bleu)
    ic(chrf)
