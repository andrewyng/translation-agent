import json

import sacrebleu
from google.cloud import translate
from icecream import ic


def translate_text(
    text: str = "YOUR_TEXT_TO_TRANSLATE",
    project_id: str = "YOUR_PROJECT_ID",
    source_lang: str = "en-US",
    target_lang: str = "es",
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": source_lang,
            "target_language_code": target_lang,
        }
    )

    # Display the translation for each input text provided
    return response.translations[0].translated_text


if __name__ == "__main__":
    with open("./data/floresp-v2.0-rc.3/devtest/devtest.eng_Latn") as f:
        source_text = f.readlines()

    with open("./data/floresp-v2.0-rc.3/devtest/devtest.jpn_Jpan") as f:
        reference_text = f.readlines()

    reference_text = [x.replace("\n", "") for x in reference_text]
    source_text = [x.replace("\n", "") for x in source_text]

    translations = [
        translate_text(text=x, project_id="santerre", target_lang="ja")
        for x in source_text
    ]

    translation_list = []
    for idx, val in enumerate(translations):
        new_dict = {"source_text": source_text[idx], "translation": val}
        translation_list.append(new_dict)

    with open("./translations/google_en_jap.json", "w") as f:
        json.dump(translation_list, f)

    bleu = sacrebleu.corpus_bleu(
        translations, [reference_text], tokenize="flores200"
    )
    chrf = sacrebleu.corpus_chrf(translations, [reference_text])

    ic(bleu)
    ic(chrf)
