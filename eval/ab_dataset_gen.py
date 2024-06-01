import json

import translation_agent as ta
from dotenv import load_dotenv
from google.cloud import translate


load_dotenv()


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
    source_lang, target_lang, country = "English", "Chinese", "China"
    with open("./sample-texts/google_en_spa_flores_sample.json") as f:
        data = json.load(f)

    translations_agents = []
    translations_google = []
    for entry in data:
        print(f"Source text:\n\n{entry["source_txt"]}\n------------\n")
        translation_google = translate_text(
            text=entry["source_txt"],
            project_id="santerre",
            source_lang="en-US",
            target_lang="zh",
        )

        translation_agents = ta.translate(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=entry["source_txt"],
            country=country,
        )

        new_dict_google = {
            "source_txt": entry["source_txt"],
            "translation": translation_google,
        }

        new_dict_agents = {
            "source_txt": entry["source_txt"],
            "translation": translation_agents,
        }

        translations_google.append(new_dict_google)
        translations_agents.append(new_dict_agents)

    with open("google_en_man_flores_sample.json", "w") as ts:
        json.dump(translations_google, ts)

    with open("gpt4_en_man_flores_sample.json", "w") as ta:
        json.dump(translations_agents, ta)
