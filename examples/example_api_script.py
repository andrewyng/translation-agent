import json
import os
import requests

if __name__ == "__main__":
    source_lang, target_lang, country = "English", "Spanish", "Mexico"

    relative_path = "sample-texts/sample-short1.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()

    print(f"Source text:\n\n{source_text}\n------------\n")
    url = "http://localhost:8000/translate/"
    data = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_text": source_text,
        "country": country,
        # "model": "gpt-4-turbo",   # option params, you can change it to other model if you like.
        # "chunk_model": "gpt4", # option params
        # "max_tokens": 1000 # option params
    }
    res = requests.post(url, data=json.dumps(data))
    if res.status_code == 200:
        translation = res.text
        print(f"Translation:\n\n{translation}")
    else:
        print("API Error, status code is ", res.status_code)
