import datetime
import json
import random

import gradio as gr


with (
    # open("../translations/translations_en_spa.json") as gpt4_spa,
    open("../translations/gpt4a_en_spa_dp.json") as gpt4_spa,
    open("../translations/gpt4a_en_bul_dp.json") as gpt4_bul,
    open("../translations/gpt4a_en_man_dp.json") as gpt4_man,
    open(
        "../translations/gpt4_en_spa_flores_sample.json"
    ) as gpt4_spa_flores,
    open(
        "../translations/gpt4_en_bul_flores_sample.json"
    ) as gpt4_bul_flores,
    open(
        "../translations/gpt4_en_man_flores_sample.json"
    ) as gpt4_man_flores,
    # open("../translations/translations_en_deu.json") as gpt4_deu,
    # open("../translations/translations_en_jpn.json") as gpt4_jap,
    open("../translations/google_en_spa_dp.json") as goog_spa,
    open("../translations/google_en_bul_dp.json") as goog_bul,
    open("../translations/google_en_man_dp.json") as goog_man,
    open(
        "../translations/google_en_spa_flores_sample.json"
    ) as goog_spa_flores,
    open(
        "../translations/google_en_bul_flores_sample.json"
    ) as goog_bul_flores,
    open(
        "../translations/google_en_man_flores_sample.json"
    ) as goog_man_flores,
    # open("../translations/google_en_spa.json") as goog_spa,
    # open("../translations/google_en_deu.json") as goog_deu,
    # open("../translations/google_en_fra.json") as goog_fra,
    # open("../translations/google_en_por.json") as goog_por,
    # open("../translations/google_en_kor.json") as goog_kor,
    # open("../translations/google_en_jap.json") as goog_jap,
):
    gpt4_en_spa = json.load(gpt4_spa)
    gpt4_en_bul = json.load(gpt4_bul)
    gpt4_en_man = json.load(gpt4_man)
    gpt4_en_spa_f = json.load(gpt4_spa_flores)
    gpt4_en_bul_f = json.load(gpt4_bul_flores)
    gpt4_en_man_f = json.load(gpt4_man_flores)
    # gpt4_en_deu = json.load(gpt4_deu)
    # gpt4_en_jap = json.load(gpt4_jap)
    goog_en_spa = json.load(goog_spa)
    goog_en_bul = json.load(goog_bul)
    goog_en_man = json.load(goog_man)
    goog_en_spa_f = json.load(goog_spa_flores)
    goog_en_bul_f = json.load(goog_bul_flores)
    goog_en_man_f = json.load(goog_man_flores)

    # goog_en_deu = json.load(goog_deu)
    # goog_en_fra = json.load(goog_fra)
    # goog_en_por = json.load(goog_por)
    # goog_en_kor = json.load(goog_kor)
    # goog_en_jap = json.load(goog_jap)

gpt4_en_spa += gpt4_en_spa_f
gpt4_en_bul += gpt4_en_bul_f
gpt4_en_man += gpt4_en_man_f

goog_en_spa += goog_en_spa_f
goog_en_bul += goog_en_bul_f
goog_en_man += goog_en_man_f


share_js = """
function () {
    const captureElement = document.querySelector('#share-region-annoy');
    // console.log(captureElement);
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'guardrails-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [];
}
"""

css = """
#language_box {width: 20%}
"""


def gen_random():
    """
    Generate a random index within the range of available translations in gpt4_en_spa.

    Returns:
        int: A random integer between 0 (inclusive) and len(gpt4_En_Spa) - 1.
    """
    return random.randint(0, len(gpt4_en_spa) - 1)


model_info = [
    {
        "model": "gpt-4-turbo",
        "company": "OpenAI",
        "type": "Decoder LLM",
        "link": "https://google.com",
    },
    {
        "model": "google-translate",
        "company": "Google",
        "type": "NMT",
        "link": "https://google.com",
    },
    {
        "model": "DeepL",
        "company": "DeepL",
        "type": "NMT",
        "link": "https://google.com",
    },
]

models = ["gpt-4-turbo", "google-translate"]


def get_model_description_md(models):
    """
    Generate a Markdown description of the models in `models`.

    Args:
        models (list): A list of model names.

    Returns:
        str: A Markdown string containing descriptions of each model.
    """

    model_description_md = "_________________________________ <br>"
    ct = 0
    visited = set()
    for _, name in enumerate(models):
        if name in visited:
            continue
        else:
            minfo = [x for x in model_info if x["model"] == name]
            visited.add(name)
            one_model_md = f"[{minfo[0]['model']}]({minfo[0]['link']}): {minfo[0]['type']}"
            new_line = "_________________________________ <br>"
            model_description_md += f" {one_model_md} <br> {new_line}"
            ct += 1
    return model_description_md


gpt4_language_dict = {
    "Spanish": gpt4_en_spa,
    "Bulgarian": gpt4_en_bul,
    "Chinese": gpt4_en_man,
}
google_language_dict = {
    "Spanish": goog_en_spa,
    "Bulgarian": goog_en_bul,
    "Chinese": goog_en_man,
}


def change_language(txtbox1, txtbox2, src_txtbox, new_lang):
    new_idx = gen_random()
    print(new_idx)
    txtbox1 = gpt4_language_dict[new_lang][new_idx]["translation"]
    txtbox2 = google_language_dict[new_lang][new_idx]["translation"]
    src_txtbox = gpt4_language_dict[new_lang][new_idx]["source_txt"]
    new_idx_random = gr.State(value=new_idx, render=False)
    print(new_idx_random.value)
    return txtbox1, txtbox2, src_txtbox


def write_answer(component, model):
    print(model)
    match component:
        case "👈  A is better":
            output = "A"
            model_name = model
            gr.Info(f"{model_name} won!")
        case "👉  B is better":
            output = "B"
            model_name = [x for x in models if x != model][0]
            gr.Info(f"{model_name} won!")
        case "🤝  Tie":
            output = "tie"
            model_name = "tie"
            gr.Info("'Tis a tie")
        case "👎  Both are bad":
            output = "both-bad"
            model_name = "both-bad"
            gr.Info("Both were bad!")
        case _:
            output = None
            model_name = None
    new_dict = {
        "time": datetime.datetime.now(),
        "output": output,
        "win_model": model_name,
    }
    print(new_dict)


def regen(language, model):
    new_idx = gen_random()
    print(new_idx)
    model_a = random.choice(models)
    txtbox1_model = gr.State(value=model_a, render=True)
    match txtbox1_model.value:
        case "gpt-4-turbo":
            init_value_a = gpt4_language_dict[language][new_idx]["translation"]

            init_value_b = google_language_dict[language][new_idx][
                "translation"
            ]

        case "google-translate":
            init_value_a = google_language_dict[language][new_idx][
                "translation"
            ]

            init_value_b = gpt4_language_dict[language][new_idx]["translation"]
        case _:
            init_value_a = None
            init_value_b = None

    txtbox1 = init_value_a
    txtbox2 = init_value_b
    src_txtbox = gpt4_language_dict[language][new_idx]["source_txt"]
    return txtbox1, txtbox2, src_txtbox, txtbox1_model.value
