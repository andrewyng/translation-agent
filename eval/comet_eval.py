import json

from comet import download_model
from comet import load_from_checkpoint


model_path = download_model("wmt20-comet-da")  # ("Unbabel/wmt20-comet-qe-da")
model = load_from_checkpoint(model_path)

ref_path = "./data/floresp-v2.0-rc.3/devtest/devtest."

with open(ref_path + "fra_Latn") as f:
    reference_text = f.readlines()

reference_text = [x.replace("\n", "") for x in reference_text]


translations_path = "./translations/"

# dir_list = os.listdir(translations_path)

dir_list = [
    "translations_en_fra.json"
]  # ,'deepl_en_kor.json', 'deepl_en_jap.json', 'translations_nllb_en_spa.json']

for translation_file in dir_list:
    with open(translations_path + translation_file) as f:
        jdata = json.load(f)

    jdata_reformated = []
    for i in range(len(jdata)):
        temp_dict = {}

        temp_dict["src"] = jdata[i]["source_text"].strip()
        temp_dict["mt"] = jdata[i]["translation"].strip()
        temp_dict["ref"] = reference_text[i].strip()
        jdata_reformated.append(temp_dict)

    model_output = model.predict(jdata_reformated, batch_size=8, num_workers=0)

    print(translation_file, "\t", round(model_output[1], 2))
