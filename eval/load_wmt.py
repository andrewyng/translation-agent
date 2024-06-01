# from datasets import inspect_dataset, load_dataset_builder
# from datasets import load_dataset


def load_wmt(target_lang, wmt_dir):
    """
    # Using the NLLB paper setup. en-target pairs come from different WMT versions as follows:
    # WMT 19: RU, ZH-Hans
    # WMT 14: DE, FR
    # WMT 13: ES
    """
    if target_lang == "RU":
        source_file = wmt_dir + "wmt19_test/txt/newstest2019-enru-src.en.txt"
        target_file = wmt_dir + "wmt19_test/txt/newstest2019-enru-ref.ru.txt"
    elif target_lang == "ZH":
        source_file = wmt_dir + "wmt19_test/txt/newstest2019-enzh-src.en.txt"
        target_file = wmt_dir + "wmt19_test/txt/newstest2019-enzh-ref.zh.txt"
    elif target_lang == "DE":
        source_file = wmt_dir + "wmt14_test/txt/newstest2014-deen-src.en.txt"
        target_file = wmt_dir + "wmt14_test/txt/newstest2014-deen-ref.de.txt"
    elif target_lang == "FR":
        source_file = wmt_dir + "wmt14_test/txt/newstest2014-fren-src.en.txt"
        target_file = wmt_dir + "wmt14_test/txt/newstest2014-fren-ref.fr.txt"

    elif target_lang == "ES":
        source_file = wmt_dir + "wmt13_test/txt/newstest2013-src.en.txt"
        target_file = wmt_dir + "wmt13_test/txt/newstest2013-src.es.txt"

    return source_file, target_file


# example usage
sf, tf = load_wmt("ZH", "./wmt")
