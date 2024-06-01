#!/bin/bash

mkdir -p "$PWD/data/wmt"

BASE_DIR="$PWD/data/wmt"

##MOSES_SCRIPT ="$PWD/input-from-sgm.perl"

wget -O "$BASE_DIR/wmt19_test.tgz" http://data.statmt.org/wmt19/translation-task/test.tgz 
wget -O  "$BASE_DIR/wmt14_test_full.tgz" http://www.statmt.org/wmt14/test-full.tgz
wget -O  "$BASE_DIR/wmt13_test.tgz" http://www.statmt.org/wmt13/test.tgz
##
mkdir -p "$BASE_DIR/wmt19_test" &&  tar xvf "$BASE_DIR/wmt19_test.tgz" -C  "$BASE_DIR/wmt19_test"
mkdir -p "$BASE_DIR/wmt14_test" &&  tar xvf "$BASE_DIR/wmt14_test_full.tgz" -C  "$BASE_DIR/wmt14_test"
mkdir -p "$BASE_DIR/wmt13_test" && tar xvf "$BASE_DIR/wmt13_test.tgz" -C  "$BASE_DIR/wmt13_test"

# WMT 13 Spanish (es-en)
mkdir -p  "$BASE_DIR/wmt13_test/txt"
$MOSES_SCRIPT < "$BASE_DIR/wmt13_test/test/newstest2013-src.en.sgm" > "$BASE_DIR/wmt13_test/txt/newstest2013-src.en.txt"
$MOSES_SCRIPT  < "$BASE_DIR/wmt13_test/test/newstest2013-src.es.sgm" > "$BASE_DIR/wmt13_test/txt/newstest2013-src.es.txt"

# WMT 14: German, French (en-de, en-fr)
mkdir -p "$BASE_DIR/wmt14_test/txt"
$MOSES_SCRIPT < "$BASE_DIR/wmt14_test/test-full/newstest2014-fren-src.en.sgm" > "$BASE_DIR/wmt14_test/txt/newstest2014-fren-src.en.txt"
$MOSES_SCRIPT  < "$BASE_DIR/wmt14_test/test-full/newstest2014-fren-ref.fr.sgm" > "$BASE_DIR/wmt14_test/txt/newstest2014-fren-ref.fr.txt"

$MOSES_SCRIPT < "$BASE_DIR/wmt14_test/test-full/newstest2014-deen-src.en.sgm" > "$BASE_DIR/wmt14_test/txt/newstest2014-deen-src.en.txt"
$MOSES_SCRIPT  < "$BASE_DIR/wmt14_test/test-full/newstest2014-deen-ref.de.sgm" > "$BASE_DIR/wmt14_test/txt/newstest2014-deen-ref.de.txt"


# WMT 19: RU, ZH-Hans
mkdir -p "$BASE_DIR/wmt19_test/txt"
$MOSES_SCRIPT  < "$BASE_DIR/wmt19_test/sgm/newstest2019-enzh-src.en.sgm" > "$BASE_DIR/wmt19_test/txt/newstest2019-enzh-src.en.txt"
$MOSES_SCRIPT < "$BASE_DIR/wmt19_test/sgm/newstest2019-enzh-ref.zh.sgm" > "$BASE_DIR/wmt19_test/txt/newstest2019-enzh-ref.zh.txt"

$MOSES_SCRIPT < "$BASE_DIR/wmt19_test/sgm/newstest2019-enru-src.en.sgm" > "$BASE_DIR/wmt19_test/txt/newstest2019-enru-src.en.txt"
$MOSES_SCRIPT < "$BASE_DIR/wmt19_test/sgm/newstest2019-enru-ref.ru.sgm" > "$BASE_DIR/wmt19_test/txt/newstest2019-enru-ref.ru.txt"
