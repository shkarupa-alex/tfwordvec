#!/usr/bin/env bash

EXTRACTOR_URL="https://raw.githubusercontent.com/attardi/wikiextractor/master/cirrus-extract.py"
if [[ ! -f cirrus-extract.py ]]; then
  echo "Downloading..."
  wget ${EXTRACTOR_URL}
fi;


DUMPS_URL="https://dumps.wikimedia.org/other/cirrussearch/20190805/"
DUMP_FILES=($(curl $DUMPS_URL | sed -e 's/.*"\([a-z][a-z]wiki-.*-content.json.gz\)".*/\1/' | grep ".gz"  | grep -v " " | sort -u | grep ruwi))

echo "Downloading..."
mkdir -p wiki-dumps
(
  cd wiki-dumps

  for FILE in "${DUMP_FILES[@]}"; do
    if [[ ! -f ${FILE} ]]; then
      wget ${DUMPS_URL}${FILE}
    fi
  done
)


echo "Extracting..."
rm -rf wiki_dataset
mkdir wiki_dataset
(
  cd wiki_dataset

  for FILE in "${DUMP_FILES[@]}"; do
    python ../cirrus-extract.py ../wiki-dumps/${FILE} -o ${FILE} -b 20M
  done
)


echo "GZipping..."
gzip -r wiki_dataset
