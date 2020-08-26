#!/usr/bin/env bash

EXTRACTOR_URL="https://raw.githubusercontent.com/shkarupa-alex/wikiextractor/master/cirrus-extract.py"
if [[ ! -f cirrus-extract.py ]]; then
  echo "Downloading..."
  wget "${EXTRACTOR_URL}"
fi;


TOC_URL="https://dumps.wikimedia.org/other/cirrussearch/"
DUMP_DIR=$(curl $TOC_URL | sed -e 's/.*"\([0-9][0-9]*\/\)".*/\1/' | grep -E '[0-9]{8}' | tail -n 2 | head -n 1)

DUMPS_URL="${TOC_URL}${DUMP_DIR}"
mapfile -t DUMP_FILES < <(curl "$DUMPS_URL" | sed -e 's/.*"\([a-z][a-z]wiki-.*-content.json.gz\)".*/\1/' | grep ".gz"  | grep -v " " | sort -u | grep ruwi)

echo "Downloading..."
DUMPS_DIR="wiki_dumps"
mkdir -p "${DUMPS_DIR}"
(
  cd "${DUMPS_DIR}" || exit

  for FILE in "${DUMP_FILES[@]}"; do
    if [[ ! -f "${FILE}" ]]; then
      wget "${DUMPS_URL}${FILE}"
    fi
  done
)


echo "Extracting..."
DATASET_DIR="wiki_dataset"
rm -rf "${DATASET_DIR}"
mkdir "${DATASET_DIR}"
(
  cd "${DATASET_DIR}" || exit

  for FILE in "${DUMP_FILES[@]}"; do
    python ../cirrus-extract.py "../${DUMPS_DIR}/${FILE}" -o "${FILE}" -b 20M
  done
)


echo "GZipping..."
gzip -r "${DATASET_DIR}"

rm -rf "${DUMPS_DIR}"
rm cirrus-extract.py