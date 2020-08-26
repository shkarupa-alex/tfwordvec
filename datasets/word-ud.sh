#!/usr/bin/env bash

TREEBANKS_URL="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz?sequence=4&isAllowed=y"
TREEBANKS_NAME="ud-treebanks-v2.4"

if [[ ! -f "${TREEBANKS_NAME}.tgz" ]]; then
  echo "Downloading..."
  wget "${TREEBANKS_URL}" -O "${TREEBANKS_NAME}.tgz" || exit
fi;

if [[ ! -d "${TREEBANKS_NAME}" ]]; then
  echo "Unpacking..."
  tar zxvf "${TREEBANKS_NAME}.tgz"
fi;


DATASET_DIR="ud_word_dataset"
rm -rf "${DATASET_DIR}"
mkdir "${DATASET_DIR}"
(
  cd "${TREEBANKS_NAME}" || exit

  for DIR in * ; do
    if [[ -d "${DIR}" ]]; then
      echo "Processing ${DIR}"

      for FILE in "${DIR}"/*.conllu ; do
        # cat ${FILE} | grep -v "^#" | cut -f2 | perl -CSD -Mutf8 -0 -pe 's/(.)\r?\n/$1 /g' >> ../${DATASET_DIR}/${DIR}.txt
        grep -E "^$|\t(ADJ|ADV|NOUN|PROPN|VERB)\t" < "${FILE}" | grep -v "^#" | cut -f2 | perl -CSD -Mutf8 -0 -pe 's/(.)\r?\n/$1 /g' >> "../${DATASET_DIR}/${DIR}.txt"
      done

      # Checking for no-space languages
      for FILE in "${DIR}"/*.conllu ; do
        BREAKS=$(grep -E -c "_$" < "${FILE}")
        JOINS=$(grep -c "SpaceAfter=No" < "${FILE}")
        echo "${DIR}/${FILE} ${BREAKS} ${JOINS}"
      done
    fi;
  done
)


echo "Checking..."
rm "${DATASET_DIR}/UD_Japanese-BCCWJ.txt"
rm "${DATASET_DIR}/UD_Arabic-NYUAD.txt"
rm "${DATASET_DIR}/UD_English-ESL.txt"
rm "${DATASET_DIR}/UD_French-FTB.txt"
rm "${DATASET_DIR}/UD_Hindi_English-HIENCS.txt"
grep -R "___" "${DATASET_DIR}/" | sed "s/:.*//" | uniq -c
grep -R "_ _ _" "${DATASET_DIR}/" | sed "s/:.*//" | uniq -c


echo "GZipping..."
gzip -r "${DATASET_DIR}"

rm "${TREEBANKS_NAME}.tgz"
rm -rf "${TREEBANKS_NAME}"