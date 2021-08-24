#!/usr/bin/env bash

TREEBANKS_URL="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3687/ud-treebanks-v2.8.tgz?sequence=1&isAllowed=y"
TREEBANKS_NAME="ud-treebanks-v2.8"

if [[ ! -f "${TREEBANKS_NAME}.tgz" ]]; then
  echo "Downloading..."
  wget "${TREEBANKS_URL}" -O "${TREEBANKS_NAME}.tgz" || exit
fi;

if [[ ! -d "${TREEBANKS_NAME}" ]]; then
  echo "Unpacking..."
  tar zxvf "${TREEBANKS_NAME}.tgz"
fi;


DATASET_DIR="ud_dataset"
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
        BREAKS=$(cat "${FILE}" | grep "\t" | grep -v "# " | grep -v -c "SpaceAfter=No" < "${FILE}")
        JOINS=$(cat "${FILE}" | grep "\t" | grep -v "# " | grep -c "SpaceAfter=No" < "${FILE}")
        echo "${DIR}/${FILE} ${BREAKS} ${JOINS}"
      done
    fi;
  done
)


echo "Checking..."
rm "${DATASET_DIR}/UD_Arabic-NYUAD.txt"
rm "${DATASET_DIR}/UD_English-ESL.txt"
rm "${DATASET_DIR}/UD_English-GUMReddit.txt"
rm "${DATASET_DIR}/UD_French-FTB.txt"
rm "${DATASET_DIR}/UD_Hindi_English-HIENCS.txt"
rm "${DATASET_DIR}/UD_Japanese-BCCWJ.txt"
rm "${DATASET_DIR}/UD_Mbya_Guarani-Dooley.txt"
grep -R "___" "${DATASET_DIR}/" | sed "s/:.*//" | uniq -c
grep -R "_ _ _" "${DATASET_DIR}/" | sed "s/:.*//" | uniq -c


echo "GZipping..."
gzip -r "${DATASET_DIR}"

rm "${TREEBANKS_NAME}.tgz"
rm -rf "${TREEBANKS_NAME}"