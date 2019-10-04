#!/usr/bin/env bash

TREEBANKS_URL="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz?sequence=4&isAllowed=y"
TREEBANKS_NAME="ud-treebanks-v2.4"

if [[ ! -f ${TREEBANKS_NAME}.tgz ]]; then
  echo "Downloading..."
  wget ${TREEBANKS_URL} -O ${TREEBANKS_NAME}.tgz
fi;

if [[ ! -d ${TREEBANKS_NAME} ]]; then
  echo "Unpacking..."
  tar zxvf ${TREEBANKS_NAME}.tgz
fi;


echo "Cleaning..."
rm -rf ud_dataset
mkdir ud_dataset
(
  cd ${TREEBANKS_NAME}

  for DIR in * ; do
    if [[ -d ${DIR} ]]; then
      echo "Processing ${DIR}"

      for FILE in ${DIR}/*.conllu ; do
        cat $FILE | grep "# text = " | sed 's/# text = //g' >> ../ud_dataset/${DIR}.txt
      done
    fi;
  done
)


echo "Checking..."
rm ud_dataset/UD_Japanese-BCCWJ.txt
rm ud_dataset/UD_Arabic-NYUAD.txt
rm ud_dataset/UD_English-ESL.txt
rm ud_dataset/UD_French-FTB.txt
rm ud_dataset/UD_Hindi_English-HIENCS.txt
grep -R "___" ud_dataset/ | sed "s/:.*//" | uniq -c
grep -R "_ _ _" ud_dataset/ | sed "s/:.*//" | uniq -c


echo "GZipping..."
gzip -r ud_dataset
