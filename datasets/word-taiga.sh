#!/usr/bin/env bash

CORPORA_URL="https://linghub.ru/static/Taiga/retagged_taiga.tar.gz"
CORPORA_NAME="taiga_corpora"


#if [[ ! -d ${CORPORA_NAME} ]]; then
#  if [[ ! -f ${CORPORA_NAME}.tar.gz ]]; then
#    echo "Downloading..."
#    wget ${CORPORA_URL} -O ${CORPORA_NAME}.tar.gz
#  fi;
#
#  echo "Unpacking..."
#  mkdir -p ${CORPORA_NAME}/archives/
#  tar zxvf ${CORPORA_NAME}.tar.gz -C ${CORPORA_NAME}/archives/
#  rm ${CORPORA_NAME}.tar.gz
#fi;


for FILE in ${CORPORA_NAME}/archives/*.tar.gz ; do
  rm -rf ${CORPORA_NAME}/tmp/
  mkdir -p ${CORPORA_NAME}/tmp/
  echo "Processing ${FILE} ..."
  tar zxvf ${FILE} -C ${CORPORA_NAME}/tmp/
  grep -R -L "# sent_id =" ${CORPORA_NAME}/tmp/ | grep ".txt" | xargs rm
  rm -f ${CORPORA_NAME}/${FILE##*/}_full.txt ${CORPORA_NAME}/${FILE##*/}.txt
  find ${CORPORA_NAME}/tmp/ -type f -name "*.txt" -exec cat "{}" | grep -v "^#" | cut -f2 | perl -CSD -Mutf8 -0 -pe 's/(.)\r?\n/$1 /g' >> ${CORPORA_NAME}/${FILE##*/}_full.txt
  find ${CORPORA_NAME}/tmp/ -type f -name "*.txt" -exec cat "{}" | grep -E "^$|\t(ADJ|ADV|NOUN|PROPN|VERB)\t" | grep -v "^#" | cut -f2 | perl -CSD -Mutf8 -0 -pe 's/(.)\r?\n/$1 /g' >> ${CORPORA_NAME}/${FILE##*/}.txt
#  rm ${FILE}
done

for FILE in ${CORPORA_NAME}/archives/*.zip ; do
  rm -rf ${CORPORA_NAME}/tmp/
  mkdir -p ${CORPORA_NAME}/tmp/
  echo "Processing ${FILE} ..."
  unzip ${FILE} -d ${CORPORA_NAME}/tmp/
  grep -R -L "# sent_id =" ${CORPORA_NAME}/tmp/ | grep ".txt" | xargs rm
  rm -f ${CORPORA_NAME}/${FILE##*/}_full.txt ${CORPORA_NAME}/${FILE##*/}.txt
  find ${CORPORA_NAME}/tmp/ -type f -name "*.txt" -exec cat "{}" | grep -v "^#" | cut -f2 | perl -CSD -Mutf8 -0 -pe 's/(.)\r?\n/$1 /g' >> ${CORPORA_NAME}/${FILE##*/}_full.txt
  find ${CORPORA_NAME}/tmp/ -type f -name "*.txt" -exec cat "{}" | grep -E "^$|\t(ADJ|ADV|NOUN|PROPN|VERB)\t" | grep -v "^#" | cut -f2 | perl -CSD -Mutf8 -0 -pe 's/(.)\r?\n/$1 /g' >> ${CORPORA_NAME}/${FILE##*/}.txt
#  rm ${FILE}
done


#echo "Cleaning..."
#rm -rf ud_dataset
#mkdir ud_dataset
#(
#  cd ${TREEBANKS_NAME}
#
#  for DIR in * ; do
#    if [[ -d ${DIR} ]]; then
#      echo "Processing ${DIR}"
#
#      for FILE in ${DIR}/*.conllu ; do
#        cat $FILE | grep "# text = " | sed 's/# text = //g' >> ../ud_dataset/${DIR}.txt
#      done
#    fi;
#  done
#)
#
#
#echo "Checking..."
#rm ud_dataset/UD_Japanese-BCCWJ.txt
#rm ud_dataset/UD_Arabic-NYUAD.txt
#rm ud_dataset/UD_English-ESL.txt
#rm ud_dataset/UD_French-FTB.txt
#rm ud_dataset/UD_Hindi_English-HIENCS.txt
#grep -R "___" ud_dataset/ | sed "s/:.*//" | uniq -c
#grep -R "_ _ _" ud_dataset/ | sed "s/:.*//" | uniq -c
#
#
#echo "GZipping..."
#gzip -r ud_dataset


# head -n 100 /Users/alex/HDD/Develop/semtech/tfstbd/data/source/russian/ru_taiga-ud-dev.conllu | grep -v "^#" | cut -f2
#grep -v -E "\t(PUNCT|SYM|X)\t"
#союзы, местоимения, предлоги, частицы
#perl -CSD -Mutf8 -0 -pe 's/(.)\r?\n/$1 /g' tmp0.txt