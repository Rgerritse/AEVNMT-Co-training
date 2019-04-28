#!/usr/bin/env bash
PRED_DIR=$1
SESSION=$2
STEP=$3
REF=$4
LANG=$5

FILE="${PRED_DIR}/${SESSION}-${STEP}.raw.${LANG}"
UNBPE_FILE="${PRED_DIR}/${SESSION}-${STEP}.unbpe.${LANG}"
DETOK_FILE="${PRED_DIR}/${SESSION}-${STEP}.detok.${LANG}"

# UNDO BPE
sed -r 's/(@@ )|(@@ ?$)//g' $FILE > $UNBPE_FILE

# DETOKENIZE
perl data/mosesdecoder/scripts/tokenizer/detokenizer.perl -q < $UNBPE_FILE > $DETOK_FILE

# Compute bleu
sacrebleu --input $DETOK_FILE $REF --score-only
