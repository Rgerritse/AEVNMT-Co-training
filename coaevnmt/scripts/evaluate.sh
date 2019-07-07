#!/usr/bin/env bash
FILE=$1
REF=$2

# DETOKENIZE
DETOK="${FILE}.detok"

perl data/mosesdecoder/scripts/tokenizer/detokenizer.perl -q < $FILE > $DETOK
rm $FILE

# Compute bleu
sacrebleu --input $DETOK $REF --score-only
