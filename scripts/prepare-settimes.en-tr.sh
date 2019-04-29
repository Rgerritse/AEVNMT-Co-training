#!/usr/bin/env bash

cd ..
mkdir -p data
cd data

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
TTC=$SCRIPTS/recaser/train-truecaser.perl
TC=$SCRIPTS/recaser/truecase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000
ZIP=en-tr.txt.zip

if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  echo "Cloning subword-nmt for bpe"
  git clone https://github.com/rsennrich/subword-nmt.git
fi

URL="http://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-tr.txt.zip"

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=tr
lang=en-tr
prep=setimes.tokenized.en-tr
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep
#
echo "Downloading data from ${URL}..."
cd $orig
wget "$URL" -O $ZIP

if [ -f $ZIP ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

unzip $ZIP
cd ../..

python split_setimes.en-tr.py \
    --in_dir data/$orig/ \
    --in_src SETIMES.en-tr.en \
    --in_tgt SETIMES.en-tr.tr \
    --out_dir data/$tmp/ \
    --out_src_train train.en \
    --out_tgt_train train.tr \
    --out_src_dev valid.en \
    --out_tgt_dev valid.tr \
    --out_src_test test.en \
    --out_tgt_test test.tr

cd data/

for l in $src $tgt; do
  for d in train valid test; do
    f=$d.$l
    tok=$d.tok.$l
    echo "Tokenizing $f..."
    perl $TOKENIZER -q -l $l -threads 8 < $tmp/$f > $tmp/$tok
  done
done


for l in $src $tgt; do
  echo "Training $l truecase model..."
  perl $TTC --model $tmp/tc-model.$l --corpus $tmp/train.tok.$l
done


for l in $src $tgt; do
  for d in train valid test; do
    echo "Truecasing $tmp/$d.tok.$l"
    perl $TC --model $tmp/tc-model.$l < $tmp/$d.tok.$l > $tmp/$d.tc.$l
  done
done

TRAIN=$tmp/train.${src}-${tgt}
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.tc.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for d in train valid test; do
        f=$d.tc.$L
        bpe=$d.bpe.$L
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/$bpe
    done
done

echo -e "<unk>\n<s>\n</s>" > $prep/vocab.bpe
cat $tmp/train.bpe.$src $tmp/train.bpe.$tgt | \
  $BPEROOT/get_vocab.py | cut -f1 -d ' ' >> $prep/vocab.bpe

for d in train valid test; do
  echo "Cleaning $tmp/$d.bpe.$src and $tmp/$d.bpe.$tgt"
  perl $CLEAN $tmp/$d.bpe $src $tgt $prep/$d 1 50
done
