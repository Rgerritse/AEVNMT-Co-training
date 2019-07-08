#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=aev_train
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Loading modules
echo "Loading modules"
module load python/3.5.0
module load CUDA/8.0.44-GCCcore-5.4.0
module load cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0

# Venv
source ${HOME}/code/venv/bin/activate

cp -r $HOME/code/coaevnmt/data/ ${TMPDIR}/data

session="cond_nmt_lr"

hyps=(0.005, 0.001 0.0005 0.0001)
NUM_GPUS=4
for ((GPU=0; GPU < ${NUM_GPUS}; GPU++ ))
do

	OUTPUT_DIR="${HOME}/code/output/${session}-${hyps[$GPU]}"
	mkdir ${OUTPUT_DIR}

	CUDA_VISIBLE_DEVICES=${GPU} python -u ${HOME}/code/coaevnmt/main.py \
		--session "${session}-${hyps[$GPU]}" \
		--config ${HOME}/code/coaevnmt/config/cond_nmt.json \
		--data_dir ${TMPDIR}/data/en-tr/bilingual/train_100000.en-tr \
		--out_dir ${OUTPUT_DIR} \
		--learning_rate ${hyps[$GPU]} \
		&> ${HOME}/code/coaevnmt/output/${session}-${hyps[$GPU]}/log_file &

done
wait
