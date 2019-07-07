#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=coaev_train
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

cp -r $HOME/code/data/ ${TMPDIR}/data

session="coaevnmt_latent"

mkdir ${HOME}/code/output/${session}

hyps=(32 64 128 256)
NUM_GPUS=4
for ((GPU=0; GPU < ${NUM_GPUS}; GPU++ ))
do

	OUTPUT_DIR="${HOME}/code/output/${session}-${hyps[$GPU]}"
	mkdir ${OUTPUT_DIR}

	CUDA_VISIBLE_DEVICES=${GPU} python -u ${HOME}/code/train_semi.py \
		--session "${session}-${hyps[$GPU]}" \
		--config ${HOME}/code/config/coaevnmt.json \
		--data_dir ${TMPDIR}/data/setimes.tokenized.en-tr \
		--out_dir ${OUTPUT_DIR} \
		--latent_size ${hyps[$GPU]} \
		&> ${HOME}/code/output/${session}-${hyps[$GPU]}/log_file &

done
wait
