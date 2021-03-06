#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cakld2
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Loading modules
echo "Loading modules"
module load python/3.5.0
module load CUDA/10.0.130
module load cuDNN/7.4.2-CUDA-10.0.130

# Venv
echo "Loading virtual environment"
source ${HOME}/code/venv/bin/activate

echo "Coping data"
cp -r $HOME/code/coaevnmt/data/ ${TMPDIR}/data

OUTPUT_DIR="${HOME}/code/coaevnmt/output"
echo "Creating ${OUTPUT_DIR}"
mkdir -p ${OUTPUT_DIR}

echo "Starting training"
NUM_GPUS=4
for ((GPU=0; GPU < ${NUM_GPUS}; GPU++ ))
do
	session="coaevnmt_curc_diff_greedy_lr2_kldiff25_en-de_run_${GPU}"

	CUDA_VISIBLE_DEVICES=${GPU} python -u ${HOME}/code/coaevnmt/train_semi.py \
		--session "${session}" \
		--config ${HOME}/code/coaevnmt/config/coaevnmt_multi30k_curc_diff_greedy_lr2_en-de.json \
		--data_dir ${TMPDIR}/data/multi30k \
		--out_dir ${OUTPUT_DIR} \
		--kl_free_nats_style indv \
		--kl_free_nats 2.5 \
		&> "${OUTPUT_DIR}/log_file-${session}" &

done
wait
