#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=conb3
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
	session="conmt_greedy_no_warmup_en-de_run_${GPU}"

	CUDA_VISIBLE_DEVICES=${GPU} python -u ${HOME}/code/coaevnmt/train_semi.py \
		--session "${session}" \
		--config ${HOME}/code/coaevnmt/config/conmt_multi30k_en-de.json \
		--data_dir ${TMPDIR}/data/multi30k \
		--out_dir ${OUTPUT_DIR} \
		--bilingual_warmup 0 \
		&> "${OUTPUT_DIR}/log_file-${session}" &

done
wait
