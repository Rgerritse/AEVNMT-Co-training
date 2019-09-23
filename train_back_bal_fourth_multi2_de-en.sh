#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cbfde2
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
	session="cond_back_balanced_fourth_de-en_run_$((GPU+4))"

	CUDA_VISIBLE_DEVICES=${GPU} python -u ${HOME}/code/coaevnmt/train_super.py \
		--session "${session}" \
		--config ${HOME}/code/coaevnmt/config/cond_nmt_multi30k_de-en.json \
		--data_dir ${TMPDIR}/data/multi30k \
		--out_dir ${OUTPUT_DIR} \
		--train_prefix "training_fourth" \
		--back_prefix "back_fourth_de-en_$((GPU+4))" \
		&> "${OUTPUT_DIR}/log_file-${session}" &

done
wait
