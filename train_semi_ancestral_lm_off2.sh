#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=semi_a_off_train
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Loading modules
echo "Loading modules"
module load python/3.5.0
module load CUDA/8.0.44-GCCcore-5.4.0
module load cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0

# Venv
echo "Loading virtual environment"
source ${HOME}/code/venv/bin/activate

echo "Coping data"
cp -r $HOME/code/coaevnmt/data/ ${TMPDIR}/data

OUTPUT_DIR="${HOME}/code/coaevnmt/output"
echo "Creating ${OUTPUT_DIR}"
mkdir -p ${OUTPUT_DIR}

session="coaevnmt_ancestral_lm_off_run_5"

python -u ${HOME}/code/coaevnmt/train_semi_new.py \
	--session "${session}" \
	--config ${HOME}/code/coaevnmt/config/coaevnmt_multi30k_ancestral_lm_off.json \
	--data_dir ${TMPDIR}/data/multi30k \
	--out_dir ${OUTPUT_DIR} \
	&> "${OUTPUT_DIR}/log_file-${session}"
