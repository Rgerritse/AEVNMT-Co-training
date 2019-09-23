#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=se_lr1
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

session="coaevnmt_lr1_curriculum_en-de_run_4"

python -u ${HOME}/code/coaevnmt/train_semi.py \
	--session "${session}" \
	--config ${HOME}/code/coaevnmt/config/coaevnmt_multi30k_lr1_en-de.json \
	--data_dir ${TMPDIR}/data/multi30k \
	--out_dir ${OUTPUT_DIR} \
	&> "${OUTPUT_DIR}/log_file-${session}"
