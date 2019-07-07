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

cp -r $HOME/code/data/ ${TMPDIR}/data

session="aevnmt_type"

hyp="bahdanau"

OUTPUT_DIR="${HOME}/code/output/${session}-${hyp}"
mkdir ${OUTPUT_DIR}

python -u ${HOME}/code/main.py \
	--session "${session}-${hyp}" \
	--config ${HOME}/code/config/aevnmt.json \
	--data_dir ${TMPDIR}/data/setimes.tokenized.en-tr \
	--out_dir ${OUTPUT_DIR} \
	--attention ${hyp} \
	&> ${HOME}/code/output/${session}-${hyp}/log_file
