#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=gengpu                # GPU partition
#SBATCH --time=40:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=8                         # CPU cores
#SBATCH --mem=64G                          # RAM
#SBATCH --gres=gpu:a100:1                 # 1 A100 GPU
#SBATCH --job-name=MIL_train              # Job name (no spaces)
#SBATCH --output=/home/pab1998/Code9_no_leakage/logs/%x_%j.out
#SBATCH --error=/home/pab1998/Code9_no_leakage/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=parisbozzuti2026@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /home/pab1998/Code9_no_leakage/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd /home/pab1998/Code9_no_leakage

# Best hyperparams from Optuna Trial 5 (val loss: 0.2255)
# Config already updated: lr=5.225e-05, weight_decay=4.696e-04,
# patch_proj_dropout=0.511, classifier_dropout=0.198,
# class_weight_benign=2.137, entropy_lambda=9.976e-04

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /home/pab1998/Code9_no_leakage/no_train_val_leakage_splits/data_splits_new_03.npz \
