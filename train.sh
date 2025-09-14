#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -n 20
#SBATCH --mem=20000
#SBATCH --gres=gpu:2
#SBATCH -p short
#SBATCH -t 23:59:00
#SBATCH --mail-user=ksrivastava1@wpi.edu
#SBATCH --output=/home/ksrivastava1/1xgpt/slurm-%j.out 


source venv/bin/activate
python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --output_dir data/genie_model --max_eval_steps 10