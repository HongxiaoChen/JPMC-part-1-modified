#!/bin/bash
#SBATCH --job-name=HMC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=5-0
#SBATCH --mem=100G
cd codes
python _Table1_reproduction.py
python _figure2_reproduction.py
python _figure3_reproduction.py
python _figure4_reproduction.py
python _figure5_reproduction.py
python _figure6_reproduction.py
python _figure7_reproduction.py
python _figure8_reproduction.py
python _figure9_reproduction.py
python _figure10_reproduction.py
python _figure11_reproduction.py



