# RL_ForestCarbonSequestration

conda create -n tfgpu tensorflow python=3.6.8 \
conda install tensorflow-gpu==1.14 (or just tensorflow) \
pip install gym \
git clone https://github.com/openai/baselines.git \
cd baselines/ \
pip install -e . \

python env.py \
sbatch --job-name=mcm --gres=gpu:2 --partition=gpu -t 1-0 --mem=30G -o 0219.txt ./run.sh \
sbatch --job-name=mcm -t 1-0 --cpus-per-task=32 -o 0219.txt --mem=30G ./run.sh \
