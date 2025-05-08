ENV_NAME=euro_par_artifact

conda env remove -n $ENV_NAME
conda create -n $ENV_NAME python=3.11.9
conda activate $ENV_NAME

pip install -e human-eval-infilling
pip install vllm
pip install jsonlines
pip install fuzzywuzzy
pip install python-Levenshtein

python scripts/download_model.py