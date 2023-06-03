
# Set up conda env
conda create -n swreg python=3.9
conda activate swreg
pip install -r requirements.txt --no-cache-dir

# Open Jupyter Lab
jupyter-lab --no-browser --port=8880 --ip='0.0.0.0'

# train_datasets.py
Subword regularization is done here

# ner_finetune.ipynb 
Main notebook: the only thing that is needed to run