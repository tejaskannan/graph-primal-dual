import subprocess
import os
from utils.utils import load_params

params_folder = 'model_params'
datasets_folder = 'datasets'
params = ['london_neighborhood_sparsemax_exp.json', 'london_gated_gat_sparsemax_exp.json']

# Ensure that all files exist
for params_file in params:
    params_path = os.path.join(params_folder, params_file)
    assert os.path.exists(params_path)

    params_dict = load_params(params_path)
    assert os.path.exists(os.path.join(datasets_folder, params_dict['model']['dataset_name']))

print('Checked all files.')

for params_file in params:
    params_path = os.path.join(params_folder, params_file)
    subprocess.run(['python', 'main.py', '--train', '--params', params_path])
