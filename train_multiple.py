import subprocess
import os

params_folder = 'model_params'
params = ['cambridge_neighborhood_softmax_quadratic.json', 'cambridge_neighborhood_sparsemax_quadratic.json',
          'sf_neighborhood_softmax_quadratic.json', 'sf_neighborhood_sparsemax_quadratic.json',
          'sf_gated_gat_softmax_quadratic.json', 'sf_neighborhood_softmax_cubic.json',
          'sf_neighborhood_sparsemax_cubic.json', 'sf_gated_gat_softmax_cubic.json']

# Ensure that all files exist
for params_file in params:
    assert os.path.exists(os.path.join(params_folder, params_file))

print('Checked all files.')

for params_file in params:
    params_path = os.path.join(params_folder, params_file)
    subprocess.run(['python', 'main.py', '--train', '--params', params_path])
