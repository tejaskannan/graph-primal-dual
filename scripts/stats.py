import argparse
import numpy as np
import scipy.stats as stats
import os
import pandas as pd
from utils.utils import load_params


parser = argparse.ArgumentParser(description='Compare cost files.')
parser.add_argument('--model-params', help='Path to Model.', required=True)

args = parser.parse_args()

model_params = load_params(args.model_params)

name1, model1_path = model_params['model1']['name'], model_params['model1']['path']
name2, model2_path = model_params['model2']['name'], model_params['model2']['path']

df1 = pd.read_csv(os.path.join(model1_path, 'costs.csv'))
df2 = pd.read_csv(os.path.join(model2_path, 'costs.csv'))

fc1 = df1['Flow Cost']
fc2 = df2['Flow Cost']

print('{0} {1}'.format(np.average(df1['Flow Cost']), np.std(df1['Flow Cost'])))
print('{0} {1}'.format(np.average(df2['Flow Cost']), np.std(df2['Flow Cost'])))

diff = df1['Flow Cost'] - df2['Flow Cost']
stat, p_value = stats.wilcoxon(x=diff)
print('t-stat: {0}'.format(stat))
print('p-value: {0}'.format(p_value))
