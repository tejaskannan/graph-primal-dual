import argparse
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

stat, p_value = stats.ttest_ind(a=df1['Flow Cost'], b=df2['Flow Cost'])
print(stat)
print(p_value)
