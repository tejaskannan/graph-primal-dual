import numpy as np
import pandas as pd
import json
import argparse
import os
from utils.utils import load_params

COSTS_FILE = 'costs.csv'
OUTPUT_BASE = '../figures'

BASELINE_FIELD = 'Model'
AVG_PERC_FIELD = 'Median % Increase'
STD_PERC_FIELD = 'IQR for %'
AVG_ABS_FIELD = 'Median Increase'
STD_ABS_FIELD = 'IQR'
SERIES_FIELD = 'Graph'

def bold_labels(latex_table):
    lines = []
    table_rows = latex_table.split('\n')
    start = 0
    for row in table_rows:
        if row.startswith('\\') and not row.startswith('\\multirow'):
            lines.append(row)
        else:
            tokens = row.split(' & ')
            bolded = []
            for i, label in enumerate(tokens):
                cleaned = label.replace('\\\\', '').strip()
                if len(cleaned) > 0 and (start <= 1 or i == 0):
                    bolded.append('\\textbf{%s}' % cleaned)
                else:
                    bolded.append(cleaned)

            bolded = [text for i, text in enumerate(bolded) if i == 0 or len(text) > 0]
            if len(bolded) > 1:
                lines.append(' & '.join(bolded) + '\\\\')
            start += 1

    # Fix issue with line headers
    lines[2] = lines[3].replace('\\\\', ' ') + lines[2][1:]
    lines.pop(3)
    return '\n'.join(lines)


parser = argparse.ArgumentParser(description='Compare cost files.')
parser.add_argument('--params', help='Parameters JSON file.', required=True)

args = parser.parse_args()

params = load_params(params_file_path=args.params)

model_folder = params['base_folder']

# Load target dataset into Pandas
costs_file = COSTS_FILE
if 'target_optimizer' in params:
    costs_file = 'costs-{0}.csv'.format(params['target_optimizer'])

target_file = os.path.join(model_folder, params['target_path'], costs_file)
target_df = pd.read_csv(target_file)

output_folder = os.path.join(OUTPUT_BASE, params['output_folder'])
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

formatters = {
    AVG_PERC_FIELD: lambda x: ('%.3f' % x) + '%',
    STD_PERC_FIELD: lambda x: '%.3f' % x,
    AVG_ABS_FIELD: lambda x: '%.3f' % x,
    STD_ABS_FIELD: lambda x: '%.3f' % x
}


baseline_names = [baseline['name'] for baseline in params['baselines']]

df = pd.DataFrame()

target_fields = params['target_fields'] if 'target_fields' in params else params['fields']

for target_field, field in zip(target_fields, params['fields']):
    
    # Initialize data dictionary
    data = {
        SERIES_FIELD: [],
        BASELINE_FIELD: [],
        AVG_ABS_FIELD: [],
        AVG_PERC_FIELD: [],
    }

    print(np.median(target_df[target_field]))

    for baseline in params['baselines']:
        baseline_path = os.path.join(model_folder, baseline['path'], COSTS_FILE)
        baseline_df = pd.read_csv(baseline_path)

        numeric_baseline = pd.to_numeric(baseline_df[field])
        percent_diff = 100 * ((numeric_baseline - target_df[target_field]) / target_df[target_field])
        abs_diff = numeric_baseline - target_df[target_field]

        print(baseline['name'])
        print('Median: {0}'.format(np.median(baseline_df[field])))
        print('Min: {0}'.format(len(np.where(percent_diff < 0))))
        
        data[SERIES_FIELD].append(field)
        data[BASELINE_FIELD].append(baseline['name'])
        data[AVG_PERC_FIELD].append(np.median(percent_diff))
        # data[STD_PERC_FIELD].append(np.percentile(percent_diff, 75) - np.percentile(percent_diff, 25))
        data[AVG_ABS_FIELD].append(np.median(abs_diff))
        # data[STD_ABS_FIELD].append(np.percentile(abs_diff, 75) - np.percentile(abs_diff, 25))     

    field_df = pd.DataFrame.from_dict(data=data)
    df = df.append(field_df)

    # df = pd.DataFrame.from_dict(data=data, orient='columns')
df.set_index([SERIES_FIELD, BASELINE_FIELD], inplace=True)
latex_table = df.to_latex(formatters=formatters, column_format='llcccc', multirow=True, multicolumn=True)
        
field_paths = [field.lower().replace(' ', '-') for field in params['fields']]
out_filename = '-'.join(field_paths)
if 'target_optimizer' in params:
    out_filename += '-' + params['target_optimizer']
out_filename += '.tex'
out_path = os.path.join(output_folder, out_filename)

with open(out_path, 'w') as out_file:
    out_file.write(bold_labels(latex_table))

params_filename = 'params'
if 'target_optimizer' in params:
    params_filename += '-' + params['target_optimizer']
params_filename += '.json'
out_path = os.path.join(output_folder, out_filename)
params_path = os.path.join(output_folder, params_filename)
with open(params_path, 'w') as params_file:
    params_file.write(json.dumps(params))
