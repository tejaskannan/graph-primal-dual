import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import argparse
import os.path
from utils.utils import load_params

mpl.use('pgf')
pgf_with_pdflatex = {
    'pgf.texsystem': 'lualatex',
    'pgf.rcfonts': False,
    'font.family': 'serif'
}
mpl.rcParams.update(pgf_with_pdflatex)
mpl.rcParams.update({'errorbar.capsize': 3})
mpl.rcParams.update({'font.size': 14})

COST_FIELD = 'Flow Cost'
LOG_PATH = 'costs.csv'

parser = argparse.ArgumentParser(description='Comparing running times.')
parser.add_argument('--params', help='Path to params file.', required=True)

params = load_params(parser.parse_args().params)
model_folder = params['base_folder']

field = params['fields'][0]

target_path = os.path.join(model_folder, params['target_path'], 'costs.csv')
target_df = pd.read_csv(target_path)

for baseline in params['baselines']:
    baseline_path = os.path.join(model_folder, baseline['path'], 'costs.csv')
    baseline_df = pd.read_csv(baseline_path)

    numeric_baseline = pd.to_numeric(baseline_df[field])
    percent_diff = 100 * ((numeric_baseline - target_df[field]) / target_df[field])

df = pd.DataFrame(percent_diff)
ax, bp = df.boxplot(column=field, notch=0, sym='', medianprops={'color': 'red'}, return_type='both')

cmap = cm.get_cmap('Spectral')
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color=cmap(0.9))

fill_map = cm.get_cmap('Blues')
for patch in bp['boxes']:
    xdata, ydata = patch.get_data()
    color = fill_map(0.1)
    ax.fill_between(xdata, ydata, facecolor=color)
    # patch.set(color='black', fillstyle='full')

print(np.count_nonzero(np.where(df['Flow Cost'] < 0)))


ax.grid(False)
ax.set_xlabel('')
ax.set_ylabel('Median Percent Increase')
ax.set_title('Flow Costs for Gated GAT Relative to the\nNeighborhood Model on NYC-1000')
ax.set_xticks([])
ax.set_yticklabels(['%1.1f%%' % i for i in ax.get_yticks()])

# plt.show()

plt.savefig(params['output_folder'] + '.pdf')
plt.savefig(params['output_folder'] + '.pgf')
