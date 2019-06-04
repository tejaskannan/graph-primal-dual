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

COST_FIELD = 'Flow Cost'
LOG_PATH = 'costs.csv'

parser = argparse.ArgumentParser(description='Comparing running times.')
parser.add_argument('--params', help='Path to params file.', required=True)

params = load_params(parser.parse_args().params)

xs = []
ys = {}
upper_errors = {}
lower_errors = {}
for series in params['series']:
    xs.append(series['name'])

    target_df = pd.read_csv(os.path.join(series['target']['path'], LOG_PATH))
    target_costs = target_df[COST_FIELD]

    for group in series['groups']:
        df = pd.read_csv(os.path.join(group['path'], LOG_PATH))

        if group['name'] not in ys:
            ys[group['name']] = []
            upper_errors[group['name']] = []
            lower_errors[group['name']] = []

        percent_diff = (target_costs - df[COST_FIELD]) / target_costs
        median = np.median(percent_diff)
        ys[group['name']].append(median)
        upper_errors[group['name']].append(np.percentile(percent_diff, 75) - median)
        lower_errors[group['name']].append(median - np.percentile(percent_diff, 25))

##### STYLE CHOICES #####
ind = np.arange(len(xs))  # the x locations for the groups
width = 0.2  # the width of the bars
#########################

fig, ax = plt.subplots()
cmap = cm.get_cmap('Spectral')

shift = 1
offsets = np.linspace(start=-shift, stop=shift, num=len(ys), endpoint=True)
colors = np.linspace(start=0, stop=1, num=len(ys), endpoint=True)
for i, name in enumerate(ys.keys()):
    errors = [lower_errors[name], upper_errors[name]]
    ax.bar(ind + offsets[i] * width, ys[name], width, yerr=errors, label=name, color=cmap(colors[i]))

ax.set_xticks(ind)
ax.set_xticklabels(xs)
ax.legend(prop={'size': 8})

ax.set_yticklabels(['%1.1f%%' % (i * 100) for i in ax.get_yticks()])

ax.set_xlabel('Cost Function')
ax.set_ylabel('Median Percent Decrease')
ax.set_title(params['title'])

plt.axhline(0, color='k', linestyle=':')

plt.savefig(params['output_file'] + '.pdf')
plt.savefig(params['output_file'] + '.pgf')
