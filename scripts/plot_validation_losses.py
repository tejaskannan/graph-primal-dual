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


LOG_FILE = 'log.csv'
TRAIN = 'Avg Train Loss'
VALID = 'Avg Valid Loss'
EPOCH = 'Epoch'
TRAIN_SERIES = 'Train'
VALID_SERIES = 'Valid'
X_LABEL = 'Epoch'
Y_LABEL = 'Average Loss (Duality Gap)'

full_title = 'Average Validation Loss per Epoch'

parser = argparse.ArgumentParser(description='Compare cost files.')
parser.add_argument('--model-params', help='Path to Model.', required=True)

args = parser.parse_args()

model_params = load_params(args.model_params)

after = model_params['after']
after_title = 'Average Validation Loss per Epoch after Epoch {0}'.format(after)

losses_df = pd.DataFrame()
names = []

for model in model_params['models']:
    name, path = model['name'], model['path']
    log_path = os.path.join(path, LOG_FILE)
    df = pd.read_csv(log_path)
    valid_df = pd.DataFrame(df['Avg Valid Loss'])
    valid_df = valid_df.rename(mapper={'Avg Valid Loss': name}, axis=1)
    losses_df = pd.concat([losses_df, valid_df], axis=1)
    names.append(name)

output_folder = model_params['save_folder']
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

out_path = os.path.join(output_folder, 'losses')

cmap = cm.get_cmap(name='Spectral')

fig, ax = plt.subplots()
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_title(full_title)

losses_df[names].plot(ax=ax, colormap=cmap)
plt.legend(loc='best')
# plt.hlines(y=0, xmin=0, xmax=len(losses_df), colors='k', linestyles=':')
plt.savefig(out_path + '.pdf')
plt.savefig(out_path + '.pgf')

fig, ax = plt.subplots()
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_title(after_title)

after_df = losses_df[names][after:]
after_df.plot(ax=ax, colormap=cmap)
# plt.hlines(y=0, xmin=after, xmax=len(losses_df), colors='k', linestyles=':')

plt.legend(loc='best')
plt.savefig(out_path + '-after.pdf')
plt.savefig(out_path + '-after.pgf')

