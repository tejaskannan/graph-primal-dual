import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os.path
from adjustText import adjust_text

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

### Graph Style choices ###
name = 'Neighborhood-Sparsemax'
full_title = 'Average Loss per Epoch for {0}'.format(name)
after = 5
after_title = 'Average Loss per Epoch after Epoch {0} for {1}'.format(after, name)
###########################

parser = argparse.ArgumentParser(description='Compare cost files.')
parser.add_argument('--model', help='Path to Model.', required=True)

args = parser.parse_args()
log_path = os.path.join(args.model, LOG_FILE)

df = pd.read_csv(log_path)

fig, ax = plt.subplots()
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_title(full_title)

df[[TRAIN, VALID]].plot(ax=ax)
plt.legend(loc='best')

# annotate minimum validation loss
ymin, xmin = df[VALID].min(), df[VALID].idxmin()
ax.plot([xmin], [ymin], 'ko', markersize=2.0)
text = ax.text(s='({0}, {1})'.format(xmin, round(ymin, 3)), x=xmin, y=ymin, fontsize=8)

valid_xs = list(np.arange(len(df[VALID])))
valid_ys = list(df[VALID].to_numpy())
train_xs = list(np.arange(len(df[TRAIN])))
train_ys = list(df[TRAIN].to_numpy())
adjust_text([text], [xmin] + valid_xs + train_xs, [ymin] + valid_ys + train_ys)

full_path = os.path.join(args.model, 'loss_full')
plt.savefig(full_path + '.pdf')
plt.savefig(full_path + '.pgf')

# Create figure by truncating early losses
fig, ax = plt.subplots()
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_title(after_title)

df[[TRAIN, VALID]][after:].plot(ax=ax)
plt.legend(loc='best')

ymin, xmin = df[VALID][after:].min(), df[VALID][after:].idxmin()
text = ax.text(s='({0}, {1})'.format(xmin, round(ymin, 3)), x=xmin, y=ymin, fontsize=8)
ax.plot([xmin], [ymin], 'ko', markersize=2.0)

valid_xs = list(np.arange(len(df[VALID][after:])))
valid_ys = list(df[VALID][after:].to_numpy())
train_xs = list(np.arange(len(df[TRAIN][after:])))
train_ys = list(df[TRAIN][after:].to_numpy())
adjust_text([text], [xmin] + valid_xs + train_xs, [ymin] + valid_ys + train_ys)

after_path = os.path.join(args.model, 'loss_after')
plt.savefig(after_path + '.pdf')
plt.savefig(after_path + '.pgf')
