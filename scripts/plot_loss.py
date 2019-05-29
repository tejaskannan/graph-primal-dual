import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os.path

mpl.use('pgf')
pgf_with_pdflatex = {
    'pgf.texsystem': 'lualatex',
    'pgf.rcfonts': False
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

parser = argparse.ArgumentParser(description='Compare cost files.')
parser.add_argument('--model', help='Path to Model.', required=True)

args = parser.parse_args()
log_path = os.path.join(args.model, LOG_FILE)

df = pd.read_csv(log_path)

fig, ax = plt.subplots()
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_title('Average Loss per Epoch')

df[[TRAIN, VALID]].plot(ax=ax)
plt.legend(loc='best')

full_path = os.path.join(args.model, 'loss_full')
plt.savefig(full_path + '.pdf')
plt.savefig(full_path + '.pgf')


fig, ax = plt.subplots()
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_title('Average Loss per Epoch after Epoch 5')

df[[TRAIN, VALID]][5:].plot(ax=ax)
plt.legend(loc='best')

after_path = os.path.join(args.model, 'loss_after')
plt.savefig(after_path + '.pdf')
plt.savefig(after_path + '.pgf')
