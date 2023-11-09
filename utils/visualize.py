import os 

ROOT = os.getcwd()
WORK_DIR = os.path.dirname(ROOT)

path_results = os.path.join(WORK_DIR, 'results')
os.makedirs(path_results, exist_ok=True)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__  == '__main__':
    df_det = pd.read_csv(os.path.join(ROOT, 'EAST/results/losses.csv'))
    df_det.drop(columns='Unnamed: 0', axis=1, inplace=True)
    df_rec = pd.read_csv(os.path.join(ROOT, 'CRNN/results/losses.csv'))
    df_rec.drop(columns='Unnamed: 0', axis=1, inplace=True)

    plt.figure(figsize=(20, 20))
    
    losses_det = sns.lineplot(
            data=df_det,
            markers=True, dashes=False
        )
    
    losses_det.figure.savefig(os.path.join(path_results, 'det_training.png'))

    plt.figure(figsize=(20, 20))
    
    losses_rec = sns.lineplot(
        data=df_rec,
        markers=True, dashes=False
    )
    losses_rec.figure.savefig(os.path.join(path_results, 'rec_training.png'))
    

