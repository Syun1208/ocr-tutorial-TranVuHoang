import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import glob

ROOT = os.getcwd()
WORK_DIR = os.path.dirname(ROOT)

def visualize(df, type='text_detection'):
    plt.figure(figsize=(20, 10))
    ax = df.plot.bar()

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.ylabel('Frequency')
    ax.axes.get_xaxis().set_visible(False)
    plt.xlabel('Object')
    plt.title(f'The amount of {type} datasets')
    plt.savefig(os.path.join(WORK_DIR, f'results/eda_{type}.png'))

if __name__ == '__main__':
    data_types = ['train', 'validation', 'test']
    dict_det = dict()
    dict_rec = dict()
    for data_type in data_types:
        len_data_det = len(glob.glob(os.path.join(WORK_DIR, f'data/naver-clova-ix-det/{data_type}_img/*.png'))) 
        len_data_rec = len(glob.glob(os.path.join(WORK_DIR, f'data/naver-clova-ix-rec/{data_type}_img/*.jpg')))  
        dict_det[f'{data_type}_img'] = len_data_det
        dict_rec[f'{data_type}_img'] = len_data_rec
    
    df_det = pd.DataFrame(dict_det, index=[0])
    print(df_det.iloc[:, 0].values[0])
    df_rec = pd.DataFrame(dict_rec, index=[0])
    print(df_det)
    print(df_rec)
    visualize(df_det)
    visualize(df_rec, 'text recognition')
