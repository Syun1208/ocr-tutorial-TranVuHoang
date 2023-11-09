import pyarrow.parquet as pq
import os
import pandas as pd
import glob
import tqdm

ROOT = os.getcwd()
WORK_DIR = os.path.dirname(ROOT)


def get_images_from_lfs(path_read: str):

    table = pq.read_table(path_read)
    save_datasets_dir = os.path.join(ROOT, 'images')
    os.makedirs(save_datasets_dir, exist_ok=True)
    splited_datasets_save = os.path.join(save_datasets_dir, path_read.split('/')[-1].split('-')[0])
    os.makedirs(splited_datasets_save, exist_ok=True)
    save_labels_dir = os.path.join(ROOT, 'labels')
    os.makedirs(save_labels_dir, exist_ok=True)
    splited_labels_save = os.path.join(save_labels_dir, path_read.split('/')[-1].split('-')[0])
    os.makedirs(splited_labels_save, exist_ok=True)

    index = 0
    for img in table['image']:
        b = img['bytes'].as_py()
        with open(os.path.join(splited_datasets_save, f'{index}.png'), 'wb') as f_datasets:
            f_datasets.write(b)
            index+=1

    index = 0
    for label in table['ground_truth']:
        converted_json = str(label)
        with open(os.path.join(splited_labels_save, f'{index}.json'), 'w') as f_labels:
            f_labels.write(converted_json)
            index+=1
    f_datasets.close()
    f_labels.close()

if __name__ == '__main__':
    # Please clone link to get file parquet: git clone https://huggingface.co/datasets/naver-clova-ix/cord-v1
    
    lfs_dir = sorted(glob.glob(os.path.join(ROOT, 'cord-v1/data/*.parquet')))

    for parquet in tqdm.tqdm(lfs_dir, colour='red', desc='Extracting datasets from cloud'):
        get_images_from_lfs(parquet)

