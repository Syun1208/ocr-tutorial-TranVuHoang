# Data preprocessing
'''
p1,p2,p3,p4,p5,p6,p7,p8,w
'''
import glob
import shutil
from IPython.display import clear_output
import sys
import os
import tqdm
import json
import cv2
sys.path.append('../')

ROOT = os.getcwd()
WORK_DIR = os.path.dirname(ROOT)


def copy_images(source_folder, destination_folder):
    """
    Copy image files from the source folder to the destination folder.

    Args:
        source_folder (str): The path to the source folder containing image files.
        destination_folder (str): The path to the destination folder where images will be copied.
    """
    # Ensure the destination folder exists, or create it if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith((".jpg", ".png", ".gif", ".bmp", ".jpeg"))]

    # Loop through the image files and copy them to the destination folder
    for image_file in image_files:
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)

        # Copy the image file to the destination folder
        shutil.copy2(source_path, destination_path)


class DataPreprocessing:
    '''
    Folder trees:

      - datasetes_dir:

        /content/datasets
        .
        ├── test
        ├── train
        └── validation

      - annotation_dir:
        /content/labels
        .
        ├── test
        ├── train
        └── validation
    '''
    def __init__(self, datasets_dir, annotation_dir):
        self.datasets_dir = datasets_dir
        self.annotation_dir = annotation_dir
        self.list_folder_images = os.listdir(self.datasets_dir)
        self.list_folder_labels = os.listdir(self.annotation_dir)

    def data_recognition(self):

      for type_folder in tqdm.tqdm(self.list_folder_images, desc='Converting to CRNN Labels: ', colour='green'):

        clear_output(wait=True)

        abs_folder_images = os.path.join(self.datasets_dir, type_folder)
        abs_folder_labels = os.path.join(self.annotation_dir, type_folder)

        folder_crnn = os.path.join(WORK_DIR, 'data/naver-clova-ix-rec')
        os.makedirs(folder_crnn, exist_ok=True)
        folder_crnn_images = os.path.join(folder_crnn, f'{type_folder}_img/')
        os.makedirs(folder_crnn_images, exist_ok=True)

        for json_path in os.listdir(abs_folder_labels):
          abs_json_path = os.path.join(abs_folder_labels, json_path)
          image = cv2.imread(os.path.join(abs_folder_images, json_path.replace('json', 'png')))

          if json_path == '114.json' or json_path == '79.json' or json_path == '34.json':
              continue

          with open(abs_json_path, 'r') as f:
              json_data = json.loads(f.read())

          words_info = json_data["valid_line"]

          index=0
          for word_info in words_info:
              for word in word_info["words"]:
                  x1, y1 = int(word["quad"]["x1"]), int(word["quad"]["y1"])
                  x2, y2 = int(word["quad"]["x2"]), int(word["quad"]["y2"])
                  x3, y3 = int(word["quad"]["x3"]), int(word["quad"]["y3"])
                  x4, y4 = int(word["quad"]["x4"]), int(word["quad"]["y4"])


                  top_left_x = min([x1,x2,x3,x4])
                  top_left_y = min([y1,y2,y3,y4])
                  bot_right_x = max([x1,x2,x3,x4])
                  bot_right_y = max([y1,y2,y3,y4])

                  cropped_image = image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

                  text = word["text"]

                  cv2.imwrite(os.path.join(folder_crnn_images, f'{text}_{index}.jpg'), cropped_image)
                  index = index + 1

      print("Converted successfully!")

    def data_detection(self):

        for type_folder in tqdm.tqdm(self.list_folder_images, desc='Converting to EAST Labels: ', colour='green'):

            clear_output(wait=True)

            abs_folder_images = os.path.join(self.datasets_dir, type_folder)
            abs_folder_labels = os.path.join(self.annotation_dir, type_folder)

            folder_east = os.path.join(WORK_DIR, 'data/naver-clova-ix-det')

            os.makedirs(folder_east, exist_ok=True)
            folder_east_images = os.path.join(folder_east, f'{type_folder}_img/')
            os.makedirs(folder_east_images, exist_ok=True)
            folder_east_labels = os.path.join(folder_east, f'{type_folder}_gt/')
            os.makedirs(folder_east_labels, exist_ok=True)

            copy_images(abs_folder_images, folder_east_images)

            for json_path in os.listdir(abs_folder_labels):
              abs_json_path = os.path.join(abs_folder_labels, json_path)
              file_east_label = open(os.path.join(folder_east_labels, os.path.splitext(abs_json_path.split('/')[-1])[0] + '.txt'), 'w+')
              with open(abs_json_path, 'r') as f:
                  json_data = json.loads(f.read())

              words_info = json_data["valid_line"]

              for word_info in words_info:
                  for word in word_info["words"]:
                      x1, y1 = int(word["quad"]["x1"]), int(word["quad"]["y1"])
                      x2, y2 = int(word["quad"]["x2"]), int(word["quad"]["y2"])
                      x3, y3 = int(word["quad"]["x3"]), int(word["quad"]["y3"])
                      x4, y4 = int(word["quad"]["x4"]), int(word["quad"]["y4"])

                      file_east_label.write(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{word["text"]}\n')


        print("Converted successfully!")

if __name__ == '__main__':
    path_images = os.path.join(WORK_DIR, 'data/images/')
    path_labels = os.path.join(WORK_DIR, 'data/labels/')

    data_preprocessing = DataPreprocessing(datasets_dir=path_images, annotation_dir=path_labels)
    data_preprocessing.data_detection()
    data_preprocessing.data_recognition()
    