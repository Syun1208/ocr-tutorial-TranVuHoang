import sys

sys.path.append('../')

import torch
from typing import Any
from PIL import Image
import os
from pathlib import Path

# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)

sys.path.append(os.path.join(ROOT, 'EAST'))

from EAST.model import EAST
from EAST.dataset import get_rotate_mat
from EAST.detect import detector, plot_boxes

path_results = os.path.join(WORK_DIR, 'results')
os.makedirs(path_results, exist_ok=True)
det_results = os.path.join(path_results, 'detection')
os.makedirs(det_results, exist_ok=True)

class TextDetection:
    
    def __init__(self, model_path) -> None:
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = EAST().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        

    def predict(self, image_path, *args: Any, **kwds: Any) -> Any:
        
        image = Image.open(image_path)
        boxes = detector(image, self.model, self.device)
        detected_image = plot_boxes(image, boxes)
        detected_image.save(os.path.join(det_results, image_path.split('/')[-1]))
        print('Saved in: ', os.path.join(det_results, image_path.split('/')[-1]))
        
        return boxes
    

if __name__ == '__main__':
    image_path = os.path.join(WORK_DIR, 'data/naver-clova-ix-det/test_img/10.png')
    text_detection = TextDetection(model_path=os.path.join(ROOT, 'EAST/pths/model_epoch_100.pth'))
    boxes = text_detection.predict(image_path)
    print(boxes)