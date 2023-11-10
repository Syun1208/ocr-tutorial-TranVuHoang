import sys

sys.path.append('../')

from typing import Any
import torch
import os
from torch.autograd import Variable
import argparse
from PIL import Image
import cv2
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

sys.path.append(os.path.join(ROOT, 'CRNN'))

from CRNN import utils, params, dataset
from CRNN.models.crnn import CRNN


path_results = os.path.join(WORK_DIR, 'results')
os.makedirs(path_results, exist_ok=True)
rec_results = os.path.join(path_results, 'recognition')
os.makedirs(rec_results, exist_ok=True)

class TextRecognition:
    
    def __init__(self, model_path) -> None:
        
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        nclass = len(params.alphabet) + 1
        self.model = CRNN(params.imgH, params.nc, nclass, params.nh)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if params.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transformer = dataset.resizeNormalize((100, 32))
        self.converter = utils.strLabelConverter(params.alphabet) 
        
               
    def predict(self, image_path, *args: Any, **kwds: Any) -> Any:
        
        if type(image_path) == str:  
            image = Image.open(image_path).convert('L')
        else:
            image = Image.fromarray(image_path).convert('L')
            
        image = self.transformer(image)

        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.LongTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))
        
        if type(image_path) == str:  
            with open(os.path.join(rec_results, os.path.splitext(image_path.split('/')[-1])[0] + '.txt'), 'w+') as text_save:
                text_save.write(sim_pred)
        
            print('Saved in: ', os.path.join(rec_results, os.path.splitext(image_path.split('/')[-1])[0] + '.txt'))
        
        return sim_pred


if __name__ == "__main__":
    image_path = os.path.join(WORK_DIR, 'data/naver-clova-ix-rec/test_img/-40.000%_7.jpg')
    image = cv2.imread(image_path)
    text_recognition = TextRecognition(model_path=os.path.join(ROOT, 'CRNN/pths/CRNN_99_250.pth'))
    text = text_recognition.predict(image)
    print(text)
