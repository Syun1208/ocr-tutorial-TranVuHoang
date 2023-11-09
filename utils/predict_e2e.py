import sys
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

sys.path.append('../')
sys.path.append(os.path.join(ROOT, 'EAST'))
sys.path.append(os.path.join(ROOT, 'CRNN'))

from typing import Any
import os
import numpy as np
import cv2
from predict_det import TextDetection
from predict_rec import TextRecognition


path_results = os.path.join(WORK_DIR, 'results')
os.makedirs(path_results, exist_ok=True)
e2e_results = os.path.join(path_results, 'e2e')
os.makedirs(e2e_results, exist_ok=True)

class TextE2E:
    
    
    def __init__(self, model_det_path, model_rec_path) -> None:
        
        self.text_detection = TextDetection(model_path=model_det_path)
        self.text_recognition = TextRecognition(model_path=model_rec_path)
        
        self.predicted_texts = list()
        
    def predict(self, image_path, *args: Any, **kwds: Any) -> Any:
        
        image = cv2.imread(image_path)

        predicted_boxes = self.text_detection.predict(image_path)
        
        for box in predicted_boxes:
            
            box = list(map(lambda x: int(x), box.tolist()))
           
            x1, y1, x2, y2, x3, y3, x4, y4 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]
            
            top_left_x = min([x1,x2,x3,x4])
            top_left_y = min([y1,y2,y3,y4])
            bot_right_x = max([x1,x2,x3,x4])
            bot_right_y = max([y1,y2,y3,y4])
            
            cropped_image = image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

            predicted_text = self.text_recognition.predict(cropped_image)
            self.predicted_texts.append(predicted_text)
            
            cv2.polylines(image, [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)], isClosed=True,
                      color=(255, 0, 0), thickness=2)
            cv2.putText(image, predicted_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imwrite(os.path.join(e2e_results, image_path.split('/')[-1]), image) 
        
        return predicted_boxes, self.predicted_texts
    
if __name__ == '__main__':
    image_path = os.path.join(WORK_DIR, 'data/naver-clova-ix-det/test_img/2.png')
    text_e2e = TextE2E(model_det_path=os.path.join(ROOT, 'EAST/pths/model_epoch_100.pth'), model_rec_path=os.path.join(ROOT, 'CRNN/pths/CRNN_99_250.pth'))
    boxes, texts = text_e2e.predict(image_path)
    print(boxes)
    print(texts)