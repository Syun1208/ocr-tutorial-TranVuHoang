import sys
import os

ROOT = os.getcwd()
WORK_DIR = os.path.dirname(ROOT)

sys.path.append('../')
sys.path.append(os.path.join(ROOT, 'utils'))

from utils.predict_e2e import TextE2E
from utils.predict_det import TextDetection
from utils.predict_rec import TextRecognition


if __name__ == '__main__':
    
    # Text Detection
    image_path = os.path.join(ROOT, 'data/naver-clova-ix-det/test_img/2.png')
    text_detection = TextDetection(model_path=os.path.join(ROOT, 'utils/EAST/pths/model_epoch_100.pth'))
    boxes = text_detection.predict(image_path)
    print(boxes)
    
    # Text Recognition
    image_path = os.path.join(ROOT, 'data/naver-clova-ix-rec/test_img/-40.000%_7.jpg')
    text_recognition = TextRecognition(model_path=os.path.join(ROOT, 'utils/CRNN/pths/CRNN_99_250.pth'))
    text = text_recognition.predict(image_path)
    print(text)

    # Text E2E: Text Detection + Recognition
    image_path = os.path.join(ROOT, 'data/naver-clova-ix-det/test_img/2.png')
    text_e2e = TextE2E(model_det_path=os.path.join(ROOT, 'utils/EAST/pths/model_epoch_100.pth'), model_rec_path=os.path.join(ROOT, 'utils/CRNN/pths/CRNN_99_250.pth'))
    boxes, texts = text_e2e.predict(image_path)
    print(boxes)
    print(texts)

    