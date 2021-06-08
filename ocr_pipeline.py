import cv2

from text_detection import Tflite_Detection
from text_recognition import Tflite_Recognizer

class OCR(object):
    def __init__(self, detect_model, recog_model):
        self.tflite_detection = Tflite_Detection(model_path=detect_model)
        self.tflite_recognizer = Tflite_Recognizer(model_path=recog_model)

    def extract_info(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        list_texts = []

        # detect
        polys = self.tflite_detection.detect_text(image)

        # recog
        for bbox in polys:
            image_text = self.tflite_detection.crop_image(image, bbox)

            preds = self.tflite_recognizer.inference(image_text)
            text = self.tflite_recognizer.ctc_decode(preds)

            list_texts.append(text)
        
        return list_texts

if __name__ == '__main__':
    input_json = {
        'path': "demo.PNG",
    }
    
    model_json = {
        'recog_model': "crnn_float16.tflite",
        'detect_model': "craft_float_320.tflite",
    }

    ocr_model = OCR(detect_model=model_json["detect_model"], recog_model=model_json["recog_model"])
    
    list_texts = ocr_model.extract_info(input_json["path"])
    print(list_texts)