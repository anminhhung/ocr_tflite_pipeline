import cv2
import torch
import numpy as np
import os 
import tensorflow as tf

from pydantic import BaseModel


class Tflite_Recognizer(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()

    def read_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image/127.5 - 1.0
        image = cv2.resize(image,(100, 32),interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        image = np.float32(image)

        return image

    def inference(self, image):
        input_image = self.read_image(image)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output

    def ctc_decode(self, preds, characters="0123456789abcdefghijklmnopqrstuvwxyz@/:.-"):
        pred_index = np.argmax(preds, axis=2)
        char_list = list(characters)
        char_dict = {}

        for i, char in enumerate(char_list):
            char_dict[char] = i + 1

        char_list = ['_'] + char_list
        BLANK = 0
        texts = []
        output = pred_index[0, :]
        characters = []
        for i in range(preds.shape[1]):
            if output[i] != BLANK and (not (i > 0 and output[i - 1] == output[i])):
                characters.append(char_list[output[i]])
            text = ''.join(characters)
            
        return text

if __name__ == '__main__':
    input_json = {
        'path': "1AB8.jpg",
    }
    
    model_json = {
        'model': "crnn_float16.tflite"
    }
    
    tflite_recognizer = Tflite_Recognizer(model_path=model_json["model"])

    # predict
    image = cv2.imread(input_json["path"])
    preds = tflite_recognizer.inference(image)
    text = tflite_recognizer.ctc_decode(preds)
    
    print("Predicted:  ", text)