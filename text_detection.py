import cv2
import os
import torch
import numpy as np
import tensorflow as tf

from torch.autograd import Variable
from utils.craft_helper import resize_aspect_ratio, normalizeMeanVariance, getDetBoxes, adjustResultCoordinates

class Tflite_Detection(object):
    def __init__(self, model_path, text_threshold = 0.7, link_threshold = 0.3,\
                 low_text = 0.4, canvas_size = 320, mag_ratio = 1.5, poly=False):
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly

        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()

    def inference_craft(self, input_image):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()

        y = self.interpreter.get_tensor(self.output_details[0]['index'])
        feature = self.interpreter.get_tensor(self.output_details[1]['index'])

        return y, feature

    # format polys tl, tr, br, bl
    def draw_image(self, raw_image, polys, alpha=3):
        h, w, _ = raw_image.shape
        for bbox in polys:
            bbox = np.array(bbox).astype(np.int32).reshape((-1))
            bbox = bbox.reshape(-1, 2)
    
            min_coord = [int(bbox[0][0]* (w/self.canvas_size)), int(bbox[0][1]* (h/self.canvas_size))-alpha]
            max_coord = [int(bbox[2][0]* (w/self.canvas_size)), int(bbox[2][1]* (h/self.canvas_size))]

            cv2.rectangle(raw_image, min_coord, max_coord, (0, 0, 255), 2)
        
        return raw_image

    def crop_image(self, raw_image, bbox, alpha=3): # alpha: set: ymin - alpha
        h, w, _ = raw_image.shape
        min_coord = [int(bbox[0][0]* (w/self.canvas_size)), int(bbox[0][1]* (h/self.canvas_size))-alpha]
        max_coord = [int(bbox[2][0]* (w/self.canvas_size)), int(bbox[2][1]* (h/self.canvas_size))]

        return raw_image[min_coord[1]:max_coord[1], min_coord[0]:max_coord[0]]

    def detect_text(self, image_rgb):
        image = cv2.resize(image_rgb, dsize=(self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LINEAR)
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, \
                        self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)

        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        image_data = normalizeMeanVariance(img_resized)
        image_data = torch.from_numpy(image_data).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        image_data = Variable(image_data.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

        image_data = image_data.cpu().detach().numpy()
        y, feature = self.inference_craft(image_data)

        y = torch.from_numpy(y)
        feature = torch.from_numpy(feature)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        
        return polys

if __name__ == '__main__':
    tflite_detection = Tflite_Detection("craft_float_320.tflite")

    image_path = 'demo.PNG'
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    polys = tflite_detection.detect_text(image)

    cv2.imshow("image", tflite_detection.draw_image(image, polys))
    cv2.waitKey(0) 