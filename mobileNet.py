import tensorflow as tf
import cv2
import numpy as np

class MobileNet(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()

    def read_image(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(image, 0)
        image = np.float32(image)

        return image

    def inference(self, image_path):
        input_image = self.read_image(image_path)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output

if __name__ =='__main__':
    input_json = {
        'path': "demo.PNG",
    }
    
    model_json = {
        'model': "mobilenet_v1_1.0_224.tflite",
    }

    model = MobileNet(model_path=model_json["model"])

    output = model.inference(input_json["path"])
    print(output)