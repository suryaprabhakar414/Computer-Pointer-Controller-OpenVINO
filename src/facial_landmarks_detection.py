'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class Facial_Landmarks_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self):
        
        self.network = IENetwork(model='/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml', weights='/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin')
        self.plugin = IECore()
        self.exec_network = self.plugin.load_network(self.network,"CPU")
        self.input_blob  = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape

    def get_output_shape(self):
        return self.network.outputs[self.output_blob].shape
    
    def preprocess_input(self, image):
        input_shape = self.get_input_shape()
        image = cv2.resize(image,(input_shape[3],input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1,*image.shape)
        return image

    def check_model(self):
        raise NotImplementedError

    
    def inference(self, image):
        #self.exec_network.start_async(request_id=0,inputs={self.input_blob:image.astype(np.float32)})
        self.exec_network.infer(inputs={self.input_blob:image.astype(np.float32)})
        return
        
    
    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status
    
    def predict(self, image):
        image_FLD = self.preprocess_input(image)
        self.inference(image_FLD)
        result = self.exec_network.requests[0].outputs[self.output_blob][0]
        coordinates = self.preprocess_output(result,image,image.shape[0],image.shape[1])
        left_eye_x_min = coordinates['left_eye'][0]-10
        left_eye_x_max = coordinates['left_eye'][0]+10
        left_eye_y_min = coordinates['left_eye'][1]-10
        left_eye_y_max = coordinates['left_eye'][1]+10

        right_eye_x_min = coordinates['right_eye'][0]-10
        right_eye_x_max = coordinates['right_eye'][0]+10
        right_eye_y_min = coordinates['right_eye'][1]-10
        right_eye_y_max = coordinates['right_eye'][1]+10

        
        left_eye_image = image[left_eye_x_min:left_eye_x_max, left_eye_y_min:left_eye_y_max]
        right_eye_image = image[right_eye_x_min:right_eye_x_max, right_eye_y_min:right_eye_y_max]
        
        eye_coordinates = {'left_eye':[left_eye_x_min,left_eye_y_min,left_eye_x_max,left_eye_y_max],
        'right_eye':[right_eye_x_min,right_eye_y_min,right_eye_x_max,right_eye_y_max]}




        return left_eye_image,right_eye_image,eye_coordinates


    def preprocess_output(self, result, image,height,width):
        left_eye_x = int(result[0]*width)
        left_eye_y = int(result[1]*height)
        right_eye_x = int(result[2]*width)
        right_eye_y = int(result[3]*height)

        coordinates = {'left_eye':[left_eye_x,left_eye_y],'right_eye':[right_eye_x,right_eye_y]}
        
        return coordinates
        

        
