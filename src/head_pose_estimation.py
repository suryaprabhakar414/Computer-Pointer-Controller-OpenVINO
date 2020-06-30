'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class Head_Pose_Estimation:
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
        
        self.network = IENetwork(model='/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/downloader/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml', weights='/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/downloader/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin')
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
        image_HPE = self.preprocess_input(image)
        self.inference(image_HPE)
        result = self.exec_network.infer(inputs={self.input_blob:image_HPE.astype(np.float32)})
        #result = self.exec_network.requests[0].outputs[self.output_blob][0]
        #coordinates = self.preprocess_output(result,image,image.shape[0],image.shape[1])
    
        return result


    def preprocess_output(self, result, image,height,width):        
        return 
        

        
