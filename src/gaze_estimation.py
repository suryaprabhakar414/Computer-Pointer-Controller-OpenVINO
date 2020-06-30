'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import math 

class Gaze_Estimation:
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
        self.input_shapes = {}

    def load_model(self):
        
        self.network = IENetwork(model='/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/downloader/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml', weights='/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/downloader/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.bin')
        self.plugin = IECore()
        self.exec_network = self.plugin.load_network(self.network,"CPU")
        self.input_blob  = list(self.network.inputs.keys())
        self.output_blob = next(iter(self.network.outputs))
        
        for i in self.input_blob:
            self.input_shapes[i] = self.network.inputs[i].shape
    
    
        
    def get_input_shape(self):
    
        return self.input_shapes

    def get_output_shape(self):
        return self.network.outputs[self.output_blob].shape
    
    def preprocess_input(self, image, name):
        #print(self.input_shapes['left_eye_image'])
        
        input_shape = self.input_shapes[name]
        image = cv2.resize(image,(input_shape[3],input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1,*image.shape)
        return image

    def check_model(self):
        raise NotImplementedError

    
    def inference(self, info):
        #self.exec_network.start_async(request_id=0,inputs={self.input_blob:image.astype(np.float32)})
        self.exec_network.infer(inputs=info)
        return
        
    
    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status
    
    def predict(self, left_eye_image,right_eye_image,head_pose_angles):
        
        left_eye = self.preprocess_input(left_eye_image,'left_eye_image')
        right_eye = self.preprocess_input(right_eye_image,'right_eye_image')
        head_poses = []
        for i in head_pose_angles:
            head_poses.append(head_pose_angles[i])
        head_poses = np.array(head_poses)
        info = {'left_eye_image':left_eye,'right_eye_image':right_eye,'head_pose_angles':head_poses}
        self.inference(info)
        
        mouse_coordinates,gaze_vector = self.preprocess_output(head_poses)

        return mouse_coordinates,gaze_vector


    def preprocess_output(self, head_poses):  
        result = self.exec_network.requests[0].outputs[self.output_blob][0]
        roll = head_poses[2]
        cos_x = math.cos(roll*math.pi/180.0)
        sin_x = math.sin(roll*math.pi/180.0)
        mouse_x = result[0]*cos_x + result[1]*sin_x
        mouse_y = result[1]*cos_x - result[0]*sin_x
        return (mouse_x,mouse_y),result
        

        
