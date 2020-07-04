'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import math 

class Gaze_Estimation:
    
    def __init__(self, device, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.device = device
        self.extensions = extensions
        self.input_shapes = {}

    def load_model(self,model_path):
        model = model_path+"gaze-estimation-adas-0002.xml"
        weights = model_path+"gaze-estimation-adas-0002.bin"
        self.network = IENetwork(model=model,weights = weights)
        self.plugin = IECore()
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)
        self.check_model()
        self.exec_network = self.plugin.load_network(self.network,self.device)
        self.input_blob  = list(self.network.inputs.keys())
        self.output_blob = next(iter(self.network.outputs))
        for i in self.input_blob:
            self.input_shapes[i] = self.network.inputs[i].shape
    
    def preprocess_input(self, left_eye,right_eye):
        
        left_eye = cv2.resize(left_eye,(60,60))
        left_eye = left_eye.transpose((2,0,1))
        left_eye = left_eye.reshape(1,*left_eye.shape)

        right_eye = cv2.resize(right_eye,(60,60))
        right_eye = right_eye.transpose((2,0,1))
        right_eye = right_eye.reshape(1,*right_eye.shape)
        
        return left_eye,right_eye

    def check_model(self):
        sl = self.plugin.query_network(network=self.network, device_name=self.device)
        ul = [l for l in self.network.layers.keys() if l not in sl]
        if len(ul) != 0:
            print("Unsupported layers found: {}".format(ul))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    
    
    def predict(self, left_eye_image,right_eye_image,head_pose_angles,inference_type):
        
        left_eye,right_eye = self.preprocess_input(left_eye_image,right_eye_image)
        head_poses = np.array(head_pose_angles)
        info = {'left_eye_image':left_eye,'right_eye_image':right_eye,'head_pose_angles':head_poses}
        if(inference_type=="async"):
            self.exec_network.start_async(request_id=0,inputs=info)
            status = self.exec_network.requests[0].wait(-1)
            if status==0:
                result = self.exec_network.requests[0].outputs
        else:
            self.exec_network.infer(inputs=info)
            result = self.exec_network.requests[0].outputs
        
        gaze_vector = np.reshape(result['gaze_vector'],(3,1))
        
        mouse_coordinates = self.preprocess_output(head_poses,gaze_vector)
        return mouse_coordinates,gaze_vector


    def preprocess_output(self, head_poses,result):  
        
        roll = head_poses[2]
        cos_x = math.cos(roll*math.pi/180.0)
        sin_x = math.sin(roll*math.pi/180.0)
        mouse_x = result[0]*cos_x + result[1]*sin_x
        mouse_y = result[1]*cos_x - result[0]*sin_x
        return (mouse_x,mouse_y)
        

        
