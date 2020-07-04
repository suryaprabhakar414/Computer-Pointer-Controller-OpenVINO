'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class Head_Pose_Estimation:
    
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

    def load_model(self,model_path):
        model = model_path+"head-pose-estimation-adas-0001.xml"
        weights = model_path+"head-pose-estimation-adas-0001.bin"
        self.network = IENetwork(model=model,weights = weights)
        self.plugin = IECore()
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)
        self.check_model()
        self.exec_network = self.plugin.load_network(self.network,self.device)
        self.input_blob  = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape
    
    def preprocess_input(self, image):
        input_shape = self.get_input_shape()
        image = cv2.resize(image,(input_shape[3],input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1,*image.shape)
        return image

    def check_model(self):
        sl = self.plugin.query_network(network=self.network, device_name=self.device)
        ul = [l for l in self.network.layers.keys() if l not in sl]
        if len(ul) != 0:
            print("Unsupported layers found: {}".format(ul))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def predict(self, image,inference_type):
        image_HPE = self.preprocess_input(image)
        
        if(inference_type=="async"):

            self.exec_network.start_async(request_id=0,inputs={self.input_blob:image_HPE.astype(np.float32)})
            status = self.exec_network.requests[0].wait(-1)
            if status==0:
                result = self.exec_network.requests[0].outputs
        else:
            result = self.exec_network.infer(inputs={self.input_blob:image_HPE.astype(np.float32)})
        
        output = self.preprocess_output(result)
        return output


    def preprocess_output(self, result):  
        output = []
        output.append(result['angle_y_fc'][0][0]) 
        output.append(result['angle_p_fc'][0][0])
        output.append(result['angle_r_fc'][0][0])
        return output
        

        
