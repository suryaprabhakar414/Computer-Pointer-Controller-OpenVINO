from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
from face_detection import Face_Detection
from facial_landmarks_detection import Facial_Landmarks_Detection
from head_pose_estimation import Head_Pose_Estimation


Image = cv2.imread("/home/vegeta/Documents/image.png")

FD = Face_Detection()
FD.load_model()
image,face_coord = FD.predict(Image)

######################################################################################################################################################################

FLD = Facial_Landmarks_Detection()
FLD.load_model()
left_eye_image,right_eye_image,eye_coordinates = FLD.predict(image)

######################################################################################################################################################################

HPE = Head_Pose_Estimation()
HPE.load_model()
print(HPE.get_input_shape())
print(HPE.get_output_shape())
result = HPE.predict(image)
print(result)













