from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
from face_detection import Face_Detection
from facial_landmarks_detection import Facial_Landmarks_Detection
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation
from mouse_controller import MouseController
from input_feeder import InputFeeder
import argparse
import math
import time

def build_argparser():
    parser = argparse.ArgumentParser("Computer Pointer Controller")
    

    parser.add_argument("-i", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    #python3 main.py -i  "/home/vegeta/Documents/Python/Project/bin/demo.mp4"  -f "/opt/intel/openvino_2020.2.120/deployment_tools/tools/model_downloader/intel/face-detection-adas-binary-0001/FP32-INT1/" -fl "/opt/intel/openvino_2020.2.120/deployment_tools/tools/model_downloader/intel/landmarks-regression-retail-0009/FP32/" -hp "/opt/intel/openvino_2020.2.120/deployment_tools/tools/model_downloader/intel/head-pose-estimation-adas-0001/FP32/" -g "/opt/intel/openvino_2020.2.120/deployment_tools/tools/model_downloader/intel/gaze-estimation-adas-0002/FP32/"
    parser.add_argument("-f", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-flags", required=False, type=str,
                        default="",
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )

    parser.add_argument("-prob", required=False, type=float,default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")

    parser.add_argument("-d", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")        
    
    args = parser.parse_args()
    return args

def main():
    args = build_argparser()
    video_path = args.i
    visualize = args.flags
    count=0
    MC = MouseController('medium','fast')
    
    ##### LOADING MODELS #####
    start_time = time.time()
    FD = Face_Detection(device = args.d,threshold = args.prob)
    FD.load_model(model_path = args.f)

    FLD = Facial_Landmarks_Detection(device = args.d)
    FLD.load_model(model_path = args.fl)
    
    HPE = Head_Pose_Estimation(device = args.d)
    HPE.load_model(model_path = args.hp)

    GE = Gaze_Estimation(device = args.d)
    GE.load_model(model_path = args.g)

    total_load_time = (time.time()-start_time)*1000

    print("MODEL LOADED SUCCESSFULLY")

    ##### LOADING VIDEO FILE #####
    if(video_path=="cam"):
        IF = InputFeeder("cam")
    else:
        IF = InputFeeder("video",video_path)   
    IF.load_data()
    print("VIDEO LOADED SUCCESSFULLY")

    ##### MODEL INFERENCE #####
    start_inf_time = time.time()
    for flag,frame in IF.next_batch():
        
        if not flag:
            break
        
        if(count%5==0):
            cv2.imshow('frame',cv2.resize(frame,(500,500)))
    
        key=cv2.waitKey(60)
        count = count+1
        face,face_coordinates = FD.predict(frame)
        left_eye_image,right_eye_image,eye_coordinates = FLD.predict(face)
        head_pose_angles = HPE.predict(face)
        mouse_coordinates,gaze_vector = GE.predict(left_eye_image,right_eye_image,head_pose_angles)
        
        
        if(len(visualize)!=0):
            frame_visualize = frame.copy()
            
            if ("fd" in visualize):
                if(len(visualize)==1):
                     cv2.rectangle(frame_visualize, (face_coordinates[0], face_coordinates[1]),
                                  (face_coordinates[2], face_coordinates[3]), (255, 0, 255), 2)
                else:
                    frame_visualize = face.copy()

            if ("fld" in visualize):
                if not "fd" in visualize:
                    frame_visualize = face.copy()
            
                cv2.circle(frame_visualize,(eye_coordinates['left_eye'][0],eye_coordinates['left_eye'][1]),25,(0,0,255),2)
                cv2.circle(frame_visualize,(eye_coordinates['right_eye'][0],eye_coordinates['right_eye'][1]),25,(0,0,255),2)
                   
            if ("hp" in visualize):
                cv2.putText(frame_visualize, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(head_pose_angles[0],
                            head_pose_angles[1],head_pose_angles[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.255, (0, 255, 0), 1)

            if ("ge" in visualize):
                h = face.shape[0]
                arrow = h*0.7
                arrow_X = gaze_vector[0]*arrow
                arrow_Y = -gaze_vector[1]*arrow
                cv2.arrowedLine(frame_visualize,(eye_coordinates['left_eye'][0],eye_coordinates['left_eye'][1]),
                             (int(eye_coordinates['left_eye'][0] + arrow_X), int(eye_coordinates['left_eye'][1] + arrow_Y)),(255,0,0),2)
                cv2.arrowedLine(frame_visualize,(eye_coordinates['right_eye'][0],eye_coordinates['right_eye'][1]),
                             (int(eye_coordinates['right_eye'][0] + arrow_X), int(eye_coordinates['right_eye'][1] + arrow_Y)),(255,0,0),2)
                
            
            cv2.imshow('Visualization',cv2.resize(frame_visualize,(500,500)))

        if(count%5==0):
            MC.move(mouse_coordinates[0],mouse_coordinates[1])
        
        if key==27:
            break
    print(count)
    inference_time = (time.time()-start_inf_time) 
    fps = round(count/inference_time,1)
    print(total_load_time, inference_time, fps)

if __name__ == '__main__':
    
    main()




    







    
    















