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
import logging


def build_argparser():

    i_desc = "Specify Path to video file or enter cam for webcam"
    f_desc = "Specify Path to (.xml and .bin) of Face Detection model."
    fl_desc = "Specify Path to folder(.xml and .bin) of Facial Landmark Detection model."
    hp_desc = "Specify Path to folder(.xml and .bin) of Head Pose Estimation model."
    g_desc = "Specify Path to folder(.xml and .bin) of Gaze Estimation model."
    flags_desc = "Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space) for see the visualization of different model outputs of each frame, fd for Face Detection, fld for Facial Landmark Detection hp for Head Pose Estimation, ge for Gaze Estimation." 
    prob_desc = "Probability threshold for model to detect the face accurately from the video frame."
    d_desc = "Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)"
    l_desc = "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl."
    it_desc = "Inference type, sync for Synchronous or async for Asychronous"
    parser = argparse.ArgumentParser("Computer Pointer Controller")

    parser.add_argument("-i", required=True, type=str, help=i_desc)
    parser.add_argument("-f", required=True, type=str, help=f_desc)
    parser.add_argument("-fl", required=True, type=str,help=fl_desc)
    parser.add_argument("-hp", required=True, type=str,help=hp_desc)
    parser.add_argument("-g", required=True, type=str,help=g_desc)
    parser.add_argument("-flags", required=False, type=str,default="",help=flags_desc)
    parser.add_argument("-prob", required=False, type=float,default=0.6,help=prob_desc)
    parser.add_argument("-d", type=str, default="CPU", help=d_desc)
    parser.add_argument("-l", required=False, type=str,default=None,help=l_desc)
    parser.add_argument("-it",required = False,type = str, default = "sync",help = it_desc)
    args = parser.parse_args()
    return args

def main():

    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("Computer_Pointer_Controller.log"),
                logging.StreamHandler()
            ])
    except:
        print("File cannot be created")

    
    args = build_argparser()
    video_path = args.i
    visualize = args.flags
    count=0
    fd_inference_time = 0
    fld_inference_time = 0
    hp_inference_time = 0
    ge_inference_time = 0
    00
    MC = MouseController('medium','fast')
    
    logging.info("############## Model Load Time #############")
   
    start_time = time.time()
    first_model_time = start_time
    FD = Face_Detection(device = args.d,threshold = args.prob, extensions=args.l)
    FD.load_model(model_path = args.f)
    logging.info("Face Detection Model: {:.3f}ms".format(1000*(time.time()-first_model_time)))
    
    second_model_time = time.time()
    FLD = Facial_Landmarks_Detection(device = args.d, extensions=args.l)
    FLD.load_model(model_path = args.fl)
    logging.info("Facial Landmarks Detection Model: {:.3f}ms".format(1000*(time.time()-second_model_time)))

    third_model_time = time.time()
    HPE = Head_Pose_Estimation(device = args.d, extensions=args.l)
    HPE.load_model(model_path = args.hp)
    logging.info("Head Pose Estimation Model: {:.3f}ms".format(1000*(time.time()-third_model_time)))

    fourth_model_time = time.time()
    GE = Gaze_Estimation(device = args.d, extensions=args.l)
    GE.load_model(model_path = args.g)
    logging.info("Gaze Estimation Model: {:.3f}ms".format(1000*(time.time()-fourth_model_time)))
    logging.info("############## End ######################### ")

    Total_Model_Load_Time = 1000*(time.time()-start_time)

    ##### LOADING VIDEO FILE #####

    if(video_path=="cam"):
        IF = InputFeeder("cam")
    else:
        IF = InputFeeder("video",video_path)   
    IF.load_data()

    ##### MODEL INFERENCE #####

    start_inf_time = time.time()
    for flag,frame in IF.next_batch():
        
        if not flag:
            break
        
        if(count%5==0):
            cv2.imshow('frame',cv2.resize(frame,(500,500)))
        
        key=cv2.waitKey(60)
        count = count+1

        start_time_1 = time.time()
        face,face_coordinates = FD.predict(frame, args.it)
        fd_inference_time += (time.time()-start_time_1)

        start_time_2 = time.time()
        left_eye_image,right_eye_image,eye_coordinates = FLD.predict(face,args.it)
        fld_inference_time += (time.time()-start_time_2)

        start_time_3 = time.time()
        head_pose_angles = HPE.predict(face,args.it)
        hp_inference_time += (time.time()-start_time_3)

        start_time_4 = time.time()
        mouse_coordinates,gaze_vector = GE.predict(left_eye_image,right_eye_image,head_pose_angles,args.it)
        ge_inference_time += (time.time()-start_time_4)

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

    Total_Inference_Time = time.time()-start_inf_time
    if(count>0):
        logging.info("############## Models Inference time #######") 
        logging.info("Face Detection:{:.3f}ms".format(1000*fd_inference_time/count))
        logging.info("Facial Landmarks Detection:{:.3f}ms".format(1000*fld_inference_time/count))
        logging.info("Headpose Estimation:{:.3f}ms".format(1000*hp_inference_time/count))
        logging.info("Gaze Estimation:{:.3f}ms".format(1000*ge_inference_time/count))
        logging.info("############## End #########################") 
    
    logging.info("############## Summarized Results ##########")
    logging.info("Total Model Load_ ime: {:.3f}ms".format(Total_Model_Load_Time))
    logging.info("Total Inference Time: {:.3f}ms".format(Total_Inference_Time))
    logging.info("FPS:{}".format(int(count/Total_Inference_Time)))
    logging.info("############ End ###########################") 
    cv2.destroyAllWindows()
    IF.close()  
    

if __name__ == '__main__':
    main()
