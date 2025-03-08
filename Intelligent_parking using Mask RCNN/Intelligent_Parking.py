#import packages
import os
import numpy as np
import cv2 as openCV
import matplotlib as plt
import mrcnn.config
import mrcnn.utils

from mrcnn.model import MaskRCNN
from pathlib import Path


# set up config of Mask-RCNN 
class Config_Mask_RCNN(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    DETECTION_MIN_CONFIDENCE = 0.6
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  

# 1st level directory    
DIR_LIB = Path(".")
# Directoy of the RCNN model 
DIR_MODEL = os.path.join(DIR_LIB, "logs")


# Mask RCNN model path
coco_model = os.path.join(DIR_LIB, "mask_rcnn_coco.h5")

# uncomment the following if "mask_rcnn_coco.h5" is not downloaded
# if not os.path.exists(coco_model):
#    mrcnn.utils.download_trained_weights(coco_model)

# load the model
rcnn_model = MaskRCNN(mode = "inference", model_dir = DIR_MODEL, config = Config_Mask_RCNN())
rcnn_model.load_weights(coco_model, by_name=True)


# input video for test
VIDEO_INPUT = "Videos/Video.mp4"
VIDEO_OUTPUT = './Final_Video.avi'
video_frames = openCV.VideoCapture(VIDEO_INPUT)


# getting the bounding boxes from the frame
def get_boxes_on_vehicles(bound_boxes, class_ids):
    vehicle_boxes = []

    # iterate through all the boxes to the vehicles
    for i, box in enumerate(bound_boxes):
        if class_ids[i] in [3, 4, 8, 6]:
            vehicle_boxes.append(box)

    return np.array(vehicle_boxes)


# parking slots spotted
parked_vehicle_boxes = None
# Loop over all frames
frame_count = 0
 # Assume no spaces are free until we find one that is free
free_space = False
video_writer = None

while video_frames.isOpened():
    success, frame = video_frames.read()

    if not success:
        print("Last frame processed or video format is corrupted")
        break

    elif frame_count < 60:
      # compare two frames to cancel the detection of moving vehicle
      success, next_frame = video_frames.read()
      diff_frames = openCV.absdiff(frame, next_frame)  
      grey_motion = openCV.cvtColor(diff_frames, openCV.COLOR_BGR2GRAY)
      blur_pixel = openCV.GaussianBlur(grey_motion, (1, 1), 0)
      ret, threshold = openCV.threshold(blur_pixel, 20, 255, openCV.THRESH_BINARY)
      
      # erode the car which is moving so that it is not detected by MASKRCNN.
      dilated = openCV.dilate(threshold, np.ones((10, 10), np.uint8), iterations = 1 )
      eroded = openCV.erode(dilated, np.ones((10, 10), np.uint8), iterations = 1 )

      # processed frame by adding contours so that MaskRCNN doesn't get applied
      c, h = openCV.findContours(eroded, openCV.RETR_TREE, openCV.CHAIN_APPROX_SIMPLE)
      next_frame = openCV.drawContours(next_frame, c, -1, (0,0,0), openCV.FILLED)

      if frame_count%5 == 0:
        print("Current frame number : " + str(frame_count))
        # plt.pyplot.imshow(next_frame)
        # plt.pyplot.show()
        
        frame_count = frame_count + 1
        continue 

    # Converting the BGR color to RGB color.     
    rgb_image = frame[:, :, ::-1] 

    # Detect cars using Mask RCNN
    results = model.detect([rgb_image], verbose=0)
    res = results[0]


    if parked_vehicle_boxes is None:
        print("Marking vehicles. Frame number:  ", frame_count)
        video_frames = cv2.VideoCapture(VIDEO_INPUT)
        parked_vehicle_boxes = get_boxes_on_vehicles(res['rois'], res['class_ids'])

    elif frame_count%200 == 0 and len(parked_vehicle_boxes) != 0:
        # Get where vehicles are currently located in the frame
        vehicle_boxes = get_boxes_on_vehicles(res['rois'], res['class_ids'])

        # how much those vehicles overlap with the known parking slots
        overlaps = mrcnn.utils.compute_overlaps(parked_vehicle_boxes, vehicle_boxes)

        # For all the parking slots found around parked vehicles
        for parking_slot, overlap_areas in zip(parked_vehicle_boxes, overlaps):

            # find the max amount it was covered by any
            # vehicle detected for this slot
            max_IoU_overlap = np.max(overlap_areas)

            # corner co-ordinates of the parking slot
            y1, x1, y2, x2 = parking_slot

            # vehicle has left the slot
            if max_IoU_overlap < 0.15:
                # Parking slot is empty
                openCV.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                formatedText = "Parking Slot free : "
                openCV.putText(frame, formatedText, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
                free_space = True
            else:
                # Parking slot is still occupied 
                openCV.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Write the IoU measurement inside the box
            # font = openCV.FONT_HERSHEY_DUPLEX
            # openCV.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.7, (255, 255, 255))

    if video_writer is None:
		# Saving the Video
	    fourcc = openCV.VideoWriter_fourcc(*"MJPG")
	    video_writer = openCV.VideoWriter(VIDEO_OUTPUT, fourcc, 15, (frame.shape[1], frame.shape[0]), True)
    video_writer.write(frame)

print("Video completed")
video_writer.release()
video_frames.release()