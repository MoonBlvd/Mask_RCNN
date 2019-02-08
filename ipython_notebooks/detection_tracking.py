import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from mrcnn import visualize
from utils import *
import glob
import time
import gc

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
TRACKER_DIR = os.path.abspath("/home/yyao/Documents/car_intersection/experimenting_with_sort")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# import tracking
sys.path.append(TRACKER_DIR) 
from sort import Sort
import glob

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VIDEO_DIR = '/home/yyao/Documents/car_intersection/data/intersection/ITS/'

# indicate GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench']#, 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

num_classes = len(class_names)





# only for testing
# VIDEO_DIR = '/home/yyao/Documents/car_intersection/data/samples/201806041123003053'
VIDEO_DIR = '/home/yyao/Documents/car_intersection/data/intersection/good_data/'

all_files = glob.glob(VIDEO_DIR+'*.mp4')
ROI = [0, 0, 1920, 1200]

saver = True
display = True
    
colors = np.random.rand(32, 3)
'''for saving observations of each video'''
all_observations = {}
for video in all_files:
    
    cap = cv2.VideoCapture(video)
    
#     '''for display'''
#     if display:
#         colours = np.random.rand(32, 3)*255  # used only for display
#         plt.ion()
#         fig = plt.figure()
    
    '''init tracker'''
    use_dlibTracker = False # True to use dlib correlation tracker, False to use Kalman Filter tracker
    all_trackers =  Sort(ROI, max_age=1,min_hits=3, use_dlib=use_dlibTracker,track_masks=True)
    
    '''count time'''
    total_time = 0

#     '''write results'''
    out_path = '/home/yyao/Documents/car_intersection/tracking_output/mask_rcnn/' + video[-22:-4] + '/'
    try:
        os.stat(out_path)
        print("video hass been processed!")
        continue
    except:
        os.mkdir(out_path)
        
    out_file = out_path + 'trackings.txt'
    f_out = open(out_file, 'w')
    
#     original_W = 1280
#     original_H = 720
#     write_path = out_path
#     video_name = video[22:]
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     writer = cv2.VideoWriter(write_path + video_name, 
#                          fourcc, 10.0, (original_W,original_H))
    
    frame = 0

    '''for saving observations of each car'''
    observations = {}
    
    while(cap.isOpened()):
        ret, img = cap.read()
        
        if img is None:
            break
            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[:640, :]
                
        mrcnn_detections = model.detect([img], verbose=1)[0]
        
        interesting_objects = np.where(mrcnn_detections['class_ids']<num_classes)[0]
        
        bboxes = mrcnn_detections['rois'][interesting_objects]
        masks = mrcnn_detections['masks'][:,:,interesting_objects]
        classes = mrcnn_detections['class_ids'][interesting_objects]
        scores = mrcnn_detections['scores'][interesting_objects]
        
                
        frame += 1
        start_time = time.time()
        #update tracker
        trackers, mask_list = all_trackers.update(bboxes.astype('int'),
                                                  img=img, 
                                                  masks=masks,
                                                  classes=classes,
                                                  scores=scores) # only cars are considered so far.
        
        

        cycle_time = time.time() - start_time
        total_time += cycle_time

        print('frame: %d...took: %3fs'%(frame,cycle_time))
        
        tracked_boxes = []
        tracked_id = []
        tracked_masks = []
        tracked_classes = []
        tracked_scores = []
        for j, (d,mask) in enumerate(zip(trackers, mask_list)):
            tracked_boxes.append(d[:4])
            tracked_id.append(d[4])
            if j == 0:
                tracked_masks = mask
            else:
                tracked_masks = np.dstack([tracked_masks, mask])
#             tracked_masks.append(mask_list[j])
            tracked_classes.append(d[-1])
            tracked_scores.append(d[-2])
            
            # track_id, frame_id, age, class, score, ymin, xmin, ymax, xmax
            f_out.write('%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n' % 
                        (d[4], frame, d[5], d[7],d[6], d[0], d[1], d[2], d[3]))
        if len(tracked_id) != 0:
            tracked_boxes = np.array(tracked_boxes).astype('int')
            tracked_id = np.array(tracked_id)
            print(tracked_id)
            tracked_masks = np.reshape(tracked_masks, (tracked_masks.shape[0],tracked_masks.shape[1],j+1))
            tracked_classes = np.array(tracked_classes).astype('int')
            tracked_scores = np.array(tracked_scores)
        else:
            tracked_boxes = np.array(tracked_boxes)
            tracked_id = np.array(tracked_id)
            tracked_masks = np.array(tracked_masks)
            tracked_classes = np.array(tracked_classes)
            tracked_scores = np.array(tracked_scores)
        
        save_path = out_path + str(format(frame,'04'))+'.png'
        masked_img = visualize.display_tracklets(img,
                                            tracked_boxes,
                                            tracked_id,
                                            tracked_masks, 
                                            tracked_classes, 
                                            class_names, 
                                            tracked_scores,
                                            colors = colors,
                                            save_path = save_path)  # used only for display)
        
#         total_mask = np.zeros((720,1280),dtype=bool)
#         for i in range(tracked_masks.shape[2]):
#             total_mask = np.bitwise_or(total_mask, tracked_masks[:,:,i])
        bbox_mask = np.ones((720,1280))
        for box in tracked_boxes:
            bbox_mask[box[0]:box[2], box[1]:box[3]] = 0
        write_csv(out_path + str(format(frame,'04')) + '.csv' ,bbox_mask)
    
    print("One video is written!")
    gc.collect()
#     break
    
