'''Only do detection, no tracking'''
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import copy

from mrcnn import visualize
from sort.sort import Sort
from utils import *
import glob
import time
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained_models", "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# indicate GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# only for testing
IMG_DIR = '/media/DATA/VAD_datasets/taiwan_sa/testing/frames/'#'/media/DATA/traffic_accident_videos/images_10hz/'
OUT_DIR = '/media/DATA/VAD_datasets/taiwan_sa/testing/mask_rcnn_detections/'#'/media/DATA/traffic_accident_videos/mask_rcnn_detections/'


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1280,720,3]
    IMAGE_MAX_DIM = 1280
#     IMAGE_RESIZE_MODE = "none"
#     NUM_CLASSES = 15

config = InferenceConfig()
config.display()

def y1x1y2x2_to_xywh(boxes):
    '''
    Params:
        bounding boxes: (num_boxes, 4) in [ymin,xmin,ymax,xmax] order
    Returns:
        bounding boxes: (num_boxes, 4) in [xmin,ymin,w,h] order
    '''
    boxes = y1x1y2x2_to_x1y1x2y2(boxes)    
    boxes[:,2] -=boxes[:,0]
    boxes[:,3] -=boxes[:,1] 
    return boxes

def y1x1y2x2_to_x1y1x2y2(boxes):
    '''
    Params:
        bounding boxes: (num_boxes, 4) in [ymin,xmin,ymax,xmax] order
    Returns:
        bounding boxes: (num_boxes, 4) in [xmin, ymin,xmax, ymax] order
    '''
    tmp = copy.deepcopy(boxes[:,1])
    boxes[:,1] = boxes[:,0]
    boxes[:,0] = tmp
    
    tmp = copy.deepcopy(boxes[:,3])
    boxes[:,3] = boxes[:,2]
    boxes[:,2] = tmp
    return boxes


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Part of COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench']

num_classes = len(class_names)


all_folders = glob.glob(IMG_DIR + '*')

W = 1280
H = 720
ROI = [0, 0, 720, 1280]
display = False


colors = np.random.rand(32, 3)
'''for saving observations of each video'''
all_observations = {}

for_deepsort = True
save_det_images = False

for folder_id, folder in enumerate(all_folders):
    video_name = folder.split('/')[-1]
    print(video_name)    
    '''for display'''
    if display:
        colours = np.random.rand(32, 3)*255  # used only for display
        plt.ion()
        fig = plt.figure()
    
    '''init tracker'''
#     use_dlibTracker = False # True to use dlib correlation tracker, False to use Kalman Filter tracker
#     all_trackers =  Sort(ROI, max_age=3,min_hits=3, use_dlib=use_dlibTracker,track_masks=True)
    all_trackers =  Sort(max_age=5, min_hits=3, since_update_thresh=1)
    
    '''count time'''
    total_time = 0

#     '''write results'''
    out_file = os.path.join(OUT_DIR, video_name + '.txt')
    out_file_with_feature = os.path.join(OUT_DIR, video_name + '.npy')
    
    try:
        os.stat(OUT_DIR)
        print("video has been processed!")
#         continue
    except:
        os.mkdir(OUT_DIR)
        aa = 1
#     f_out = open(out_file, 'w')
    frame = 0
    
    '''for saving observations of each car'''
    observations = {}
    
    all_images = sorted(glob.glob(os.path.join(folder, 'images','*.jpg')))
    
    '''make dir if doesn exist'''
    SAMPLE_IMG_DIR = os.path.join(OUT_DIR, video_name)
    if not os.path.isdir(SAMPLE_IMG_DIR):
        os.mkdir(SAMPLE_IMG_DIR)
        
    output_with_feature = []
    for image_file in all_images:
        img = np.asarray(Image.open(image_file))
        
        if img is None:
            break
        # run detection
        start_time = time.time()
        mrcnn_detections  = model.detect([img], verbose=1)[0]
        cycle_time = time.time() - start_time
        total_time += cycle_time
        print('frame: %d...took: %3fs'%(frame,cycle_time))
        
        interesting_objects = np.where(mrcnn_detections['class_ids'] < num_classes)[0]
        
        bboxes = mrcnn_detections['rois'][interesting_objects] # ymin xmin ymax xmax
        # convert to xywh format for deepsort purpose
        if for_deepsort:
            deepsort_bboxes = y1x1y2x2_to_xywh(copy.deepcopy(bboxes))
        
        masks = mrcnn_detections['masks'][:,:,interesting_objects]
        classes = mrcnn_detections['class_ids'][interesting_objects]
        scores = mrcnn_detections['scores'][interesting_objects]
        features = mrcnn_detections['roi_features'][interesting_objects]
        
        frame_ids = frame * np.ones([bboxes.shape[0],1])
        track_ids = -1 * np.ones([bboxes.shape[0],1])
        complete_output_array = np.hstack([frame_ids, 
                                           track_ids, 
                                           deepsort_bboxes, 
                                           np.expand_dims(scores, axis=-1), 
                                           features])
        
        if len(output_with_feature) == 0:
            output_with_feature = complete_output_array
        else:
            output_with_feature = np.vstack([output_with_feature, complete_output_array])
            
    
#         save masked images
        if save_det_images:
            save_path = os.path.join(SAMPLE_IMG_DIR, str(format(frame,'04'))+'.jpg')
            visualize.display_instances(img, bboxes, masks, classes, class_names,
                                              scores=scores, save_path=save_path,
                                              figsize=(16, 16),
                                              show_bbox=True)
        frame += 1
    np.save(out_file_with_feature, output_with_feature)
    print("One video is written!")
#     f_out.close()
