import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

import glob
'''Run on images!'''
# del sys.modules['sort']
# del sys.modules['mrcnn']
from mrcnn import visualize
from utils import *
import glob
import time

# only for testing
SBB_IMG_DIR = '/home/yyao/Documents/sbb_nlp_simulation/data/TORCS/test_2/img_SBB_compressed/'
RAW_IMG_DIR = '/home/yyao/Documents/sbb_nlp_simulation/data/TORCS/test_2/img/'
SBB_OUT_DIR = '/home/yyao/Documents/sbb_nlp_simulation/data/TORCS/test_2/detection_SBB/'
RAW_OUT_DIR = '/home/yyao/Documents/sbb_nlp_simulation/data/TORCS/test_2/detection_RAW/'

# SBB_IMG_DIR = '/home/brianyao/Documents/SBB_NLP/data/TORCS/test_2/img_SBB_compressed/'
# RAW_IMG_DIR = '/home/brianyao/Documents/SBB_NLP/data/TORCS/test_2/img/'
# SBB_OUT_DIR = '/home/brianyao/Documents/SBB_NLP/data/TORCS/test_2/detection_SBB/'
# RAW_OUT_DIR = '/home/brianyao/Documents/SBB_NLP/data/TORCS/test_2/detection_RAW/'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "SBB_NLP_example")

# # indicate GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
               'fire hydrant', 'stop sign', 'parking meter', 'bench']

num_classes = len(class_names)

saver = False
display = False
write_file = False   
'''for saving observations of each video'''
all_observations = {}

'''for display'''
if display:
    colours = np.random.rand(32, 3)*255  # used only for display
    plt.ion()
    fig = plt.figure()

'''write results'''
frame = 0

'''for saving observations of each car'''
observations = {}
all_sbb_images = sorted(glob.glob(SBB_IMG_DIR + '*'))
all_raw_images = sorted(glob.glob(RAW_IMG_DIR + '*'))
all_raw_images = all_raw_images[:-30]

for i, sbb_image_file in enumerate(all_sbb_images):
    if i < 5342:
        continue
    
    raw_image_file = all_raw_images[i]
    sbb_img = cv2.imread(sbb_image_file)
    raw_img = cv2.imread(raw_image_file)
    
    raw_detection_output = RAW_OUT_DIR + raw_image_file[-18:-4] + '.txt'
    sbb_detection_output = SBB_OUT_DIR + raw_image_file[-18:-4] + '.txt'
    
#     try:
#         os.stat(raw_detection_output)
#         print(raw_image_file[-18:-4] + ' has already been processed!')
#         continue
#     except:
#         aa = 1 
    
    if sbb_img is None or raw_img is None:
        break

    sbb_img = cv2.cvtColor(sbb_img, cv2.COLOR_RGB2BGR)
    sbb_img = sbb_img[200:360,:,:]
    
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    raw_img = raw_img[200:360,:,:]
    
    W = raw_img.shape[1]
    H = raw_img.shape[0]
    
    # detection on raw image and save
    mrcnn_detections = model.detect([raw_img], verbose=1)[0]
    interesting_objects = np.where(np.logical_and(np.logical_or(mrcnn_detections['class_ids']==3,
                                                                mrcnn_detections['class_ids']==8,),
                                                  mrcnn_detections['class_ids']<num_classes))[0]

    bboxes = mrcnn_detections['rois'][interesting_objects]
    masks = mrcnn_detections['masks'][:,:,interesting_objects]
    class_ids = mrcnn_detections['class_ids'][interesting_objects]
    scores = mrcnn_detections['scores'][interesting_objects]
    
    if write_file:
        f_out = open(RAW_OUT_DIR + raw_image_file[-18:-4] + '.txt', 'w')
        for j, bbox in enumerate(bboxes):
            w = bbox[3]-bbox[1]
            h = bbox[2]-bbox[0]
            if w > 0.7 * W or w < 0.05 * W:
                continue
            f_out.write('%d,%.3f,%d,%d,%d,%d\n' % (class_ids[j], scores[j],bbox[0],bbox[1],bbox[2],bbox[3]))

        f_out.close()
    
    if display:
        masked_img = visualize.display_instances(raw_img, bboxes, masks, class_ids, class_names,
                                                 save_path=None,
                                                  scores=scores, title="",
                                                  figsize=(16, 16), ax=None,
                                                  show_mask=False, show_bbox=True,
                                                  colors=None, captions=None)

    # detection on SBB compreseed image and save
    mrcnn_detections = model.detect([sbb_img], verbose=1)[0]
    interesting_objects = np.where(np.logical_and(np.logical_or(mrcnn_detections['class_ids']==3,
                                                                mrcnn_detections['class_ids']==8,),
                                                  mrcnn_detections['class_ids']<num_classes))[0]

    bboxes = mrcnn_detections['rois'][interesting_objects]
    masks = mrcnn_detections['masks'][:,:,interesting_objects]
    class_ids = mrcnn_detections['class_ids'][interesting_objects]
    scores = mrcnn_detections['scores'][interesting_objects]
    
    if write_file:
        f_out = open(SBB_OUT_DIR + sbb_image_file[-24:-12] + '.txt', 'w')
        for j, bbox in enumerate(bboxes):
            w = bbox[3]-bbox[1]
            h = bbox[2]-bbox[0]
            if w > 0.8 * W or w < 0.05 * w:
                continue
            f_out.write('%d,%.3f,%d,%d,%d,%d\n' % (class_ids[j], scores[j],bbox[0],bbox[1],bbox[2],bbox[3]))

        f_out.close()
    if display:
        masked_img = visualize.display_instances(sbb_img, bboxes, masks, class_ids, class_names,
                                                 save_path=None,
                                                  scores=scores, title="",
                                                  figsize=(16, 16), ax=None,
                                                  show_mask=False, show_bbox=True,
                                                  colors=None, captions=None)
    
    frame += 1

    
#     plt.clf()
    break     