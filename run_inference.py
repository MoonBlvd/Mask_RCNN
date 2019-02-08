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
import argparse
from mrcnn.utils import y1x1y2x2_to_xywh

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from mrcnn import utils
import mrcnn.model as modellib

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MaskRCNN generate boxes")
    parser.add_argument(
        "-i", "--image_dir", help="Path to folder containing folders of images",
        default=None, required=True)
    parser.add_argument(
        "-o", "--out_dir", help="Path to folder containing folders of images",
        default=None, required=True)
    parser.add_argument(
        "--save_det_images", help='''Whether to save the box-on-images or not. 
                                    This will create a lot of folders in your directory!!''',
        default=False, action='store_true')
    parser.add_argument(
        "--for_deepsort", help='''Whether to save detection in deep-sort format or not. 
        Deepsort format is (frame_id, -1, x1, y1, w, h, feature)''', 
        default=True, action='store_true')
    parser.add_argument(
        "--image_shape", help="image shape in [W,H,channels] as a list", nargs='+', type=int,
        default=[1280, 720, 3], required=True)
    parser.add_argument(
        "-g", "--gpu", help="id of gpus to use", type=str,
        default="0")
    
    return parser.parse_args()

args = parse_args()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained_models", "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# indicate GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# only for testing
IMG_DIR = args.image_dir #'/media/DATA/VAD_datasets/taiwan_sa/testing/frames/'#'/media/DATA/traffic_accident_videos/images_10hz/'
OUT_DIR = args.out_dir #'/media/DATA/VAD_datasets/taiwan_sa/testing/mask_rcnn_detections/'#'/media/DATA/traffic_accident_videos/mask_rcnn_detections/'

print(args.image_shape[0])
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = len(args.gpu)
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = args.image_shape
    IMAGE_MAX_DIM = max(args.image_shape)
#     IMAGE_RESIZE_MODE = "none"
#     NUM_CLASSES = 15

config = InferenceConfig()
config.display()

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
all_folders = glob.glob(os.path.join(IMG_DIR, '*'))

for folder_id, folder in enumerate(all_folders):
    video_name = folder.split('/')[-1]
    print(video_name)    
    
    '''for display'''
    if args.save_det_images:
        colours = np.random.rand(32, 3)*255  # used only for display
        plt.ion()
        fig = plt.figure()

        SAMPLE_IMG_DIR = os.path.join(OUT_DIR, video_name)
        if not os.path.isdir(SAMPLE_IMG_DIR):
            os.mkdir(SAMPLE_IMG_DIR)
    
    
    '''write results'''
    out_file = os.path.join(OUT_DIR, video_name + '.txt')
    out_file_with_feature = os.path.join(OUT_DIR, video_name + '.npy')

    try:
        os.stat(out_file_with_feature)
        print("video has been processed!")
        continue
    except:
        pass
    
    frame = 0
    all_images = sorted(glob.glob(os.path.join(folder, 'images','*.jpg')))
    output_with_feature = []
    for image_file in all_images:
        img = np.asarray(Image.open(image_file))
        
        if img is None:
            break
        # run detection
        start_time = time.time()
        mrcnn_detections  = model.detect([img], verbose=1)[0]
        cycle_time = time.time() - start_time
        print('frame: %d...took: %3fs'%(frame,cycle_time))
        
        # only select specific type of objects
        interesting_objects = np.where(mrcnn_detections['class_ids'] < num_classes)[0]
        
        bboxes = mrcnn_detections['rois'][interesting_objects] # ymin xmin ymax xmax
        # convert to xywh format for deepsort purpose
        if args.for_deepsort:
            deepsort_bboxes = y1x1y2x2_to_xywh(copy.deepcopy(bboxes))
        
        masks = mrcnn_detections['masks'][:,:,interesting_objects]
        classes = mrcnn_detections['class_ids'][interesting_objects]
        scores = mrcnn_detections['scores'][interesting_objects]
        features = mrcnn_detections['roi_features'][interesting_objects]
        
        frame_ids = frame * np.ones([bboxes.shape[0],1])
        track_ids = -1 * np.ones([bboxes.shape[0],1])
        if args.for_deepsort:
            complete_output_array = np.hstack([frame_ids, 
                                            track_ids, 
                                            deepsort_bboxes, 
                                            np.expand_dims(scores, axis=-1), 
                                            features])
        else:
            complete_output_array = np.hstack([frame_ids, 
                                            track_ids, 
                                            deepsort_bboxes, 
                                            np.expand_dims(scores, axis=-1)])
        
        if len(output_with_feature) == 0:
            output_with_feature = complete_output_array
        else:
            output_with_feature = np.vstack([output_with_feature, complete_output_array])
            
    
#         save masked images
        if args.save_det_images:
            save_path = os.path.join(SAMPLE_IMG_DIR, str(format(frame,'04'))+'.jpg')
            visualize.display_instances(img, bboxes, masks, classes, class_names,
                                              scores=scores, save_path=save_path,
                                              figsize=(16, 16),
                                              show_bbox=True)
        frame += 1
    np.save(out_file_with_feature, output_with_feature)
    print("One video is written!")
#     f_out.close()
