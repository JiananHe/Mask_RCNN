import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
MRNN_ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(MRNN_ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.abspath("./logs/")
IMG_DIR = "E:/circle_data/VOCdevkit/VOC2009/JPEGImages/"


class CirclesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes(circle)

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


def load_original_image_gt(image_id):
    """
    return ground true of original image, include boxes, masks, and class_ids
    :param image_id:
    :return: gt_bbox: (obj_count, 4)
             gt_mask: (h, w, obj_count)
             gt_class_id: (obj_count,)
    """
    img_path = os.path.join(IMG_DIR, "source" + str(image_id) + ".jpg")
    labels_path = os.path.join(IMG_DIR, "source" + str(image_id) + ".txt")

    original_image = cv2.imread(img_path)
    height, width = original_image.shape[:2]

    labels = np.loadtxt(labels_path)
    if labels.ndim == 1:
            labels = np.array([labels])
    obj_count = labels.shape[0]

    gt_bbox = []
    gt_class_id = []
    gt_mask = np.zeros([height, width, obj_count], dtype=np.uint8)
    for i, label in enumerate(labels):
        gt_class_id.append(int(label[0]) + 1)
        x, y, w, h = label[1:5]
        x = int(x * width)
        y = int(y * height)
        w = abs(w * width)
        h = abs(h * height)

        s = int(min(w, h) / 2)
        x1 = int(x - (w/2))
        y1 = int(y - (h/2))
        x2 = int(x + (w/2))
        y2 = int(y + (h/2))

        gt_bbox.append([y1, x1, y2, x2])
        cv2.circle(gt_mask[:, :, i:i + 1].copy(), (x, y), s, 1, -1)

    gt_bbox = np.array(gt_bbox)
    gt_class_id = np.array(gt_class_id)
    assert gt_bbox.shape == (obj_count, 4)
    assert gt_class_id.shape == (obj_count,)

    return gt_bbox, gt_mask, gt_class_id


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


inference_config = CirclesConfig()
inference_config.display()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(MODEL_DIR, "shapes20190224T1515\\mask_rcnn_shapes_0001.h5")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
# image_ids = np.arange(1, 10)
image_ids = np.array([1, 3])
# APs = []
for image_id in image_ids:
    img_path = os.path.join(IMG_DIR, "source" + str(image_id) + ".jpg")
    original_image = cv2.imread(img_path)

    gt_bbox, gt_mask, gt_class_id = load_original_image_gt(image_id)
    # show original image
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                ["BG", "circle"], figsize=(8, 8), ax=get_ax())

    results = model.detect([original_image], verbose=1)

    r = results[0]
    # show predicted image
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                ["BG", "circle"], r['scores'], figsize=(8, 8), ax=get_ax())

    # AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
    #                                                      r["rois"], r["class_ids"], r["scores"], r['masks'])
    # APs.append(AP)
plt.show()
# print("mAP: ", np.mean(APs))