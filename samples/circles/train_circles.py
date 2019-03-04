"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

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

# Root directory of Mask R-CNN
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
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MRNN_ROOT_DIR, "mask_rcnn_coco.h5")


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
    IMAGES_PER_GPU = 2

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


class CirclesDataset(utils.Dataset):
    def load_circles(self, dataset_root, dataset_list):
        """load circle dataset from files (source1to source72)
        """
        # Add classes
        self.add_class("shapes", 1, "circle")

        # Add images
        for i in range(len(dataset_list)):
            image_path = os.path.join(dataset_root, dataset_list[i] + ".jpg")
            label_path = os.path.join(dataset_root, dataset_list[i] + ".txt")
            image = cv2.imread(image_path)

            self.add_image("shapes", image_id=i, path=image_path,
                           width=image.shape[1], height=image.shape[0], label_path=label_path)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for circles of the given image ID.
        """
        info = self.image_info[image_id]
        label_path = info['label_path']
        width = info['width']
        height = info['height']

        labels = np.loadtxt(label_path)  # labels[i]: id x y w h
        if labels.ndim == 1:
            labels = np.array([labels])

        obj_count = labels.shape[0]

        mask = np.zeros([height, width, obj_count], dtype=np.uint8)
        class_ids = []
        for i in range(obj_count):
            label_id = int(labels[i][0] + 1)
            class_name = self.class_info[label_id]["name"]
            assert class_name == "circle"  # only one class
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                (labels[i][1], labels[i][2], labels[i][3], labels[i][4]), 1)
            class_ids.append(label_id)

        # Map class names to class IDs.
        class_ids = np.array(class_ids)
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, dims, color):
        """Draws a circle from the given specs."""
        # Get the center x, y and w, h, converted from YOLO format
        x, y, w, h = dims
        height = image.shape[0]
        width = image.shape[1]
        x = int(x * width)
        y = int(y * height)
        w = abs(w * width)
        h = abs(h * height)
        s = int(min(w, h) / 2)
        image = cv2.circle(image, (x, y), s, color, -1)
        return image


if __name__ == "__main__":
    config = CirclesConfig()
    config.display()

    dataset_root = "E:\\circle_data\\VOCdevkit\\VOC2009\\JPEGImages"
    dataset_list = os.listdir(dataset_root)
    dataset_list = [s.split(".")[0] for s in dataset_list if s[0:6] == "source"]
    dataset_list = list(set(dataset_list))

    # Generating dataset
    dataset_train = CirclesDataset()
    train_list = np.random.choice(dataset_list, size=60)
    dataset_train.load_circles(dataset_root, train_list)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CirclesDataset()
    val_list = [i for i in dataset_list if i not in train_list]
    dataset_val.load_circles(dataset_root, val_list)
    dataset_val.prepare()

    # Load and display random samples
    # image_ids = np.random.choice(dataset_train.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    # model.keras_model.save_weights(model_path)
