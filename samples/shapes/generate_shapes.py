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
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


def random_shape(height, width):
    """Generates specifications of a random shape that lies within
    the given height and width boundaries.
    Returns a tuple of three valus:
    * The shape name (square, circle, ...)
    * Shape color: a tuple of 3 values, RGB.
    * Shape dimensions: A tuple of values that define the shape size
                        and location. Differs per shape type.
    """
    # Shape
    shape = random.choice(["square", "circle", "triangle"])
    # Color
    color = tuple([random.randint(0, 255) for _ in range(3)])
    # Center x, y
    buffer = 20
    y = random.randint(buffer, height - buffer - 1)
    x = random.randint(buffer, width - buffer - 1)
    # Size
    s = random.randint(buffer, height // 4)
    return shape, color, (x, y, s)


def random_image():
    """Creates random specifications of an image with multiple shapes.
    Returns the background color of the image and a list of shape
    specifications that can be used to draw the image.
    """
    # Pick random background color
    bg_color = np.array([random.randint(0, 255) for _ in range(3)])
    height = random.randint(300, 350)
    width = random.randint(350, 400)
    print("bg_color: " + str(bg_color))
    print("height: " + str(height))
    print("width: " + str(width))

    image = np.ones([height, width, 3], dtype=np.uint8) * bg_color
    mask = np.zeros([height, width, 3], dtype=np.uint8)
    labels = []
    assert image.shape == mask.shape

    # Generate a few random shapes
    shapes = []
    boxes = []
    N = random.randint(1, 4)
    for _ in range(N):
        shape, color, dims = random_shape(height, width)
        shapes.append((shape, color, dims))
        x, y, s = dims
        boxes.append([y - s, x - s, y + s, x + s])
    # Apply non-max suppression wit 0.3 threshold to avoid
    # shapes covering each other
    keep_ixs = utils.non_max_suppression(
        np.array(boxes), np.arange(N), 0.1)
    shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]

    for shape, color, dims in shapes:
        x, y, s = dims
        if shape == 'square':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
            mask = cv2.rectangle(mask, (x - s, y - s),
                                  (x + s, y + s), 1, -1)
            labels.append([1, x, y, s])
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
            mask = cv2.circle(mask, (x, y), s, 2, -1)
            labels.append([2, x, y, s])
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
            mask = cv2.fillPoly(mask, points, 3)
            labels.append([3, x, y, s])

    return image, mask, np.array(labels)


if __name__ == "__main__":
    num_simple = 100
    simple_path = os.path.join(ROOT_DIR, "samples/shapes/generated")

    for i in range(num_simple):
        temp_dir = os.path.join(simple_path, "simple" + str(i))
        if os.path.exists(temp_dir):
            ls = os.listdir(temp_dir)
            for file in ls:
                os.remove(os.path.join(temp_dir, file))
        else:
            os.makedirs(temp_dir)

        image, mask, labels = random_image()
        cv2.imwrite(os.path.join(temp_dir, "img.png"), image)
        cv2.imwrite(os.path.join(temp_dir, "label.png"), mask)
        np.savetxt(os.path.join(temp_dir, "labels.txt"), labels)