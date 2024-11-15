import os
import cv2
import torch
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np

classes = ["top", "destination", "poster", "bike_sign", "street_sign", "panel"]

# Configuration
cfg = get_cfg()
cfg.OUTPUT_DIR = "./model-od"

cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.MASK_ON = True

cfg.DATASETS.TRAIN = ("hiking_signs_train",)
cfg.DATASETS.TEST = ("hiking_signs_test",)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.STEPS = (4200, 5000)
cfg.SOLVER.MAX_ITER = 5400

cfg.DATALOADER.NUM_WORKERS = 4

predictor = DefaultPredictor(cfg)

def create_crop_folders():
    if not os.path.exists('crop/top'):
        os.makedirs('crop/top')
    if not os.path.exists('crop/destination'):
        os.makedirs('crop/destination')

def get_perspective_corners(mask):
    """
    Extract the four corners of the mask and return them.

    This function finds the contours of the mask, identifies the largest contour, and computes the minimum area 
    rectangle that encloses the contour. It then extracts the four corners of this rectangle and returns them.
    """
    # Find the contours of the mask using cv2.findContours
    # mask.astype(np.uint8) converts the mask to an 8-bit single-channel image
    # cv2.RETR_EXTERNAL retrieves only the external contours
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour based on the area using max
    contour = max(contours, key=cv2.contourArea)

    # Compute the minimum area rectangle that can enclose the largest contour
    rect = cv2.minAreaRect(contour)

    # Get the four corners of the rectangle using cv2.boxPoints
    box = cv2.boxPoints(rect)

    # Convert the coordinates of the corners to integer values using np.intp (replacing np.int0)
    box = np.intp(box)  # Replace np.int0 with np.intp
    
    return box


def correct_perspective(image, box):
    # Determine the corners
    box = np.array(box)
    sum_points = np.sum(box, axis=1)
    diff_points = np.diff(box, axis=1)

    top_left = box[np.argmin(sum_points)]
    bottom_right = box[np.argmax(sum_points)]
    top_right = box[np.argmin(diff_points)]
    bottom_left = box[np.argmax(diff_points)]

    # Define the destination points (output image size)
    width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Get the perspective transform matrix
    src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the perspective
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))

    return warped

def draw_image(image, box, mask):
    """
    Draws the bounding box and the mask on the image and displays it in a window with max size 800x600.
    :param image: The input image.
    :param box: The coordinates of the box to draw.
    :param mask: The mask to draw.
    """
    # Copy the image to avoid modifying the original
    image_copy = image.copy()

    # Draw the box as a polygon
    if box is not None:
        cv2.polylines(image_copy, [box], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw the mask on the image
    if mask is not None:
        mask_image = np.zeros_like(image_copy)
        mask_image[mask] = (0, 0, 255)  # Red color for the mask
        overlay = cv2.addWeighted(image_copy, 1.0, mask_image, 0.5, 0)
    else:
        overlay = image_copy

    # Set the window name and size
    window_name = "Image with Box and Mask"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # Show the image with the box and mask
    cv2.imshow(window_name, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_and_label_corners(image, box):
    # Copy the image to avoid modifying the original
    image_copy = image.copy()

    # Determine the corners
    box = np.array(box)
    sum_points = np.sum(box, axis=1)
    diff_points = np.diff(box, axis=1)

    top_left = box[np.argmin(sum_points)]
    bottom_right = box[np.argmax(sum_points)]
    top_right = box[np.argmin(diff_points)]
    bottom_left = box[np.argmax(diff_points)]

    # Draw the corners
    cv2.circle(image_copy, tuple(top_left), 5, (0, 255, 0), -1)
    cv2.circle(image_copy, tuple(top_right), 5, (0, 255, 0), -1)
    cv2.circle(image_copy, tuple(bottom_left), 5, (0, 255, 0), -1)
    cv2.circle(image_copy, tuple(bottom_right), 5, (0, 255, 0), -1)

    # Label the corners
    cv2.putText(image_copy, 'top_left', tuple(top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image_copy, 'top_right', tuple(top_right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image_copy, 'bottom_left', tuple(bottom_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image_copy, 'bottom_right', tuple(bottom_right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Create a window with a specific size
    cv2.namedWindow('Image with Corners', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image with Corners', 800, 600)

    # Show the image
    cv2.imshow('Image with Corners', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_and_save(image_path, predictor):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks if instances.has("pred_masks") else None
    classes = instances.pred_classes
    scores = instances.scores

    if masks is not None:
        for i, (mask, cls, score) in enumerate(zip(masks, classes, scores)):
            if score > 0.7:  # Only crop if prediction score is greater than 70%
                box = get_perspective_corners(mask.numpy())
                #draw_image(image, box, mask)
                #draw_and_label_corners(image, box)
                corrected_image = correct_perspective(image, box)
                #draw_image(corrected_image, None, None)
                if cls == 5:  # Top part
                    crop_path = f"crop/top/{os.path.splitext(os.path.basename(image_path))[0]}_top_{i+1}.jpg"
                elif cls == 1:  # Destination board
                    crop_path = f"crop/destination/{os.path.splitext(os.path.basename(image_path))[0]}_destination_{i+1}.jpg"
                else:
                    continue  # Skip other classes
                cv2.imwrite(crop_path, corrected_image)

def process_new_images(folder_path, predictor):
    create_crop_folders()
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    for filename in tqdm(filenames, desc="Processing images"):
        image_path = os.path.join(folder_path, filename)
        crop_and_save(image_path, predictor)

process_new_images("./photos/HikingSigns", predictor)
