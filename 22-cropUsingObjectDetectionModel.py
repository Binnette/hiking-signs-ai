import os
import cv2
import torch
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

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

def crop_and_save(image_path, predictor):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    classes = instances.pred_classes
    scores = instances.scores

    if boxes is not None:
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if score > 0.7:  # Only crop if prediction score is greater than 70%
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                if cls == 5:  # Top part
                    crop_path = f"crop/top/{os.path.splitext(os.path.basename(image_path))[0]}_top_{i+1}.jpg"
                elif cls == 1:  # Destination board
                    crop_path = f"crop/destination/{os.path.splitext(os.path.basename(image_path))[0]}_destination_{i+1}.jpg"
                else:
                    continue  # Skip other classes
                cv2.imwrite(crop_path, crop)

def process_new_images(folder_path, predictor):
    create_crop_folders()
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    for filename in tqdm(filenames, desc="Processing images"):
        image_path = os.path.join(folder_path, filename)
        crop_and_save(image_path, predictor)

process_new_images("./photos/HikingSigns", predictor)
