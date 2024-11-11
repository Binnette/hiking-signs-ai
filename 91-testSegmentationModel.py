import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

classes = ["top", "destination", "poster", "bike_sign", "street_sign", "panel"]

# Définir les chemins
train_set = "./coco-train-hiking-sign"
train_annotations = os.path.join(train_set, "result.json")

test_set = "./coco-test-hiking-sign"
test_annotations = os.path.join(test_set, "result.json")

# Register the dataset
def register_hiking_signs_dataset():
    register_coco_instances("hiking_signs_train", {}, train_annotations, train_set)
    register_coco_instances("hiking_signs_test", {}, test_annotations, test_set)

register_hiking_signs_dataset()

# Configuration
cfg = get_cfg()
cfg.OUTPUT_DIR = "./model-is"

cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.MASK_ON = True

cfg.DATASETS.TRAIN = ("hiking_signs_train",)
cfg.DATASETS.TEST = ("hiking_signs_test",)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.STEPS = (420, 500)
cfg.SOLVER.MAX_ITER = 540

cfg.DATALOADER.NUM_WORKERS = 4


# Create the predictor
predictor = DefaultPredictor(cfg)

# Create output folder for visualizations
output_vis_folder = "./viz-is"
os.makedirs(output_vis_folder, exist_ok=True)

# Visualization function
def visualize_predictions(image_path, predictor, output_folder):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    instances = instances[instances.scores > 0.5]  # Filter instances by score

    v = Visualizer(image[:, :, ::-1],
                   metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2,
                   instance_mode=ColorMode.IMAGE_BW)

    out = v.draw_instance_predictions(instances)
    vis_image = out.get_image()[:, :, ::-1]
    vis_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(vis_image_path, vis_image)

# Evaluation with visualization
def evaluate_with_visualization(folder_path, predictor, output_folder):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            visualize_predictions(image_path, predictor, output_folder)

# Run evaluation and visualization
evaluate_with_visualization("coco-train-hiking-sign/images", predictor, output_vis_folder)

# COCO evaluation
#evaluator = COCOEvaluator("hiking_signs_train", cfg, False, output_dir="./output-viz/")
#val_loader = build_detection_test_loader(cfg, "hiking_signs_train")
#inference_on_dataset(predictor.model, val_loader, evaluator)

print("Model evaluation and visualization completed.")
