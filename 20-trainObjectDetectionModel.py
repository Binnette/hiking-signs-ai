# python -m venv venv
# source ./venv/bin/activate
# sudo apt install python3 python3-dev git
# pip install wheel torch torchvision python-opencv numpy
# git clone https://github.com/facebookresearch/detectron2.git
# pip install -e detectron2
import os
import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

classes = ["top", "destination", "poster", "bike_sign", "street_sign", "panel"]

# Define paths
train_set = "./coco-train"
train_annotations = os.path.join(train_set, "result.json")

test_set = "./coco-test"
test_annotations = os.path.join(test_set, "result.json")

# Register the dataset
def register_dataset():
    register_coco_instances("coco_train", {}, train_annotations, train_set)
    register_coco_instances("coco_test", {}, test_annotations, test_set)

    # Set metadata
    MetadataCatalog.get("coco_train").set(
        thing_classes=classes,
        evaluator_type='coco',
    )
    MetadataCatalog.get("coco_test").set(
        thing_classes=classes,
        evaluator_type='coco',
    )

register_dataset()

# Configuration
cfg = get_cfg()
cfg.OUTPUT_DIR = "./model-od"

cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.MASK_ON = True

cfg.DATASETS.TRAIN = ("coco_train",)
cfg.DATASETS.TEST = ("coco_test",)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.STEPS = (4200, 5000)
cfg.SOLVER.MAX_ITER = 5400

cfg.DATALOADER.NUM_WORKERS = 4

# Create output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Define the Trainer with the evaluator implemented
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the model
evaluator = COCOEvaluator("coco_test", cfg, False, output_dir="./output-od/")
val_loader = build_detection_test_loader(cfg, "coco_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
