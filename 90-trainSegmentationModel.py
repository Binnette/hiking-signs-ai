import os
import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

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

cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

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
evaluator = COCOEvaluator("hiking_signs_test", cfg, False, output_dir="./-is/")
val_loader = build_detection_test_loader(cfg, "hiking_signs_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
