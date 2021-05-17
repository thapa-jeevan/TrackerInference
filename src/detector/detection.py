import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

setup_logger()

VEHICLE_IDS = torch.Tensor([2, 5, 7])


def get_model():
    cfg = get_cfg()
    model_des = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_des))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_des)
    detector = DefaultPredictor(cfg)

    def detect(img):
        outputs = detector(img)
        pred_boxes = outputs['instances'].pred_boxes.tensor
        pred_scores = outputs['instances'].scores
        pred_classes = outputs['instances'].pred_classes

        mask = (pred_classes == VEHICLE_IDS[0]) | (pred_classes == VEHICLE_IDS[1]) | (pred_classes == VEHICLE_IDS[2])

        classes = pred_classes[mask]
        scores = pred_scores[mask]
        boxes = pred_boxes[mask]

        return classes, scores, boxes

    return detect
