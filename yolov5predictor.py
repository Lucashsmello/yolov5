from __future__ import annotations
from . import unpickler
import torch
from .models import yolo
from .utils.datasets import preprocessImage
from .utils.utils import non_max_suppression, scale_coords


class Yolov5Predictor:
    def __init__(self, img_shape, conf_thres, iou_thres, yolo_model):
        self.img_shape = img_shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.yolo_model = yolo_model

    @staticmethod
    def load(weights_path, img_shape, conf_thres=0.4, iou_thres=0.5, device='cuda:0') -> Yolov5Predictor:
        """
        img_shape: (height, width)
        """
        yolo_model = torch.load(weights_path, map_location=device, pickle_module=unpickler)['model'].float().eval()
        predictor = Yolov5Predictor(img_shape, conf_thres, iou_thres, yolo_model)

        return predictor

    def preProcessImage(self, img0, device='cuda:0'):
        img = preprocessImage(img0, self.img_shape)
        img = torch.from_numpy(img).to(device).float().div(255)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img0):
        with torch.no_grad():
            img = self.preProcessImage(img0)
            preds = self.yolo_model(img)[0]
            preds = non_max_suppression(preds, self.conf_thres, self.iou_thres, classes=None, agnostic=None)[0]

            if(preds is None):
                return torch.empty(0)
            preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], img0.shape).round()
        return preds
