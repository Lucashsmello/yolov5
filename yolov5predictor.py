from __future__ import annotations
from . import unpickler
import torch
from .models import yolo
from .utils.datasets import preprocessImage
from .utils.general import non_max_suppression, scale_coords
import onnx
import onnxruntime

###Very important!:####
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
########################

class Yolov5Predictor:
    def __init__(self, img_shape, conf_thres, iou_thres, yolo_model, stride=32):
        self.img_shape = img_shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.yolo_model = yolo_model
        self.stride = stride

    @staticmethod
    def load(weights_path, img_shape, conf_thres=0.4, iou_thres=0.5, device='cuda:0') -> Yolov5Predictor:
        """
        img_shape: (height, width)
        """
        if(weights_path[-5:] == '.onnx'):
            onnx_model = onnx.load(weights_path)
            onnx.checker.check_model(onnx_model)
            ort_session = onnxruntime.InferenceSession(weights_path)
            yolo_model = ort_session
            stride = 32  # FIXME
        else:
            yolo_model = torch.load(weights_path, map_location=device, pickle_module=unpickler)['model'].float().eval()
            stride = int(yolo_model.stride.max())
        predictor = Yolov5Predictor(img_shape, conf_thres, iou_thres, yolo_model, stride=stride)

        return predictor

    def preProcessImage(self, img0, device='cuda:0'):
        img = preprocessImage(img0, self.img_shape, stride=self.stride)
        img = torch.from_numpy(img).to(device).float().div(255)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img0):
        # import sys
        with torch.no_grad():
            img = self.preProcessImage(img0)
            if(isinstance(self.yolo_model, torch.nn.Module)):
                preds = self.yolo_model(img)[0].cpu()
            else:
                ort_inputs = {self.yolo_model.get_inputs()[0].name: img.cpu().numpy()}
                ort_outs = self.yolo_model.run(None, ort_inputs)[0]
                preds = torch.from_numpy(ort_outs)

            # print(preds.shape)
            # sys.exit(1)
            preds = non_max_suppression(preds, self.conf_thres, self.iou_thres, classes=None, agnostic=None)[0]

            if(preds is None):
                return torch.empty(0)
            preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], img0.shape).round()
        return preds
