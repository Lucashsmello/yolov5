from __future__ import annotations
from . import unpickler
import torch
from .models import yolo
from .utils.datasets import preprocessImage
from .utils.general import non_max_suppression, scale_coords
import onnx
import onnxruntime
import numpy as np

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
            if(onnxruntime.get_device() != 'GPU'):
                print("WARNING: Onnx not running in GPU! (onnxruntime.get_device()='%s')" % onnxruntime.get_device())
            onnx_model = onnx.load(weights_path)
            onnx.checker.check_model(onnx_model)
            ort_session = onnxruntime.InferenceSession(weights_path)

            print("onnx session providers:", ort_session.get_providers())
            # ort_session.set_providers(['CUDAExecutionProvider'])
            yolo_model = ort_session

            stride = 32  # FIXME
        else:
            yolo_model = torch.load(weights_path, map_location=device, pickle_module=unpickler)['model'].float().eval()
            stride = int(yolo_model.stride.max())
        predictor = Yolov5Predictor(img_shape, conf_thres, iou_thres, yolo_model, stride=stride)

        return predictor

    def preProcessImage(self, img0, device='cuda:0'):
        img = preprocessImage(img0, self.img_shape, stride=self.stride)
        img = torch.from_numpy(img).to(device).float()
        img /= 255
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
                # io_binding = self.yolo_model.io_binding()
                # X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img.cpu().numpy(), 'cuda', 0)

                # io_binding.bind_input(name='images', device_type=X_ortvalue.device_name(), device_id=0,
                #                       element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
                # io_binding.bind_output('output')
                # self.yolo_model.run_with_iobinding(io_binding)
                # ort_outs = io_binding.copy_outputs_to_cpu()[0]

                ort_inputs = {self.yolo_model.get_inputs()[0].name: img.cpu().numpy()}
                ort_outs = self.yolo_model.run(None, ort_inputs)[0]
                preds = torch.from_numpy(ort_outs)

            preds = non_max_suppression(preds, self.conf_thres, self.iou_thres, classes=None, agnostic=None)[0]

            if(preds is None):
                return torch.empty(0)
            preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], img0.shape)
        return preds
