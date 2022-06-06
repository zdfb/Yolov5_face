import os
import time
import torch
import numpy as np
import torch.nn as nn

from nets.head.yolo import YoloFace
from utils.utils import cvtColor, get_anchors, preprocess_input, resize_image

from utils.utils_bbox import DecodeBox

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Yolo(object):
    def __init__(self):
        super(Yolo, self).__init__()

        model_path = 'model_data/yolov5_face.pth'
        anchors_path = 'Data/yolo_anchors.txt'

        self.anchors_mask = [[4, 5], [2, 3], [0, 1]]
        self.letterbox_image = False

        self.anchors, _ = get_anchors(anchors_path)

        self.backbone = 'cspdarknet'
        self.phi = 's'
        self.neck = 'PAN'
        self.ssh = True
        self.confidence = 0.02
        self.nms_iou = 0.4

        self.bbox_util = DecodeBox(self.anchors, self.anchors_mask, self.anchors_mask)

        model = YoloFace(anchors_mask = self.anchors_mask, phi = self.phi, backbone = self.backbone, neck = self.neck, ssh = self.ssh, pretrained = False)
        model.load_state_dict(torch.load(model_path, map_location = device))
        model = model.eval()

        self.model = model.to(device)
    
    def detect_image(self, image, input_shape = [1600, 1600]):
        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)

        image_data = resize_image(image, (input_shape[1], input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(device)


            outputs = self.model(images)

            outputs = self.bbox_util.decode_box(outputs, input_shape)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), input_shape, image_shape, self.letterbox_image, 
                                                        conf_thres = self.confidence, nms_thres = self.nms_iou)
            
            if results[0] is None:
                return None
            else:
                return results[0]
