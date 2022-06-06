import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from functools import partial
import torch.nn.functional as F


class YoloLoss(nn.Module):
    def __init__(self, anchors, input_shape, cuda, anchors_mask = [[4, 5], [2, 3], [0, 1]]):
        super(YoloLoss, self).__init__()

        self.anchors = anchors
        self.bbox_attrs = 5 + 10
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)
    
    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
    
    def box_giou(self, b1, b2):
    
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
    
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / (union_area + 1e-9)

        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-9)
        return giou
    
    def forward(self, l, input, targets, y_true):

        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # (batch_size, 3, in_h, in_w, 4 + 1 + 10)
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2]) 
        h = torch.sigmoid(prediction[..., 3]) 

        conf = torch.sigmoid(prediction[..., 4])

        pred_landmarks = prediction[..., 5:]

        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)

        if self.cuda:
            y_true = y_true.type_as(x)
        
        giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
        loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])

        train_land_index = torch.where(y_true[..., -1] == 0)
        pred_landmarks = pred_landmarks[train_land_index]
        
        y_true_landmarks = y_true[train_land_index]
        loss_landm = torch.mean(F.smooth_l1_loss(pred_landmarks, y_true_landmarks[..., 5: -1]))
        tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        loss_conf = torch.mean(self.BCELoss(conf, tobj))

        loss = loss_loc + 10 * loss_landm + loss_conf

        return loss, loss_loc, loss_landm, loss_conf

    
    def get_pred_boxes(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w):

        bs = len(targets)

        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)
        
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)

        return pred_boxes