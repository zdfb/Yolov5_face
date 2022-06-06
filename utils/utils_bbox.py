import torch
import numpy as np
from torchvision.ops import nms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DecodeBox():
    def __init__(self, anchors, input_shape, anchors_mask = [[4, 5], [2, 3], [0, 1]]):
        super(DecodeBox, self).__init__()

        self.anchors = anchors
        self.bbox_attrs = 5 + 10
        self.input_shape = input_shape

        self.anchors_mask = anchors_mask
    
    def decode_box(self, inputs, input_shape):
        outputs = []

        for i, input in enumerate(inputs):

            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            stride_h = input_shape[0] / input_height
            stride_w = input_shape[1] / input_width

            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            prediction = input.view(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])

            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])

            conf = torch.sigmoid(prediction[..., 4])

            landmarks = prediction[..., 5:]

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_landmarks = FloatTensor(prediction[..., 5:].shape)

            pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

            pred_landmarks[..., 0] = landmarks[..., 0] + grid_x
            pred_landmarks[..., 1] = landmarks[..., 1] + grid_y
            pred_landmarks[..., 2] = landmarks[..., 2] + grid_x
            pred_landmarks[..., 3] = landmarks[..., 3] + grid_y
            pred_landmarks[..., 4] = landmarks[..., 4] + grid_x
            pred_landmarks[..., 5] = landmarks[..., 5] + grid_y
            pred_landmarks[..., 6] = landmarks[..., 6] + grid_x
            pred_landmarks[..., 7] = landmarks[..., 7] + grid_y
            pred_landmarks[..., 8] = landmarks[..., 8] + grid_x
            pred_landmarks[..., 9] = landmarks[..., 9] + grid_y

            _scale_box = torch.Tensor([input_width, input_height, 
                                       input_width, input_height]).type(FloatTensor)

            _scale_landm = torch.Tensor([input_width, input_height, 
                                         input_width, input_height,
                                         input_width, input_height, 
                                         input_width, input_height,
                                         input_width, input_height,]).type(FloatTensor)
            
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale_box,
                                conf.view(batch_size, -1, 1), pred_landmarks.view(batch_size, -1, 10) / _scale_landm), -1)
            
            outputs.append(output.data)
        return outputs
    
    def yolo_correct(self, box_xy, box_wh, landmarks, input_shape, image_shape, letterbox_image):

        landmark0 = landmarks[..., 0:2]
        landmark1 = landmarks[..., 2:4]
        landmark2 = landmarks[..., 4:6]
        landmark3 = landmarks[..., 6:8]
        landmark4 = landmarks[..., 8:10]


        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        image_shape = image_shape[::-1]
        input_shape = input_shape[::-1]

        if letterbox_image:
            
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale   = input_shape / new_shape

            box_xy  = (box_xy - offset) * scale
            landmark0 = (landmark0 - offset) * scale
            landmark1 = (landmark1 - offset) * scale
            landmark2 = (landmark2 - offset) * scale
            landmark3 = (landmark3 - offset) * scale
            landmark4 = (landmark4 - offset) * scale
            box_wh *= scale
        
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        landms = np.concatenate([landmark0, landmark1, landmark2, landmark3, landmark4], axis = -1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        landms *= np.concatenate([image_shape, image_shape, image_shape, image_shape, image_shape], axis=-1)

        return boxes, landms

    def non_max_suppression(self, prediction, input_shape, image_shape, letterbox_image, conf_thres = 0.5, nms_thres = 0.4):

        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()

            image_pred = image_pred[conf_mask]

            if not image_pred.size(0):
                continue

            detections = image_pred.to(device)

            keep = nms(detections[:, :4], detections[:, 4], nms_thres)

            max_detections = detections[keep]

            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                landmarks = (output[i][:, 5:])
                
                boxes, landms = self.yolo_correct(box_xy, box_wh, landmarks, input_shape, image_shape, letterbox_image)
                output[i][:, :4] = boxes
                output[i][:, 5:] = landms
        return output