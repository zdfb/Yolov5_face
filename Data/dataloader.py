import cv2
import torch
import numpy as np
from PIL import Image
from random import random, sample, shuffle
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, txt_path, input_shape, anchors, anchors_mask,  epoch_now = None, epoch_length = None, train = False, mosaic = True, mixup = False, mosaic_prob = 0.5, mixup_prob = 0.5, special_aug_ratio = 0.7):
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.epoch_length = epoch_length
        self.train = train

        self.mosaic = mosaic
        self.mixup = mixup

        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

        self.special_aug_ratio = special_aug_ratio

        self.imgs_path = []
        self.words = []

        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []

        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        self.words.append(labels)

        self.epoch_now = epoch_now
        self.length = len(self.imgs_path)

        self.bbox_attrs = 5 + 10
        self.threshold = 4

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = index % self.length

        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            
            annotations = zip(self.imgs_path, self.words)
            random_annotations = sample(list(annotations), 3)
            random_annotations.append((self.imgs_path[index], self.words[index]))
            shuffle(random_annotations)

            image, box, landm = self.get_random_data_with_Mosaic(random_annotations, self.input_shape)

            if self.mixup and self.rand() < self.mixup_prob:
                annotations = zip(self.imgs_path, self.words)
                random_annotation = sample(list(annotations), 1)
                image_path_, labels_ = random_annotation[0]

                image_2, box_2, landm_2 = self.get_random_data(image_path_, labels_, self.input_shape, random = self.train)
                image, box, landm = self.get_random_data_with_MixUp(image, box, landm, image_2, box_2, landm_2)

        else:
            image, box, landm = self.get_random_data(self.imgs_path[index], self.words[index], self.input_shape, random = self.train)

        image = np.transpose(preprocess_input(np.array(image, dtype = np.float32)), (2, 0, 1))
        box = np.array(box, dtype = np.float32)
        landm = np.array(landm, dtype = np.float32)

        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            landm[:, [0, 2, 4, 6, 8]] = landm[:, [0, 2, 4, 6, 8]] / self.input_shape[1]
            landm[:, [1, 3, 5, 7, 9]] = landm[:, [1, 3, 5, 7, 9]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        
        y_true = self.get_target(box, landm)
        
        return image, box, landm, y_true

    
    def rand(self, a = 0, b = 1):
        return np.random.rand() * (b - a) + a
    
    def get_random_data(self, img_path, labels, input_shape, jitter = 0.3, hue = 0.1, sat = 0.7, val = 0.4, random = True):

        image = Image.open(img_path)
        image = cvtColor(image)

        iw, ih = image.size
        h, w = input_shape
        
        bboxs = np.zeros((0, 4))
        landmarks = np.zeros((0, 11))
        
        for label in labels:
            bbox = np.zeros((1, 4))
            landmark = np.zeros((1, 11))

            # bbox
            bbox[0, 0] = label[0]  # x1
            bbox[0, 1] = label[1]  # y1
            bbox[0, 2] = label[0] + label[2]  # x2
            bbox[0, 3] = label[1] + label[3]  # y2

            # landmarks
            landmark[0, 0] = label[4]
            landmark[0, 1] = label[5]
            landmark[0, 2] = label[7]
            landmark[0, 3] = label[8]
            landmark[0, 4] = label[10]
            landmark[0, 5] = label[11]
            landmark[0, 6] = label[13]
            landmark[0, 7] = label[14]
            landmark[0, 8] = label[16]
            landmark[0, 9] = label[17]

            if label[4] == -1:
                landmark[0, 10] = 1
            else:
                landmark[0, 10] = 0
            bboxs = np.append(bboxs, bbox, axis = 0)
            landmarks = np.append(landmarks, landmark, axis = 0)
        box = np.array(bboxs)
        landm = np.array(landmarks)

        if not random:
            scale = min(w / iw, h/ ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                
                no_trans_index = np.where(landm == -1)

                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                

                landm[:, [0, 2, 4, 6, 8]] = landm[:, [0, 2, 4, 6, 8]] * nw / iw + dx
                landm[:, [1, 3, 5, 7, 9]] = landm[:, [1, 3, 5, 7, 9]] * nh / ih + dy

                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h

                landm[:, [0, 2, 4, 6, 8]][landm[:, [0, 2, 4, 6, 8]] < 0] = 0
                landm[:, [1, 3, 5, 7, 9]][landm[:, [1, 3, 5, 7, 9]] < 0] = 0

                landm[:, [0, 2, 4, 6, 8]][landm[:, [0, 2, 4, 6, 8]] > w] = w
                landm[:, [1, 3, 5, 7, 9]][landm[:, [1, 3, 5, 7, 9]] > h] = h

                landm[no_trans_index] = -1

                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                landm = landm[np.logical_and(box_w > 1, box_h > 1)]

            
            return image_data, box, landm

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
    
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:

            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            landm[:, [0, 2, 4, 6, 8]] = landm[:, [0, 2, 4, 6, 8]] * nw / iw + dx
            landm[:, [1, 3, 5, 7, 9]] = landm[:, [1, 3, 5, 7, 9]] * nh / ih + dy

            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
                landm[:, [0, 2, 4, 6, 8]] = w - landm[:, [8, 6, 4, 2, 0]]

            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h

            landm[:, [0, 2, 4, 6, 8]][landm[:, [0, 2, 4, 6, 8]] < 0] = 0
            landm[:, [1, 3, 5, 7, 9]][landm[:, [1, 3, 5, 7, 9]] < 0] = 0

            landm[:, [0, 2, 4, 6, 8]][landm[:, [0, 2, 4, 6, 8]] > w] = w
            landm[:, [1, 3, 5, 7, 9]][landm[:, [1, 3, 5, 7, 9]] > h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            landm = landm[np.logical_and(box_w > 1, box_h > 1)]
        
        return image_data, box, landm
    
    def get_random_data_with_Mosaic(self, random_annotations, input_shape, jitter = 0.3, hue = .1, sat = 0.7, val = 0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        landm_datas = []
        index = 0 

        for annotations in random_annotations:
            image_path, labels = annotations

            image = Image.open(image_path)
            image = cvtColor(image)

            iw, ih = image.size

            bboxs = np.zeros((0, 4))
            landmarks = np.zeros((0, 11))
        
            for label in labels:
                bbox = np.zeros((1, 4))
                landmark = np.zeros((1, 11))

                # bbox
                bbox[0, 0] = label[0]  # x1
                bbox[0, 1] = label[1]  # y1
                bbox[0, 2] = label[0] + label[2]  # x2
                bbox[0, 3] = label[1] + label[3]  # y2

                # landmarks
                landmark[0, 0] = label[4]
                landmark[0, 1] = label[5]
                landmark[0, 2] = label[7]
                landmark[0, 3] = label[8]
                landmark[0, 4] = label[10]
                landmark[0, 5] = label[11]
                landmark[0, 6] = label[13]
                landmark[0, 7] = label[14]
                landmark[0, 8] = label[16]
                landmark[0, 9] = label[17]

                if label[4] == -1:
                    landmark[0, 10] = 1
                else:
                    landmark[0, 10] = 0
                bboxs = np.append(bboxs, bbox, axis = 0)
                landmarks = np.append(landmarks, landmark, axis = 0)
            box = np.array(bboxs)
            landm = np.array(landmarks)

            flip = self.rand() < .5
            
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]
                landm[:, [0, 2, 4, 6, 8]] = iw - landm[:, [8, 6, 4, 2, 0]]
            
            new_ar = iw/ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh
            
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            landm_data = []

            if len(box) > 0:

                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

                landm[:, [0, 2, 4, 6, 8]] = landm[:, [0, 2, 4, 6, 8]] * nw / iw + dx
                landm[:, [1, 3, 5, 7, 9]] = landm[:, [1, 3, 5, 7, 9]] * nh / ih + dy

                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h

                landm[:, [0, 2, 4, 6, 8]][landm[:, [0, 2, 4, 6, 8]] < 0] = 0
                landm[:, [1, 3, 5, 7, 9]][landm[:, [1, 3, 5, 7, 9]] < 0] = 0

                landm[:, [0, 2, 4, 6, 8]][landm[:, [0, 2, 4, 6, 8]] > w] = w
                landm[:, [1, 3, 5, 7, 9]][landm[:, [1, 3, 5, 7, 9]] > h] = h

                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                landm = landm[np.logical_and(box_w > 1, box_h > 1)]

                

                box_data = np.zeros((len(box), 4))
                landm_data = np.zeros((len(box), 11))

                box_data[:len(box)] = box
                landm_data[:len(box)] = landm
            
            image_datas.append(image_data)
            box_datas.append(box_data)
            landm_datas.append(landm_data)
        
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x  = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        new_boxes = self.merge_bbox(box_datas, cutx, cuty)
        new_landms = self.merge_landms(landm_datas)

        return new_image, new_boxes, new_landms
    
    def merge_landms(self, landms):

        landms = landms
        merge_landms = []
        for i in range(len(landms)):
           merge_landms.append(landms[i])
        
        merge_landms = np.vstack(merge_landms)

        return merge_landms
    
    def merge_bbox(self, bboxes, cutx, cuty):
        merge_bbox = []

        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):

                box = bboxes[i][j]
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                merge_bbox.append(tmp_box)

        return merge_bbox
    
    def get_random_data_with_MixUp(self, image_1, box_1, landm_1, image_2, box_2, landm_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5

        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)

        if len(landm_1) == 0:
            new_landms = landm_2
        elif len(landm_2) == 0:
            new_landms = landm_1
        else:
            new_landms = np.concatenate([landm_1, landm_2], axis = 0)
        
        return new_image, new_boxes, new_landms

    def get_near_points(self, x, y, i, j):

        sub_x = x - i
        sub_y = y - j

        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]

        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]

        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]

        else:
            return [[0, 0], [1, 0], [0, -1]]


    def get_target(self, targets, landms):

        num_layers = len(self.anchors_mask)

        input_shape = np.array(self.input_shape, dtype = 'int32')
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8, 3: 4}[layer] for layer in range(num_layers)]
        y_true = [np.zeros((len(self.anchors_mask[layer]), grid_shapes[layer][0], grid_shapes[layer][1], self.bbox_attrs + 1), dtype = 'float32') for layer in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[layer]), grid_shapes[layer][0], grid_shapes[layer][1]), dtype='float32') for layer in range(num_layers)]

        if len(targets) == 0:
            return y_true
        
        for layer in range(num_layers):
            in_h, in_w = grid_shapes[layer]
            anchors = np.array(self.anchors) / {0: 32, 1: 16, 2: 8, 3: 4}[layer]

            batch_target = np.zeros_like(targets)
            batch_landms = np.zeros_like(landms)

            batch_target[:, [0, 2]] = targets[:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[:, [1, 3]] * in_h

            batch_landms[:, [0, 2, 4, 6, 8]] = landms[:, [0, 2, 4, 6, 8]] * in_w
            batch_landms[:, [1, 3, 5, 7, 9]] = landms[:, [1, 3, 5, 7, 9]] * in_h
            batch_landms[:, 10] = landms[:, 10]

            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
            max_ratios = np.max(ratios, axis = -1)

            for t, ratio in enumerate(max_ratios):

                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True

                for k, mask in enumerate(self.anchors_mask[layer]):
                    if not over_threshold[mask]:
                        continue

                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))

                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)

                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[layer][k, local_j, local_i] != 0:
                            if box_best_ratio[layer][k, local_j, local_i] > ratio[mask]:
                                y_true[layer][k, local_j, local_i, :] = 0
                            else:
                                continue
                        
                        y_true[layer][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[layer][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[layer][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[layer][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[layer][k, local_j, local_i, 4] = 1

                        y_true[layer][k, local_j, local_i, 5] = batch_landms[t, 0] - local_i
                        y_true[layer][k, local_j, local_i, 6] = batch_landms[t, 1] - local_j
                        y_true[layer][k, local_j, local_i, 7] = batch_landms[t, 2] - local_i
                        y_true[layer][k, local_j, local_i, 8] = batch_landms[t, 3] - local_j
                        y_true[layer][k, local_j, local_i, 9] = batch_landms[t, 4] - local_i
                        y_true[layer][k, local_j, local_i, 10] = batch_landms[t, 5] - local_j
                        y_true[layer][k, local_j, local_i, 11] = batch_landms[t, 6] - local_i
                        y_true[layer][k, local_j, local_i, 12] = batch_landms[t, 7] - local_j
                        y_true[layer][k, local_j, local_i, 13] = batch_landms[t, 8] - local_i
                        y_true[layer][k, local_j, local_i, 14] = batch_landms[t, 9] - local_j
                        y_true[layer][k, local_j, local_i, 15] = batch_landms[t, 10]

                        if batch_landms[t, :].any() < 0:
                            y_true[layer][k, local_j, local_i, 15] = 1
                            

                        box_best_ratio[layer][k, local_j, local_i] = ratio[mask]
        return y_true

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    landmarks = []
    y_trues = [[] for _ in batch[0][3]]

    for img, box, landm, y_true in batch:
        images.append(img)
        bboxes.append(box)
        landmarks.append(landm)
        for i, sub_y_true in enumerate(y_true):
            y_trues[i].append(sub_y_true)
        
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    landmarks = [torch.from_numpy(landm).type(torch.FloatTensor) for landm in landmarks]
    y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]

    return images, bboxes, landmarks, y_trues