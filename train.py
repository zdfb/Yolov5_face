import math
import torch
import random
import logging
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from nets.head.yolo import YoloFace
from utils.utils import get_anchors
from Loss.yolo_loss import YoloLoss
from utils.utils_fit import fit_one_epoch
from Data.dataloader import YoloDataset, yolo_dataset_collate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s',
                        level=logging.INFO, handlers=[logging.FileHandler('log/train.log', mode='w'), logging.StreamHandler()])


class Train_YoloFace():
    def __init__(self):
        super(Train_YoloFace, self).__init__()

        anchors_path = 'Data/yolo_anchors.txt'
        self.txt_path = 'Data/widerface/train/label.txt'
        
        self.anchors_mask = [[4, 5], [2, 3], [0, 1]]
        
        self.anchors, _ = get_anchors(anchors_path)

        model = YoloFace(anchors_mask = self.anchors_mask, ssh = True, backbone = 'cspdarknet', phi = 's')

        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.to(device)
        
        self.model = model

        self.epochs = 250

        
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.937,  weight_decay = 5e-4)

        lf = lambda x: (((1 + math.cos(x * math.pi / self.epochs)) / 2) ** 1.0) * 0.95 + 0.05
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lf)
        self.scheduler.last_epoch = -1


        self.loss_min = 1e9 
    
    def train(self, batch_size, input_shape, epoch_now, epoch_length):
        
        train_dataset = YoloDataset(self.txt_path, input_shape = input_shape, anchors = self.anchors, anchors_mask = self.anchors_mask, train = True, epoch_length = epoch_length, epoch_now = epoch_now, mosaic = True, mixup = True)
        train_data = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 4, pin_memory = True, drop_last = True, collate_fn = yolo_dataset_collate)

        yolo_loss = YoloLoss(anchors = self.anchors, input_shape = input_shape, cuda = True, anchors_mask = self.anchors_mask)
        
        train_loss, loc_loss, landm_loss, conf_loss, time_now = fit_one_epoch(self.model, yolo_loss, self.optimizer, train_data, device)
 
        if train_loss < self.loss_min:
            self.loss_test_min = train_loss
            torch.save(self.model.state_dict(), 'yolov5_face.pth')
        self.scheduler.step()
        logging.info("epoch: {}, input_shape: {}, train_loss: {:.6f}, loc_loss : {:.6f}, landm_loss: {:.6f}, conf_loss : {:.6f}, time: {:.4f}\n ".format(epoch_now, input_shape[0], train_loss, loc_loss, landm_loss, conf_loss, time_now))

                
        
    def train_model(self):

        batch_size = 64

        for epoch in range(self.epochs):
            input_shape = [640, 640]
            self.train(batch_size, input_shape, epoch, self.epochs)

            if (epoch + 1) % 20 == 0:
                torch.save(self.model.state_dict(), 'yolov5_face' + str(epoch + 1) +'pth')

if __name__ == "__main__":
    train = Train_YoloFace()
    train.train_model()