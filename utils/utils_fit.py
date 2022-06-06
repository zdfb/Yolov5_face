import time
import torch
import numpy as np


def fit_one_epoch(model, yolo_loss, optimizer, train_data, device): 
    
    start_time = time.time() 
    model.train() 
    
    loss_list = []
    loss_loc_list = []
    loss_landm_list = []
    loss_conf_list = []

    for step, data in enumerate(train_data):   
        images, targets, landmarks, y_trues  = data[0], data[1], data[2], data[3]


        with torch.no_grad():
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
            landmarks = [ann.to(device) for ann in landmarks]
            y_trues = [ann.to(device) for ann in y_trues]
           
        optimizer.zero_grad() 
        outputs = model(images)

        loss_value_all = 0 
        loss_loc_all = 0
        loss_landm_all = 0
        loss_conf_all = 0

        for index in range(len(outputs)):

            loss_item, loss_loc_item, loss_landm_item, loss_conf_item = yolo_loss(index, outputs[index], targets, y_trues[index])
            loss_value_all += loss_item
            loss_loc_all += loss_loc_item
            loss_landm_all += loss_landm_item
            loss_conf_all += loss_conf_item

        loss_value = loss_value_all

        loss_value.backward()  
        optimizer.step()

        loss_list.append(loss_value.item())
        loss_loc_list.append(loss_loc_all.item())
        loss_landm_list.append(loss_landm_all.item())
        loss_conf_list.append(loss_conf_all.item())

        rate = (step + 1) / len(train_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss_value), end="")
    print()

    train_loss = np.mean(loss_list)
    loc_loss = np.mean(loss_loc_list)
    landm_loss = np.mean(loss_landm_list)
    conf_loss = np.mean(loss_conf_list)
    time_now = time.time() - start_time

    return train_loss, loc_loss, landm_loss, conf_loss, time_now