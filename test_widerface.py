import os
import numpy as np
from PIL import Image
from utils.utils_yoloface import Yolo
from tqdm import tqdm

yoloface = Yolo()
origin_size = True

testset_folder = './Data/widerface/val/images/'
testset_list = testset_folder[:-7] + "wider_val.txt"
save_folder = './widerface_evaluate/widerface_txt/'

with open(testset_list, 'r') as fr:
    test_dataset = fr.read().split()
num_images = len(test_dataset)

for img_name in tqdm(test_dataset):
    image_path = testset_folder + img_name

    image = Image.open(image_path)

    target_size = 1600
    max_size = 2144

    im_shape = np.array(np.shape(image)[0:2])
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    resize = float(target_size) / float(im_size_min)
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    
    if origin_size:
        resize = 1
    
    width = im_shape[0]
    height = im_shape[1]

    width = width * resize
    height = height * resize

    width = width // 32 * 32
    height = height // 32 * 32


    input_shape = [int(width), int(height)]

    result = yoloface.detect_image(image, input_shape)

    save_name = save_folder + img_name[:-4] + ".txt"
    dirname = os.path.dirname(save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(save_name, "w") as fd:
        bboxs = result
        file_name = os.path.basename(save_name)[:-4] + "\n"
        if bboxs is not None:
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
        else:
            bboxs_num = str('0') + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
