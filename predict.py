import cv2
import numpy as np
from PIL import Image
from utils.utils_yoloface import Yolo

image_path = 'selfie.jpg'

image = Image.open(image_path)

yoloface = Yolo()

result = yoloface.detect_image(image, [1600, 1600])

boxes = result[:, :4]
landmarks = result[:, 5:]

image = cv2.imread(image_path)

print(len(boxes))

for i in range(len(boxes)):
    box = boxes[i]
    b = landmarks[i] 
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

    cv2.circle(image, (int(b[0]), int(b[1])), 1, (0, 0, 255), 4)
    cv2.circle(image, (int(b[2]), int(b[3])), 1, (0, 255, 255), 4)
    cv2.circle(image, (int(b[4]), int(b[5])), 1, (255, 0, 255), 4)
    cv2.circle(image, (int(b[6]), int(b[7])), 1, (0, 255, 0), 4)
    cv2.circle(image, (int(b[8]), int(b[9])), 1, (255, 0, 0), 4)

    
cv2.imwrite('result.jpg', image)


