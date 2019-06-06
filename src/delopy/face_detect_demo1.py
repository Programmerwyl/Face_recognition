# import face_model
from src.delopy import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--model', default='C:/Users/wyl/Desktop/data/retinaface/model-r50-am-lfw', help='path to load model.')
parser.add_argument('--model', default='C:/Users/wyl/Desktop/data/retinaface/model-r50-am-lfw/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)


def img_to_encoding(image_path, model):
    img = cv2.imread(image_path)
    img = model.get_input(img)
    embedding = model.get_feature(img)
    return embedding

if __name__ == '__main__':
    image_path = 'C:/Users/wyl/Desktop/data/retinaface/image/result/7_Cheering_Cheering_7_81.jpg'
    img = cv2.imread(image_path)
    ret = model.detect_face(img)
    bbox, points = ret
    print(np.shape(bbox))
    print(np.shape(points))

    for box,point in zip(bbox,points):
        if box[4] < 0.95:
            continue
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (120, 100, 0), thickness=3)
        for i in range(5):
            cv2.circle(img, (int(point[ i]), int(point[i +5])), 2, (0, 0, 225), thickness=2)

    cv2.imshow('face',img)
    cv2.waitKey(0)