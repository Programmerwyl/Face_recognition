
from src.delopy import face_model
import argparse
import cv2
import sys
import numpy as np
import os

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


def img_to_encoding(img, model):
    img = model.get_input(img)
    embedding = model.get_feature(img)
    return embedding


database = {}
database["yunlong"] = img_to_encoding(cv2.imread("C:/Users/wyl/Desktop/data/retinaface/image/yunlong.jpg"), model)


def who_is_it(img,database,model):
    embedding = img_to_encoding(img,model)
    min_dist = 1.0
    identity = None
    for name,bd in database.items():
        dist = np.sum(np.square(embedding-bd))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist,identity


if __name__ == '__main__':
    for path,dir_list,file_list in os.walk('C:/Users/wyl/Desktop/data/retinaface/image/face'):
        for image in file_list:
            img_path = os.path.join(path,image)
            img = cv2.imread(img_path)
            who_is_it(img,database,model)

