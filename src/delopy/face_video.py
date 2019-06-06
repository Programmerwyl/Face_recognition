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

database = {}
database["yunlong"] = img_to_encoding("C:/Users/wyl/Desktop/data/retinaface/image/yunlong.jpg", model)

video_path = 'C:/Users/wyl/Desktop/data/testdata/VID_20190411_104846.mp4'

def who_is_it(embedding,database):
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
    cap = cv2.VideoCapture(video_path)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        ret = model.detect_face(frame)
        bbox, points = ret
        for box, point in zip(bbox, points):
            if box[4] < 0.9:
                continue
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (120, 100, 0), thickness=3)
            for i in range(5):
                cv2.circle(frame, (int(point[i]), int(point[i + 5])), 2, (0, 0, 225), thickness=2)
            # points = point[0, :].reshape((2, 5)).T
            # aligned = model.get_input_single(frame,box,point)
            # embedding = model.get_feature(aligned)
            # min_dist, identity = who_is_it(embedding,database)
        cv2.imshow('face',frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
