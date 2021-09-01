import torch
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os
import cv2
from tqdm import tqdm
from retinaface import RetinaFace
import numpy as np
import pandas as pd


def get_cropped_and_fixed_images():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mtcnn = MTCNN(keep_all=True, device=device)
    new_img_dir = '/opt/ml/input/data/train/new_imgs'
    df = pd.read_csv("/opt/ml/code/labeled_data.csv")

    os.mkdir("/opt/ml/input/data/train/new_imgs")

    cnt = 0

    for id_index in tqdm(range(len(df)//7)):
        normal_img_index = id_index*7 + 3
        normal_path = df.iloc[normal_img_index].img_path
        normal_img = cv2.imread(normal_path)

        img_fixed_dir = '_'.join(
            [df.iloc[id_index*7].id, df.iloc[id_index*7].gender, "Asian", str(df.iloc[id_index*7].age)])

        tmp = os.path.join(new_img_dir, img_fixed_dir)

        try:
            os.mkdir(tmp)
        except:
            pass

        # mtcnn 적용
        boxes, _ = mtcnn.detect(normal_img)
        padding_x = 50
        padding_y = 70

        if isinstance(boxes, np.ndarray):
            xmin = int(boxes[0, 0])-padding_x
            ymin = int(boxes[0, 1])-padding_x
            xmax = int(boxes[0, 2])+padding_y
            ymax = int(boxes[0, 3])+padding_y

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > 384:
                xmax = 384
            if ymax > 512:
                ymax = 512

        # boexes size 확인
        else:
            result_detected = RetinaFace.detect_faces(normal_img)

            if type(result_detected) == dict:
                xmin = int(result_detected["face_1"]
                           ["facial_area"][0]) - padding_x
                ymin = int(result_detected["face_1"]
                           ["facial_area"][1]) - padding_x
                xmax = int(result_detected["face_1"]
                           ["facial_area"][2]) + padding_y
                ymax = int(result_detected["face_1"]
                           ["facial_area"][3]) + padding_y

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > 384:
                    xmax = 384
                if ymax > 512:
                    ymax = 512
            else:
                xmin = 0
                ymin = 0
                xmax = 384
                ymax = 512

        for img_type_index in range(7):
            img_index = id_index*7+img_type_index
            img_path = df.iloc[img_index].img_path
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[ymin:ymax, xmin:xmax, :]
            plt.imsave(os.path.join(tmp, df.iloc[img_index].stem+'.jpg'), img)


if __name__ == '__main__':
    get_cropped_and_fixed_images()
