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
    new_img_dir = '/opt/ml/input/data/eval/new_imgs'
    image_path = '/opt/ml/input/data/eval/images/'
    df = pd.read_csv("/opt/ml/input/data/eval/info.csv")

    os.mkdir("/opt/ml/input/data/eval/new_imgs")

    cnt = 0

    for index in tqdm(range(len(df))):

        id = df.iloc[index].ImageID
        path = image_path + id
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mtcnn 적용
        boxes, probs = mtcnn.detect(img)

        if isinstance(boxes, np.ndarray):
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > 384:
                xmax = 384
            if ymax > 512:
                ymax = 512

            img = img[ymin:ymax, xmin:xmax, :]
        # boexes size 확인
        else:
            result_detected = RetinaFace.detect_faces(path)

            if type(result_detected) == dict:
                xmin = int(result_detected["face_1"]["facial_area"][0]) - 30
                ymin = int(result_detected["face_1"]["facial_area"][1]) - 30
                xmax = int(result_detected["face_1"]["facial_area"][2]) + 30
                ymax = int(result_detected["face_1"]["facial_area"][3]) + 30

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > 384:
                    xmax = 384
                if ymax > 512:
                    ymax = 512

                img = img[ymin:ymax, xmin:xmax, :]

            else:
                pass

        tmp = os.path.join(new_img_dir, id)
        cnt += 1

        plt.imsave(tmp, img)


if __name__ == '__main__':
    get_cropped_and_fixed_images()
