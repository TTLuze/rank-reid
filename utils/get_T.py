import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def test_T():
    img_path = '../../dataset/bobo/resize/'
    #ori_name = 'IMG_1008.JPG'
    ori_name = '1502_c8l04f0.JPG'
    ori_path = os.path.join(img_path, ori_name)

    img = cv2.imread(ori_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pts1 = np.float32([[89,139],[189,137],[94,287],[183,290]]) #左上，右上，左下，右下
    pts2 = np.float32([[152,159],[256,152],[150,306],[242,311]])
    #pts2 = np.float32([[0,0],[500,0],[0,700],[500,700]])


    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300,700))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def get_mask():
    pass

def test_mask():
    img_path = '../../dataset/bobo/resize/'
    #ori_name = 'IMG_1008.JPG'
    ori_name = '1502_c8l04f0.JPG'
    ori_path = os.path.join(img_path, ori_name)

    img = cv2.imread(ori_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pts1 = np.float32([[89,139],[189,137],[94,287],[183,290]]) #左上，右上，左下，右下
    pts2 = np.float32([[152,159],[256,152],[150,306],[242,311]])
    #pts2 = np.float32([[0,0],[500,0],[0,700],[500,700]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    #mask = np.zeros([550,220], dtype=np.uint8)
    #mask[140:290,90:185] = 1
    #delta =
    mask = np.zeros((150,95,3),dtype=np.uint8)
    img[140:290,90:185] = mask

    dst = cv2.warpPerspective(img, M, (300,700))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

test_mask()