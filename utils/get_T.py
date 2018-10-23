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

def test_mask():
    img_path = '../../dataset/bobo/resize/'
    #ori_name = 'IMG_1008.JPG'
    ori_name = '1502_c8l04f0.JPG'
    tar_name = '1502_c9l05f0.JPG'
    ori_path = os.path.join(img_path, ori_name)
    tar_path = os.path.join(img_path, tar_name)

    img = cv2.imread(ori_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread(tar_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    pts1 = np.float32([[89,139],[189,137],[94,287],[183,290]]) #左上，右上，左下，右下
    pts2 = np.float32([[52,159],[156,152],[50,306],[142,311]])
    #pts2 = np.float32([[0,0],[500,0],[0,700],[500,700]])
    M = cv2.getPerspectiveTransform(pts1, pts2)


    mask = np.zeros([550,220,3], dtype=np.uint8)
    mask[140:290,90:185,:] = 1

    noise = np.ones((550,220,3),dtype=np.float32)*100 * mask
    dst = cv2.warpPerspective(noise, M, (220, 550))

    dst2 = img2*(dst<1e-1) + dst
    dst2 = dst2.astype(np.uint8)


    #dst = cv2.warpPerspective(img, M, (300,700))

    plt.subplot(141),plt.imshow(img),plt.title('Input')
    plt.subplot(142),plt.imshow(img2),plt.title('Output')
    plt.subplot(143), plt.imshow(dst), plt.title('Output')
    plt.subplot(144), plt.imshow(dst2), plt.title('Output')
    plt.show()


def get_T(pts1, pts2):
    trans = cv2.getPerspectiveTransform(pts1, pts2)
    return trans

#x是非a0的图，转换方式为trans
#mask为a0的加噪区域
#mask 是大小为（550，220，3）的0，1矩阵，np.uint8
#noise是大小为（550，220，3）的矩阵，np.float32，是的优化变量
#mask和trans都是设定好的，其中mask和trans计算的pst1也许可以共用 待写
def get_adv_x(x, noise, mask, trans):
    noise = noise*mask

    # dst为经过trans后的噪声矩阵，0表示不加噪声区域，非0表示噪声值
    dst = cv2.warpPerspective(noise, trans, (220, 550))

    #将噪声覆盖给x
    adv_x = x*(dst<1e-1) + dst
    return adv_x



test_mask()