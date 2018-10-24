import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf

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

    #test for contrib.image.transform
    #with tf.device('/cpu:0'):
    img_tf = tf.Variable(noise)
    M = np.array(np.mat(M).I)
    transform1 = np.reshape(M, 9)[:8]
    #transform1 = [1,0,0,0,1,0,0,0]
    noise2 = tf.contrib.image.transform(img_tf, transform1, 'BILINEAR')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        noise2 = sess.run(noise2)



    #dst = cv2.warpPerspective(img, M, (300,700))

    plt.subplot(141),plt.imshow(img),plt.title('Input')
    plt.subplot(142),plt.imshow(img2),plt.title('Output')
    plt.subplot(143), plt.imshow(noise2), plt.title('Output')
    plt.subplot(144), plt.imshow(dst2), plt.title('Output')
    plt.show()

#当需要增加/修改 某个person的mask时调用
#pid为用户id，pts为一个长度为4的list，注意与trans的顺序不同
def save_mask(pid, pts, dir_path="../../dataset/bobo/mask/"):

    mask = np.zeros([550,220,3], dtype=np.uint8)
    mask[pts[0]:pts[1],pts[2]:pts[3],:] = 1

    np.save(os.path.join(dir_path, str(pid)+'.npy'), mask)

#only excute once
def init():
    dir_path="../../dataset/bobo/mask/"
    pts1 = np.zeros((792,4,2), dtype=np.float32)
    pts2 = np.zeros((792,4,2), dtype=np.float32)
    np.save(os.path.join(dir_path, 'pts1.npy'), pts1)
    np.save(os.path.join(dir_path, 'pts2.npy'), pts2)
    transforms = np.zeros((792,8), dtype=np.float32)
    
    for i in range(792):
        transform = cv2.getPerspectiveTransform(pts1[i], pts2[i])
        transforms[i] = np.reshape(transform,9)[:8]
    
    np.save(os.path.join(dir_path, 'transforms.npy'), transforms)


#当需要增加/修改 某张图片的trans时调用
#img_index:该图片下标；pts1，pts2表示图片相对与a0的映射坐标
def set_pts(img_index, pts1, pts2):
    dir_path = "../../dataset/bobo/mask/"
    pts1_np = np.load(os.path.join(dir_path, "pts1.npy"))
    pts2_np = np.load(os.path.join(dir_path, "pts2.npy"))
    transforms = np.load(os.path.join(dir_path, "transforms.npy"))

    pts1_np[img_index] = pts1
    pts2_np[img_index] = pts2

    transform = cv2.getPerspectiveTransform(pts1, pts2)
    transforms[img_index] = np.reshape(transform,9)[:8]

    np.save(os.path.join(dir_path, 'pts1.npy'), pts1_np)
    np.save(os.path.join(dir_path, 'pts2.npy'), pts2_np)
    np.save(os.path.join(dir_path, 'transforms.npy'), transforms)

#test_mask()

#pts1 = np.float32([[89,139],[189,137],[94,287],[183,290]]) #左上，右上，左下，右下
#pts2 = np.float32([[89,139],[189,137],[94,287],[183,290]])
#set_pts(32, pts1, pts2)
test_mask()