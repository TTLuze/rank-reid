import os
import numpy as np
from keras.preprocessing import image
import scipy.misc
from keras.applications.resnet50 import preprocess_input

dir_path = 'E:\PycharmProjects\cvpr 2018\exp\dataset\Market-1501\query'
ori = '0001_c1s1_001051_00.jpg'

image_name1 = '1.jpg'
image_name2 = '2.png'
dist_path = 'C:\\Users\Administrator\Desktop'

image_path = os.path.join(dir_path, ori)
x = np.array(scipy.misc.imread(image_path),dtype=np.float32)
#x = np.array(scipy.misc.imresize(scipy.misc.imread(image_path),(224,224)),dtype=np.float32)
x = preprocess_input(x)


#perturb the image x


#transfrom back
mean = [103.939, 116.779, 123.68]
x[..., 2] += mean[2]
x[..., 1] += mean[1]
x[..., 0] += mean[0]
x = x[...,::-1]# 'BGR'->'RGB'    #noted that 'RGB'->'BGR' x = x[..., ::-1]

#save the perturbed image
x = x.astype(np.uint8)
image_path = os.path.join(dist_path, image_name2)
scipy.misc.imsave(image_path, x)