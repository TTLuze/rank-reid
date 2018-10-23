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
#img = np.array(scipy.misc.imresize(scipy.misc.imread(f),(299,299)),dtype=np.float32)
# /255-.5


image_data = preprocess_input(x)



x = x.astype(np.uint8)
image_path = os.path.join(dist_path, image_name2)
scipy.misc.imsave(image_path, x)