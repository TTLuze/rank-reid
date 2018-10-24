#首先测试我们resize的数据集的效果:把5个人随机挑选一部分作为probe，其余人并入gallery做galery
# 有必要的话做transfer learning
#接着搜集mask、trans的信息
#写adv代码
#10.23 上述全部完成

import tensorflow as tf
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import numpy as np
import scipy.misc
import json

import time
import os
import sys
sys.path.append('../')
from baseline_dis.evaluate_v2 import extract_feature, sort_similarity, map_rank_quick_eval
from utils.file_helper import write

#这个版本可以直接运行，不好的地方是读取img用的是keras原生，因此弃了
def test_predict(net, probe_path, gallery_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    Query_f, Query_info = extract_feature(probe_path, net)

    a=time.time()
    gallery_f, gallery_info = extract_feature(gallery_path, net)
    b=time.time()
    print(b-a)

    #index为从query中随机选出的18张照片的下标，index_g为加入到gallery的照片下标
    #选择规则是每人每个C下选一张，选择的id是从（0，num_id）， 其余用作training
    #360即为第5个id的第一张照片下标;24为每人每C下的张数
    #t表示每人每C选t张到gallery中
    num_id = 11
    t=6
    index = np.random.randint(0,24,size=3*num_id)
    index = index + np.arange(3*num_id)*24

    index_g = np.random.randint(0,24,size=3*num_id*t)+ (np.arange(3*num_id)).repeat(t)*24
    index_g = list(set(index_g) - set(index))

    query_f = Query_f[index]
    query_info = [Query_info[i] for i in index]

    gallery_f = np.concatenate((gallery_f, Query_f[index_g]), axis=0)
    gallery_info.extend([Query_info[i] for i in index_g]) #由于info是元组的list

    result, result_argsort = sort_similarity(query_f, gallery_f)

    log_path = 'market_result_eval.log'
    rank1_acc, rank5_acc, rank10_acc, mAP = map_rank_quick_eval(query_info, gallery_info, result_argsort)
    write(log_path, '%f\t%f\t%f\t%f\n' % (rank1_acc, rank5_acc, rank10_acc, mAP))

#dir_path为路径，img_names为待load的照片名list（用的时候是dir下全体照片的名字）
#index有值的时候，仅load index；否则load全体img_names
#load函数使用的是scipy.misc.imread
def load(dir_path, img_names, index=None):
    if index is None:
        index = np.arange(len(img_names))

    image=[]
    infos = []

    for i in index:
        image_path = os.path.join(dir_path, img_names[i])
        x = np.array(scipy.misc.imresize(scipy.misc.imread(image_path),(224,224)),dtype=np.float32)
        image.append(x)

        arr = img_names[i].split('_')
        person = int(arr[0])
        camera = int(arr[1][1])
        infos.append((person, camera))

    image = np.array(image)
    image = preprocess_input(image)
    return image, infos

#相比与load()，没有resize和preprocessing
def load_raw(dir_path, img_names, index=None):
    if index is None:
        index = np.arange(len(img_names))

    image=[]
    infos = []

    for i in index:
        image_path = os.path.join(dir_path, img_names[i])
        x = np.array(scipy.misc.imread(image_path),dtype=np.float32)
        image.append(x)

        arr = img_names[i].split('_')
        person = int(arr[0])
        camera = int(arr[1][1])
        infos.append((person, camera))

    image = np.array(image)
    return image, infos

def evaluate(net, probe_path, gallery_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    img_names = sorted(os.listdir(probe_path))
    gallery_name = sorted(os.listdir(gallery_path))[:-1]

    #设定好probe集合中用作probe的index，和将要加入到gellary的index

    # 选择规则是每人每个C下选一张，选择的id是从（0，num_id）， 其余用作training
    #index为从query中随机选出的3*num_id张照片的下标，index_g为加入到gallery的照片下标
    #24为每人每C下的张数
    #t表示每人每C选t张到gallery中
    num_id = 11
    t = 6
    index = np.random.randint(0, 24, size=3 * num_id)
    index = index + np.arange(3 * num_id) * 24
    index_g = np.random.randint(0, 24, size=3 * num_id * t) + (np.arange(3 * num_id)).repeat(t) * 24
    index_g = list(set(index_g) - set(index))

    #读取probe,gallery，并计算其features
    query, query_info = load(probe_path, img_names, index)
    query_f = net.predict(query, batch_size=128)
    gallery_add, gallery_add_infos = load(probe_path, img_names, index_g)
    gallery_add_f = net.predict(gallery_add, batch_size=128)

    #gallery, gallery_info = load(gallery_path, gallery_name)
    #gallery_f = net.predict(gallery, batch_size=128)
    gallery_f = np.load('gallery_f.npy') #本地持久化，节约时间
    gallery_info = [tuple(x) for x in json.load(open('gallery_info.json', 'r'))]

    gallery_f = np.concatenate((gallery_f, gallery_add_f), axis=0)
    gallery_info.extend([gallery_add_infos[i] for i in range(len(index_g))]) #由于info是元组的list

    #计算相似度，排序，计算CMC，mAP
    result, result_argsort = sort_similarity(query_f, gallery_f)
    log_path = 'market_result_eval.log'
    rank1_acc, rank5_acc, rank10_acc, mAP = map_rank_quick_eval(query_info, gallery_info, result_argsort)
    write(log_path, '%f\t%f\t%f\t%f\n' % (rank1_acc, rank5_acc, rank10_acc, mAP))

#相比evaluate，这里是设定好了probe（一个人）和gallery的id
def single_evaluate(net, probe_path, gallery_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    img_names = sorted(os.listdir(probe_path))
    gallery_name = sorted(os.listdir(gallery_path))[:-1]

    '''
    num_id = 1
    t = 6
    index = np.random.randint(0, 24, size=1 * num_id)
    index = index + np.arange(1 * num_id) * 24
    index_g = np.random.randint(0, 24, size=3 * 11 * t) + (np.arange(3 * 11)).repeat(t) * 24
    index_g = sorted(list(set(index_g) - set(index)))
    '''

    index = np.array([30])
    index_g = np.concatenate(
        (np.array([0,4,8,12,18,20,24,28,32,40,44,46,50,54,58,62,68,70]),
         np.random.randint(0, 24, size=3 * 10 * 6) + (np.arange(3 * 10)).repeat(6) * 24 + 72)
    )

    #读取probe,gallery，并计算其features
    query, query_info = load(probe_path, img_names, index)
    query_f = net.predict(query, batch_size=128)
    gallery_add, gallery_add_infos = load(probe_path, img_names, index_g)
    gallery_add_f = net.predict(gallery_add, batch_size=128)

    #gallery, gallery_info = load(gallery_path, gallery_name)
    #gallery_f = net.predict(gallery, batch_size=128)
    gallery_f = np.load('gallery_f.npy') #本地持久化，节约时间
    gallery_info = [tuple(x) for x in json.load(open('gallery_info.json', 'r'))]

    gallery_f = np.concatenate((gallery_f, gallery_add_f), axis=0)
    gallery_info.extend([gallery_add_infos[i] for i in range(len(index_g))]) #由于info是元组的list

    #计算相似度，排序，计算CMC，mAP
    result, result_argsort = sort_similarity(query_f, gallery_f)
    log_path = 'adv_market_result_eval.log'
    rank1_acc, rank5_acc, rank10_acc, mAP = map_rank_quick_eval(query_info, gallery_info, result_argsort)
    write(log_path, '%f\t%f\t%f\t%f\n' % (rank1_acc, rank5_acc, rank10_acc, mAP))
'''
#test the evaluation
net = load_model('../baseline_dis/market-pair-pretrain.h5')
net = Model(inputs=[net.get_layer('resnet50').get_input_at(0)],
            outputs=[net.get_layer('resnet50').get_output_at(0)])

gallery_path = '../../dataset' + '/Market-1501/bounding_box_test'
probe_path = '../../dataset' + '/bobo/resize'
single_evaluate(net, probe_path, gallery_path)
evaluate(net, probe_path, gallery_path)
'''

def eucl_dist(inputs):
    x, y = inputs
    # return K.mean(K.square((x - y)), axis=1)
    return tf.square((x - y))


LEARNING_RATE = 1e-2
MAX_ITERATIONS = 100

#mask表示对a0图的加噪区域
#img1,img2表示用来计算adv用的集合.读入的是未resize，未preprocess的图像
#img1,img2是pair对，表示同一个人不同C下的pair
#trans1表示img1每张图相对a0的transform集合；同理trans2
def adv(net, mask, generator):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])

    #初始化噪声出纯色（目前是灰色）
    modifier = np.ones((550,220,3),dtype=np.float32)*100
    modifier = tf.Variable(modifier)
    noise = tf.tanh(modifier)*255

    x1 = tf.placeholder(shape=(None,550, 220, 3), dtype='float32')
    transform1 = tf.placeholder(shape=(None, 8), dtype='float32')
    x2 = tf.placeholder(shape=(None,550, 220, 3), dtype='float32')
    transform2 = tf.placeholder(shape=(None, 8), dtype='float32')

    noise1 = tf.contrib.image.transform(noise * mask, transform1, 'BILINEAR') #这里transform可能要转置
    noise2 = tf.contrib.image.transform(noise * mask, transform2, 'BILINEAR')

    x1_adv_o = x1*(tf.cast(noise1<1e-1, dtype=tf.float32)) + noise1 #the function output
    x2_adv_o = x2*(tf.cast(noise2<1e-1, dtype=tf.float32)) + noise2

    x1_adv = tf.image.resize_images(x1_adv_o, [224,224]) #这里的双向线性插值是默认值可以丢掉
    x2_adv = tf.image.resize_images(x2_adv_o, [224,224])

    x1_adv = preprocess_input(x1_adv)
    x2_adv = preprocess_input(x2_adv)

    feature1 = tf.squeeze(net(x1_adv),axis=[1,2])
    feature2 = tf.squeeze(net(x2_adv),axis=[1,2])

    feature1_norm = tf.nn.l2_normalize(feature1, axis=1)
    feature2_norm = tf.nn.l2_normalize(feature2, axis=1)
    dist_batch = tf.matmul(feature1_norm, feature2_norm, transpose_a=False, transpose_b=True)
    dist = tf.reduce_sum(dist_batch, axis=[0])

    # Setup the adam optimizer and keep track of variables we're creating
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(-dist, var_list=[modifier])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]
    init = tf.variables_initializer(var_list=[new_vars]+modifier)

    #Run the attack
    # method2: BIM
    # w_grads = tf.gradients(dist, modifier)

    #method1: optimization
    with tf.Session() as sess:
        sess.run(init)
        for iteration in range(MAX_ITERATIONS):
            img1, img2, trans1, trans2 = next(generator)
            _, dist_np = sess.run([train, dist], feed_dict={x1:img1, x2:img2,
                                                            transform1:trans1,
                                                            transform2:trans2})

            if iteration % (MAX_ITERATIONS // 10) == 0:
                print(iteration, dist_np)
        noise_np = sess.run(noise)

    return noise_np

def pair_generator(imgs, infos, trans, batch_size):
    camera_id = [x[1] for x in infos]

    while True:
        left_index=[]
        right_index=[]
        tmp_len=0

        while True:
            tmp = np.random.randint(len(imgs), size=2)
            if camera_id[tmp[0]] != camera_id[tmp[1]]:
                left_index.append(tmp[0])
                right_index.append(tmp[1])
                tmp_len+=1
            else:
                continue

            if tmp_len == batch_size:
                break

        yield imgs[left_index], imgs[right_index], trans[left_index], trans[right_index]

def attack(net, probe_path, mask_path):
    #load mask
    pid = 1502
    mask = np.load(os.path.join(mask_path, str(pid)+'.npy'))

    #load imgs, infos, trans
    img_names = sorted(os.listdir(probe_path))
    index = np.array([0,4,8,12,18,20,24,28,30,32,40,44,46,50,54,58,62,68,70])
    imgs, infos = load_raw(probe_path, img_names, index)
    trans = np.load(os.path.join(mask_path,'transforms.npy'))[index]  #trans.npy是包含了792张照片的trans，(792，8)

    #generate the pair
    batch_size = 16
    generator = pair_generator(imgs, infos, trans, batch_size)

    noise = adv(net, mask, generator)

    #接着利用noise imgs trans把图片保存至本地

if __name__ == '__main__':
    net = load_model('../baseline_dis/market-pair-pretrain.h5')
    net = Model(inputs=[net.get_layer('resnet50').get_input_at(0)],
                outputs=[net.get_layer('resnet50').get_output_at(0)])

    gallery_path = '../../dataset' + '/Market-1501/bounding_box_test'
    probe_path = '../../dataset' + '/bobo/resize'

    mask_path = '../../dataset' + '/bobo/mask'

    attack(net, probe_path, mask_path)

