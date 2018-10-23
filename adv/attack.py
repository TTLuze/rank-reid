#首先测试我们resize的数据集的效果，有必要的话做transfer learning
#接着搜集mask、trans的信息
#写adv代码
#10.23 上述全部完成

from keras.models import Model
from keras.models import load_model

import sys
sys.path.append('../')
from baseline_dis import evaluate


print(evaluate)
net = load_model('../adv/market-pair-pretrain.h5')
net = Model(inputs=[net.get_layer('resnet50').get_input_at(0)],
            outputs=[net.get_layer('resnet50').get_output_at(0)])

