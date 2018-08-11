import numpy as np
import tensorflow as tf
import sonnet as snt
import mxnet.gluon.model_zoo as zoo

models = zoo.vision

alexnet = models.alexnet(pretrained=True)

print(alexnet)
vgg = models.vgg19(pretrained=True)

print(vgg)


# Example of one iteration Two GPU neural Net
data = next_batch()
data[gpu0].copyfrom(data[0:50])
data[gpu1].copyfrom(data[50:100])
# forward, backprop on GPU 0
fc1[gpu0] = FullcForward(data[gpu0], fc1_weight[gpu0])
fc2[gpu0] = FullcForward(fc1[gpu0], fc2_weight[gpu0])
fc2_ograd[gpu0] = LossGrad(fc2[gpu0], label[0:50])
fc1_ograd[gpu0], fc2_wgrad[gpu0] = FullcBackward(fc2_ograd[gpu0] , fc2_weight[gpu0])
              _, fc1_wgrad[gpu0] = FullcBackward(fc1_ograd[gpu0] , fc1_weight[gpu0])

# forward, backprop on GPU 1
fc1[gpu1] = FullcForward(data[gpu1], fc1_weight[gpu1])
fc2[gpu1] = FullcForward(fc1[gpu1], fc2_weight[gpu1])
fc2_ograd[gpu1] = LossGrad(fc2[gpu1], label[50:100])
fc1_ograd[gpu1], fc2_wgrad[gpu1] = FullcBackward(fc2_ograd[gpu1] , fc2_weight[gpu1])
              _, fc1_wgrad[gpu1] = FullcBackward(fc1_ograd[gpu1] , fc1_weight[gpu1])

# aggregate gradient and update
fc1_wgrad[cpu]  = fc1_wgrad[gpu0] + fc1_wgrad[gpu1]
fc2_wgrad[cpu]  = fc2_wgrad[gpu0] + fc2_wgrad[gpu1]
fc1_weight[cpu] -= lr *  fc1_wgrad[cpu]
fc2_weight[cpu] -= lr *  fc2_wgrad[cpu]
fc1_weight[cpu].copyto(fc1_weight[gpu0] , fc1_weight[gpu1])
fc2_weight[cpu].copyto(fc2_weight[gpu0] , fc2_weight[gpu1])
