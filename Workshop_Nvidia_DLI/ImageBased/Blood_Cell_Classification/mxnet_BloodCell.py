from mxnet import nd, gluon, init, autograd, gpu
from mxnet.gluon import nn
import mxnet
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
import pickle
import numpy as np
import string
from skimage.color import rgb2gray

batch_size = 256
ctx = gpu(0)
input = '_20Per.pkl' #'.pkl' #
directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'
# mode = 'train'
# with open(directory + mode + '_Data' + input,'rb') as outputDir:
#     FullIm = pickle.load(outputDir)
#
# with open(directory + mode + '_Label' + input,'rb') as outputDir:
#     FullLb = pickle.load(outputDir)



def acc(output, label):
    Accuracy = (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()
    return Accuracy

def readingDataset(mode):

    # mxnet.random.seed(42) # Fix the seed for reproducibility
    # X = mxnet.random.uniform(shape=(10, 3))
    # y = mxnet.random.uniform(shape=(10, 1))
    # dataset = mxnet.gluon.data.dataset.ArrayDataset(X, y)

    # mnist_train = mxnet.gluon.data.vision.datasets.FashionMNIST(train=True)
    # X, y = mnist_train[10]
    # X

    # Normalizing the Data
    # mode = 'train'
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])

    with open(directory + mode + '_Data' + input,'rb') as outputDir:
        FullIm = pickle.load(outputDir)

    sz = FullIm.shape
    FullIm_Gray = np.zeros((sz[0],sz[1],sz[2],1))
    for i in range(sz[0]):
        im = FullIm[i,...].reshape((sz[1],sz[2],sz[3]))
        FullIm_Gray[i,...,0] = rgb2gray(im)

    with open(directory + mode + '_Label' + input,'rb') as outputDir:
        FullLb = pickle.load(outputDir)

    # FullLb2 = np.zeros((len(FullLb),1))
    # classes = np.unique(FullLb)
    # for cl in range(len(classes)):
    #     ind = np.where(FullLb == classes[cl])
    #     FullLb2[ind] = int(cl)

    FullLb_NDArray = mxnet.nd.array(np.asarray(FullLb)) # ,ctx=gpu(0)
    FullIm_NDArray = mxnet.nd.array(FullIm_Gray)# ,ctx=gpu(0)

    BloodCell_dataset = gluon.data.dataset.ArrayDataset(FullIm_NDArray, FullLb_NDArray)
    # BloodCell_dataset = BloodCell_dataset.transform_first(transformer)
    data = gluon.data.DataLoader( BloodCell_dataset , batch_size=batch_size , shuffle=True , num_workers=4)

    return data

train_data = readingDataset('train')
valid_data = readingDataset('test')



net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=32, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=64, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(4)
    )

for B in enumerate(train_data):
    data = B[1][0]
    data.shape
    output = net(data)
    break

# net.load_params('net.params', ctx=ctx)

# net.collect_params().initialize(ctx=ctx)

net.initialize(init=init.Xavier()) # ,ctx=ctx
print(net)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epoch = 2
for epoch in range(num_epoch):
    print(' --- epoch: ' + str(epoch) + ' --- ')
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time()
    step = 0
    for A in enumerate(train_data):
        data = A[1][0]
        label = A[1][1]
        step = step + 1
        print('   step: ' + str(step) + ' --- ')
        # print(data)
        # print(label)
        # data = data.as_in_contex(ctx)
        # label = label.as_in_contex(ctx)

        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)

    # calculate validation accuracy
    for data, label in valid_data:
        valid_acc += acc(net(data), label)

    print("Epoch %d: Loss: %.3f, Train acc %.3f, Test acc %.3f, Time %.1f sec" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data),
        valid_acc/len(valid_data), time()-tic))

net.save_params('net.params')
