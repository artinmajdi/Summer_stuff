from mxnet import nd, gluon, init, autograd, gpu
from mxnet.gluon import nn
import mxnet
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
import pickle
import numpy as np
import string

batch_size = 256


def acc(output, label):
    Accuracy = (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()
    return Accuracy

def readingDataset(mode):
    # Normalizing the Data
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])

    directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'
    with open(directory + mode + '_Data_Temp.pkl','rb') as outputDir:
        FullIm = pickle.load(outputDir)

    with open(directory + mode + '_Label_Temp.pkl','rb') as outputDir:
        FullLb = pickle.load(outputDir)

    FullLb2 = np.zeros((len(FullLb),1))
    classes = np.unique(FullLb)
    for cl in range(len(classes)):
        ind = np.where(FullLb == classes[cl])
        FullLb2[ind] = int(cl)

    print(FullIm.shape)

    FullLb_NDArray = mxnet.nd.array(FullLb2) # ,ctx=gpu(0)
    FullIm_NDArray = mxnet.nd.array(FullIm)
    BloodCell_dataset = gluon.data.dataset.ArrayDataset(FullIm_NDArray, FullLb_NDArray)
    # BloodCell_dataset = BloodCell_dataset.transform_first(transformer)
    data = gluon.data.DataLoader( BloodCell_dataset , batch_size=batch_size , shuffle=False , num_workers=4)

    return data


train_data = readingDataset('train')
valid_data = readingDataset('test')

for a,b in train_data:
    print(a.shape)
    print(b.shape)
    break



net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv3D(channels=32, kernel_size=(5,5,1), activation='relu'),
        nn.MaxPool3D(pool_size=2, strides=(2,2,1)),
        nn.Conv3D(channels=64, kernel_size=(5,5,1), activation='relu'),
        nn.MaxPool3D(pool_size=2, strides=(2,2,1)),
        # nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(4)
    )

net.initialize(init=init.Xavier())
print(net)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epoch = 10
for epoch in range(num_epoch):
    print(' --- epoch: ' + str(epoch) + ' --- ')
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time()
    step = 0
    for data, label in train_data:
        step = step + 1
        print('   step: ' + str(step) + ' --- ')
        # forward + backward
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
