from __future__ import print_function
import numpy as np
import mxnet
from mxnet import nd, autograd, gluon, gpu
import os
import cv2
import scipy
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from skimage.color import rgb2gray
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

mxnet.random.seed(1)

ctx = mxnet.gpu(2)

## -------------------------------------------------------------------------------

batch_size = 32 # worked: 64
# input = '_20Per.pkl' #'.pkl' #
HEIGHT = 240
WIDTH = 320

downsample = 4
HEIGHT2 = int(HEIGHT/downsample)
WIDTH2 = int(WIDTH/downsample)
epochs = 110
smoothing_constant = .001

# num_inputs = HEIGHT * WIDTH
num_outputs = 4
directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'
input = '.pkl' # '_20Per.pkl' #
# numSteps = 10000 # 20000
# learning_rate = 0.01

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

def LoadingListOfFiles():

    mode = 'train'
    subClasses = os.listdir(directory + mode)
    for mode in ['train', 'test']:

        Data = []
        Label = []
        Label_Tag = []
        for i in range(len(subClasses)):
            dirr = directory + mode + '/' + subClasses[i] + '/'
            files2 = os.listdir(dirr)
            # for f in tqdm(range(int(len(files2)))):
            for f in range(int(len(files2))):
                le = len(files2[f])
                if files2[f][le-4:] == 'jpeg':
                    Data.append(directory + mode + '/' + subClasses[i] + '/' + files2[f])
                    Label_Tag.append(subClasses[i])
                    Label.append(i)

        Data = np.asarray(Data)
        Label = np.asarray(Label,dtype=np.int32)
        Label_Tag = np.asarray(Label_Tag)

        indexesRandomized = np.random.permutation(Data.shape[0])

        Data = Data[indexesRandomized]
        Label = Label[indexesRandomized]
        Label_Tag = Label_Tag[indexesRandomized]

        # Label = OneHotEncoder(n_values=n_classes).fit_transform(Label.reshape(-1, 1)).toarray()

        if mode == 'train':
            FilesAddress = {'train_data':Data,'train_label':Label,'train_label_tag':Label_Tag}
        else:
            FilesAddress['test_data'] = Data
            FilesAddress['test_label'] = Label
            FilesAddress['test_label_tag'] = Label_Tag

    # TrainPercentage = 0.8
    # L = int( TrainPercentage*Data.shape[0] )
    # FilesAddress = {'trainDataset':Data[:L,...],'train_label':Label[:L,...],'train_label_tag':Label_Tag[:L,...]}
    # FilesAddress['testDataset'] = Data[L:,...]
    # FilesAddress['test_label'] = Label[L:,...]
    # FilesAddress['test_label_tag'] = Label_Tag[L:,...]
    # FilesAddress = {'trainDataset':Data,'train_label':Label,'train_label_tag':Label_Tag}
    # FilesAddress['testDataset'] = Data
    # FilesAddress['test_label'] = Label
    # FilesAddress['test_label_tag'] = Label_Tag

    return FilesAddress

imagesDirects = LoadingListOfFiles()

def Batch(List,ind,mode):

    a1 = int(List[ind])
    a2 = int(List[ind+1])

    dataAddress = imagesDirects[mode + '_data'][a1:a2]
    label = imagesDirects[mode + '_label'][a1:a2]

    data = np.zeros((dataAddress.shape[0],HEIGHT2,WIDTH2,3))
    for i in tqdm(range(dataAddress.shape[0])):

        im = cv2.imread(dataAddress[i])
        if downsample != 1:
            im2 = scipy.misc.imresize(arr=im, size=(HEIGHT2,WIDTH2 , 3))
            im = np.asarray(im2)
            # im = rgb2gray(im)
            # im2 = np.zeros((HEIGHT2,WIDTH2,3))
            # for d in range(im.shape[2]):
            #     im2[:,:,d] = cv2.resize(im[:,:,d], dsize=(WIDTH2,HEIGHT2))

        data[i,...] = im[np.newaxis,...]
        # if i == 0:
        #     # data = np.zeros((1,HEIGHT2,WIDTH2,3))
        #     # data[0,...] = im
        #     data = np.zeros((1,HEIGHT2,WIDTH2,3))
        #     data[0,...] = im
        # else:
        #     data = np.concatenate((data,im[np.newaxis,...]), axis=0)

    data = np.array(data/255.0)

    a = datagen.flow(data,label, batch_size=data.shape[0])

    data = a[0][0]
    label = a[0][1]

    return data, label

def plotHistogram(a):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.axis('off')
    histo = plt.subplot(1,2,2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);
    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);
    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);

def BatchesList(mode,batch_size):

    data = imagesDirects[mode + '_data']
    num_TrainData = data.shape[0]

    NumBatches = int(num_TrainData/batch_size)
    List = []
    for ind in range(0,NumBatches+1):
        List = np.append(List,np.array(batch_size)*ind)

    if num_TrainData > batch_size*NumBatches:
        List = np.append(List,np.array(num_TrainData-1))

    return List

BatchesEndPointsTrain = BatchesList('train',int(imagesDirects['train_data'].shape[0]/1))
BatchesEndPointsTest = BatchesList('test',int(imagesDirects['test_data'].shape[0]/1))

def readingDataset(mode,batchSize):

    # transformer = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(0.13, 0.31)])


    # if mode in 'train':
    #     FullIm, FullLb = Batch(BatchesEndPointsTrain,0,mode)
    # else:
    #     FullIm, FullLb = Batch(BatchesEndPointsTest,0,mode)

    with open(directory + mode + '_Data' + input,'rb') as outputDir:
        FullIm = pickle.load(outputDir)


    FullIm_Gray = np.transpose(FullIm,[0,3,1,2])

    # sz = FullIm.shape
    # FullIm_Gray = np.zeros((sz[0],1 , sz[1],sz[2]))
    # for i in range(sz[0]):
    #     im = FullIm[i,...].reshape((sz[1],sz[2],sz[3]))
    #     FullIm_Gray[i,0,:,:] = rgb2gray(im)

    with open(directory + mode + '_Label' + input,'rb') as outputDir:
        FullLb = pickle.load(outputDir)

    a = datagen.flow(FullIm_Gray,FullLb, batch_size=FullIm_Gray.shape[0])

    FullIm_Gray = a[0][0]
    FullLb = a[0][1]



    FullLb_NDArray = mxnet.nd.array(np.asarray(FullLb)) # ,ctx=gpu(0)
    FullIm_NDArray = mxnet.nd.array(FullIm_Gray)# ,ctx=gpu(0)

    BloodCell_dataset = gluon.data.dataset.ArrayDataset(FullIm_NDArray, FullLb_NDArray)
    # BloodCell_dataset = BloodCell_dataset.transform_first(transformer)
    data = gluon.data.DataLoader( BloodCell_dataset , batch_size=batchSize , shuffle=True , num_workers=4)

    return data , FullIm_Gray , FullLb

trainDataset , trainData , trainLabel = readingDataset('train',batch_size)
testDataset , testData , testLabel = readingDataset('test',batch_size)

trainDatasetF , _ , _ = readingDataset('train',batch_size)
testDatasetF , _ , _ = readingDataset('test',batch_size)

# for i, (data, label) in enumerate(train_data2):
#     data = data.as_in_context(ctx)
#     print(data.shape)
#     print(label.shape)
#     break

## --------------------------------------------------------------------------------

def evaluate_accuracy(data_iterator, net):
    acc = mxnet.metric.Accuracy()
    FullPred = []
    FullLabel = []
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        FullPred.append(predictions)
        FullLabel.append(label)

    acc.update(preds=FullPred, labels=FullLabel)

    return acc.get()[1]

# def evaluate_accuracyV2(data,label, net):
#
#     FullLb_NDArray = mxnet.nd.array(np.asarray(label))
#     FullIm_NDArray = mxnet.nd.array(data)
#     BloodCell_dataset = gluon.data.dataset.ArrayDataset(FullIm_NDArray, FullLb_NDArray)
#     data_iterator = gluon.data.DataLoader( BloodCell_dataset , batch_size=int(data.shape[0]/2) , shuffle=True , num_workers=4)
#
#     for i, (data, label) in enumerate(data_iterator):
#         data = data.as_in_context(ctx)
#         label = label.as_in_context(ctx)
#         output = net(data)
#         predictions = nd.argmax(output, axis=1)
#         FullPred.append(predictions)
#         FullLabel.append(label)
#         acc.update(preds=predictions, labels=label)
#     return acc.get()[1]


def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

# trainDataset = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
#                                       batch_size, shuffle=True)
# testDataset = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
#                                      batch_size, shuffle=False)
# for i, (data, label) in enumerate(trainDataset):
#     data = data.as_in_context(ctx)
#     break

num_fc = 128
net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Dropout(0.25))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_fc, activation="relu"))
    net.add(gluon.nn.Dropout(0.5))
    net.add(gluon.nn.Dense(num_outputs))


net.collect_params().initialize(mxnet.init.Xavier(magnitude=2.24), ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

dict_characters = {1:'NEUTROPHIL',2:'EOSINOPHIL',3:'MONOCYTE',4:'LYMPHOCYTE'}
for e in tqdm(range(epochs)):

    # if (e%10 == 0) & (e != 0):
    #     trainDataset , trainData , trainLabel = readingDataset('train',batch_size)


    for i, (data, label) in enumerate(trainDataset):

        # a = datagen.flow(data,label, batch_size=data.shape[0])
        # data = a[0][0]
        # label = a[0][1]

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    train_accuracy = evaluate_accuracy(trainDatasetF, net)
    test_accuracy = evaluate_accuracy(testDatasetF, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

# test_accuracy , test_predictions , test_label = evaluate_accuracy(testDataset, net)
# train_accuracy , train_predictions , train_label = evaluate_accuracy(trainDataset, net)

# train_label = OneHotEncoder(n_values=num_outputs).fit_transform(train_label.reshape(-1, 1)).toarray()
# train_predictions = OneHotEncoder(n_values=num_outputs).fit_transform(train_predictions.reshape(-1, 1)).toarray()
# print('\n', classification_report(train_predictions, train_label, target_names=list(dict_characters.values())), sep='')
# print('\n', classification_report(test_label, test_predictions, target_names=list(dict_characters.values())), sep='')

# print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
