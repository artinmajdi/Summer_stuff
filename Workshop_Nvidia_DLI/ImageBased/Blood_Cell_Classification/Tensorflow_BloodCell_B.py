# from __future__ import print_function
from __future__ import division

import cv2
import os
import pickle
import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
# import sklearn
from sklearn.ensemble import RandomForestClassifier
# from tensorflow.examples.tutorials.mnist import input_data
# import pandas as pd
import tifffile
from sklearn.preprocessing import OneHotEncoder
from skimage.color import rgb2gray
# mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
from skimage.transform import resize
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray


batch_size = 32 # 128
n_classes = 4
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
fcNumNeurons = 128
fcNumNeurons2 = 512
fcNumNeurons3 = 512

GPU_Num = '2'

hm_epochs = 2
RandomForestNumEstimators = 100
downsample = 4

HEIGHT = 240
WIDTH = 320

HEIGHT2 = int(240/downsample)
WIDTH2 = int(320/downsample)

imSz = [HEIGHT2,WIDTH2] # [240,320]


tf.logging.set_verbosity(tf.logging.INFO)
num_classes = 4
Directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'
input = '.pkl' # '_20Per.pkl' #
numSteps = 10000 # 20000
learning_rate = 0.01

def LoadingListOfFiles():

    mode = 'train'
    subClasses = os.listdir(Directory + mode)
    for mode in ['train', 'test']:

        Data = []
        Label = []
        Label_Tag = []
        for i in range(len(subClasses)):
            dirr = Directory + mode + '/' + subClasses[i] + '/'
            files2 = os.listdir(dirr)
            # for f in tqdm(range(int(len(files2)))):
            for f in range(int(len(files2))):
                le = len(files2[f])
                if files2[f][le-4:] == 'jpeg':
                    Data.append(Directory + mode + '/' + subClasses[i] + '/' + files2[f])
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
    # FilesAddress = {'train_data':Data[:L,...],'train_label':Label[:L,...],'train_label_tag':Label_Tag[:L,...]}
    # FilesAddress['test_data'] = Data[L:,...]
    # FilesAddress['test_label'] = Label[L:,...]
    # FilesAddress['test_label_tag'] = Label_Tag[L:,...]
    # FilesAddress = {'train_data':Data,'train_label':Label,'train_label_tag':Label_Tag}
    # FilesAddress['test_data'] = Data
    # FilesAddress['test_label'] = Label
    # FilesAddress['test_label_tag'] = Label_Tag

    return FilesAddress

imagesDirects = LoadingListOfFiles()

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

import numpy as np

a = np.random.random((3,10,16))
b = np.append(a,a,axis=0)


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

    a = datagen.flow(data,label, batch_size=batch_size)

    for i in range(len(a)):
        if i == 0:
            data = a[i][0]
            label = a[i][1]
        else:
            data  = np.append(data , a[i][0] , axis=0)
            label = np.append(label, a[i][1] , axis=0)

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

def readingDataset(mode):

    directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'
    with open(directory + mode + '_Data_20Per.pkl','rb') as outputDir:
        FullIm = pickle.load(outputDir)

    sz = FullIm.shape
    FullIm_Gray = np.zeros((sz[0],imSz[0],imSz[1],1))
    for i in range(sz[0]):
        im = FullIm[i,...].reshape((sz[1],sz[2],sz[3]))
        im = rgb2gray(im)
        FullIm_Gray[i,...,0] = cv2.resize(im, (28, 28))


    with open(directory + mode + '_Label_20Per.pkl','rb') as outputDir:
        FullLb = pickle.load(outputDir)

    FullLb2 = np.zeros((len(FullLb)))
    classes = np.unique(FullLb)
    for cl in range(len(classes)):
        ind = np.where(FullLb == classes[cl])
        FullLb2[ind] = int(cl)

    return FullIm_Gray, FullLb2

BatchesEndPointsTrain = BatchesList('train',int(imagesDirects['train_data'].shape[0]/1))
BatchesEndPointsTest = BatchesList('test',int(imagesDirects['test_data'].shape[0]/1))

# train_data, train_label = Batch(BatchesEndPointsTrain,0,'train')
# eval_data, eval_label = Batch(BatchesEndPointsTest,0,'test')
# BatchesEndPointsTrain = BatchesList('train',imagesDirects['train_data'].shape[0])
# BatchesEndPointsTest = BatchesList('test',imagesDirects['test_data'].shape[0])
# dataA, labelA = Batch(BatchesEndPointsTrain,0,'train')
# plotHistogram(dataA[1])
# imSz = [28,28]
# def readingDataset(mode):
#
#     directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'
#     with open(directory + mode + '_Data_Temp.pkl','rb') as outputDir:
#         FullIm = pickle.load(outputDir)
#
#     sz = FullIm.shape
#     FullIm_Gray = np.zeros((sz[0],imSz[0],imSz[1],1))
#     for i in range(sz[0]):
#         im = FullIm[i,...].reshape((sz[1],sz[2],sz[3]))
#         im = rgb2gray(im)
#         FullIm_Gray[i,...,0] = cv2.resize(im, (28, 28))
#
#
#     with open(directory + mode + '_Label_Temp.pkl','rb') as outputDir:
#         FullLb = pickle.load(outputDir)
#
#     FullLb2 = np.zeros((len(FullLb)))
#     classes = np.unique(FullLb)
#     for cl in range(len(classes)):
#         ind = np.where(FullLb == classes[cl])
#         FullLb2[ind] = int(cl)
#
#     return FullIm_Gray, FullLb2

def cnn_model_fn(features, labels, mode):

    a = int(WIDTH2/2)
    b = int(HEIGHT2/2)

    input_layer = features["x"]

    conv1 = tf.layers.conv2d(
    inputs = input_layer,
    filters = 32,
    kernel_size=[3, 3],
    padding="same",
    activation = tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer)

    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(
    inputs = conv1,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer)

    conv3 = tf.layers.conv2d(
    inputs = conv2,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer)

    pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)
    dropout = tf.layers.dropout(inputs=pool2, rate = 0.25, training = mode == tf.estimator.ModeKeys.TRAIN)

    pool2_flat = tf.reshape(dropout, [-1,a*b*64])

    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer)

    logits = tf.layers.dropout(inputs=dense, rate = 0.5, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=logits, units=num_classes)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions) # , config=tf.contrib.learn.RunConfig(session_config=config)) # , config=config

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op) # , config=tf.contrib.learn.RunConfig(session_config=config)) # , config=config

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    A = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops) # , config=tf.contrib.learn.RunConfig(session_config=config))
    return A

def main(unused_argv):

    # with tf.device('/device:GPU:2'):

    train_data, train_label = Batch(BatchesEndPointsTrain,0,'train')
    eval_data, eval_label = Batch(BatchesEndPointsTest,0,'test')

    # mode = 'train'
    # with open(Directory + mode + '_Data' + input,'rb') as outputDir:
    #     train_data = pickle.load(outputDir)
    #
    # with open(Directory + mode + '_Label' + input,'rb') as outputDir:
    #     train_label = pickle.load(outputDir)
    #
    # mode = 'test'
    # with open(Directory + mode + '_Data' + input,'rb') as outputDir:
    #     eval_data = pickle.load(outputDir)
    #
    # with open(Directory + mode + '_Label' + input,'rb') as outputDir:
    #     eval_label = pickle.load(outputDir)
    # train_data, train_Label = readingDataset('train')
    # eval_data , eval_Label  = readingDataset('test')


    classifier = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir=Directory + 'model/') # , config=tf.contrib.learn.RunConfig(session_config=config)) #, config=config "/tmp/mnist_convnet_model")
    # model_fn = cnn_model_fn, model_dir="/media/groot/Seagate Backup Plus Drive/code/CNN_MNIST")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook( tensors=tensors_to_log, every_n_iter=500 )


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_label,
        batch_size = batch_size,
        num_epochs = None, #hm_epochs,
        shuffle = True)

    classifier.train(
        input_fn = train_input_fn,
        steps = numSteps,
        # hooks = [logging_hook]
        )

    # Evaluate the model and prints results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_label,
        num_epochs=1,
        shuffle = False)

    eval_results = classifier.evaluate(input_fn = eval_input_fn)
    # Y_pred_classes = np.argmax(PredFul,axis=1)
    # Y_true = np.argmax(Yorig,axis=1)
    # print('\n', sklearn.metrics.classification_report( Y_true, Y_pred_classes, target_names=list(dict_characters.values())), sep='')

    print(eval_results)
    print(eval_input_fn)

# if __name__ == "__main__":

config = tf.ConfigProto(log_device_placement=True) # device_count={'GPU':1} ,
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
# myGraph = tf.Graph()
with tf.Session(config=config) as sess:#  , graph=myGraph.as_default())
    sess.run(main(1))
# App = tf.app.run()
