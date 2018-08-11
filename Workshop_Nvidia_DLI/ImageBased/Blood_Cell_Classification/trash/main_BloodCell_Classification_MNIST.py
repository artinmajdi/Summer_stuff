from __future__ import print_function
from __future__ import division

import cv2
import os
import pickle
import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
# from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import tifffile
from sklearn.preprocessing import OneHotEncoder
from skimage.color import rgb2gray
# mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
from skimage.transform import resize
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator


batch_size = 128
n_classes = 4
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
fcNumNeurons = 128
fcNumNeurons2 = 512
fcNumNeurons3 = 512

GPU_Num = '1'

hm_epochs = 5
RandomForestNumEstimators = 100
downsample = 4

HEIGHT = 240
WIDTH = 320

HEIGHT2 = int(240/downsample)
WIDTH2 = int(320/downsample)

Directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'


tf.logging.set_verbosity(tf.logging.INFO)
num_classes = 10
batch_size = 128
imSz = [28,28]

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

        Label = OneHotEncoder(n_values=n_classes).fit_transform(Label.reshape(-1, 1)).toarray()

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

def Batch(List,ind,mode):

    a1 = int(List[ind])
    a2 = int(List[ind+1])

    dataAddress = imagesDirects[mode + '_data'][a1:a2]
    label = imagesDirects[mode + '_label'][a1:a2]

    for i in range(dataAddress.shape[0]):

        im = cv2.imread(dataAddress[i])
        if downsample != 1:
            im2 = scipy.misc.imresize(arr=im, size=(HEIGHT2,WIDTH2 , 3))
            im = np.asarray(im2)
            # im2 = np.zeros((HEIGHT2,WIDTH2,3))
            # for d in range(im.shape[2]):
            #     im2[:,:,d] = cv2.resize(im[:,:,d], dsize=(WIDTH2,HEIGHT2))


        if i == 0:
            data = np.zeros((1,HEIGHT2,WIDTH2,3))
            data[0,...] = im
        else:
            data = np.concatenate((data,im[np.newaxis,...]), axis=0)

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

BatchesEndPointsTrain = BatchesList('train',batch_size)
BatchesEndPointsTest = BatchesList('test',batch_size)

dataA, labelA = Batch(BatchesEndPointsTrain,0,'train')

plotHistogram(dataA[1])


def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, imSz[0], imSz[1], 1])

    conv1 = tf.layers.conv2d(
    inputs = input_layer,
    filters = 32,
    kernel_size=[5, 5],
    padding="same",
    activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(
    inputs = pool1,
    filters=64,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1,int(imSz[0]/4)*int(imSz[1]/4)*64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions) # , config=tf.contrib.learn.RunConfig(session_config=config)) # , config=config

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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

    # with tf.device('/device:GPU:0'):
    # load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
    model_fn = cnn_model_fn, model_dir="/media/groot/Seagate Backup Plus Drive/code/CNN_MNIST") # , config=tf.contrib.learn.RunConfig(session_config=config)) #, config=config "/tmp/mnist_convnet_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = batch_size,
        num_epochs = None,
        shuffle = True)

    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000,
        # hooks = [logging_hook]
        )

    # Evaluate the model and prints results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs=1,
        shuffle = False)

    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)


config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

#sess = tf.Session() # config=config) #  , graph=myGraph.as_default())
#sess.run(main(1))
App = tf.app.run()
