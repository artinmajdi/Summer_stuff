from __future__ import print_function
from __future__ import division

import cv2
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import tifffile
from sklearn.preprocessing import OneHotEncoder
from skimage.color import rgb2gray
# mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
from skimage.transform import resize
batch_size = 100
n_classes = 4
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
fcNumNeurons = 1024
fcNumNeurons2 = 1024
fcNumNeurons3 = 512


traintSize = 0.7
hm_epochs = 10
RandomForestNumEstimators = 100
downsample = 2

HEIGHT = 240
WIDTH = 320

HEIGHT2 = int(240/downsample)
WIDTH2 = int(320/downsample)

Directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'


x = tf.placeholder('float',[None,HEIGHT2,WIDTH2,3])
y = tf.placeholder('float')

# Directory+'model' = '/home/artin/artin'
# Directory = '/media/data1/artin/code/Cell/data/CiliaImages/'

def load_image(path):
    im = tifffile.imread(path)
    im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
    # im = np.reshape(im,[-1,WIDTH*HEIGHT],order='F')
    return im

def ConvertingData(train_data):

    sze = train_data.shape
    train_data2 = np.zeros([sze[0],HEIGHT,WIDTH])

    for i in range(sze[0]):
        a = train_data[i, :]
        a = np.reshape(a, [HEIGHT,WIDTH], 'F')
        train_data2[i, :,:] = a

    return train_data2

def LoadingListOfFiles():

    mode = 'train'
    subClasses = os.listdir(Directory + mode)
    for mode in ['train', 'test']:

        Data = []
        Label = []
        for i in range(len(subClasses)):
            dirr = Directory + mode + '/' + subClasses[i] + '/'
            files2 = os.listdir(dirr)
            for f in range(len(files2)):
                le = len(files2[f])
                if files2[f][le-4:] == 'jpeg':
                    Data.append(Directory + mode + '/' + subClasses[i] + '/' + files2[f])
                    Label.append(subClasses[i])

        Data = np.asarray(Data)
        Label = np.asarray(Label)

        indexesRandomized = np.random.permutation(Data.shape[0])

        Data = Data[indexesRandomized]
        Label = Label[indexesRandomized]

        if mode == 'train':
            FilesAddress = {'train_data':Data,'train_label':Label}
        else:
            FilesAddress['test_data'] = Data
            FilesAddress['test_label'] = Label

    return FilesAddress

imagesDirects = LoadingListOfFiles()

def ReadingByBatch(mode):


    for j in range(len(files)): # ): len(files)):

        im = cv2.imread(dirr + files[j])
        if (j == 0): # & (i == 0)
            sz = im.shape
            Full_Image = np.zeros((1,sz[0],sz[1],sz[2]))
            Full_Image[0,...] = im
            Full_Label = subClasses[i]
        else:
            Full_Image = np.concatenate((Full_Image,im[np.newaxis,...]), axis=0)
            Full_Label = np.append(Full_Label,subClasses[i])

def ReadingData():

    with open(Directory + 'train_Data_Temp.pkl','rb') as output:
        X_train2 = pickle.load(output)

    with open(Directory + 'test_Data_Temp.pkl','rb') as output:
        X_test2 = pickle.load(output)

    sz = X_train2.shape
    X_train = np.zeros((sz[0],HEIGHT2 , WIDTH2,3))
    for i in range(sz[0]):
        im = X_train2[i,...].reshape((sz[1],sz[2],3))
        # im = rgb2gray(im)
        if downsample != 1:
            im2 = np.zeros((HEIGHT2,WIDTH2,3))
            for d in range(im.shape[2]):
                im2[:,:,d] = cv2.resize(im[:,:,d], dsize=(WIDTH2,HEIGHT2))
            # im2 = resize(im,(1/downsample))
            X_train[i,...] = im2
        else:
            X_train[i,...] = im

    sz = X_test2.shape
    X_test = np.zeros((sz[0],HEIGHT2 , WIDTH2,3))
    for i in range(sz[0]):
        im = X_test2[i,...].reshape((sz[1],sz[2],3))
        # im = rgb2gray(im)
        if downsample != 1:
            im2 = np.zeros((HEIGHT2,WIDTH2,3))
            for d in range(im.shape[2]):
                im2[:,:,d] = cv2.resize(im[:,:,d], dsize=(WIDTH2,HEIGHT2))

            X_test[i,...] = im2
        else:
            X_test[i,...] = im


    with open(Directory + 'train_Label_Temp.pkl','rb') as output:
        Y_train2 = pickle.load(output)

    with open(Directory + 'test_Label_Temp.pkl','rb') as output:
        Y_test2 = pickle.load(output)

    Y_train3 = np.zeros((len(Y_train2),1))
    classes = np.unique(Y_train2)
    for cl in range(len(classes)):
        ind = np.where(Y_train2 == classes[cl])
        Y_train3[ind] = int(cl)

    Y_test3 = np.zeros((len(Y_test2),1))
    classes = np.unique(Y_test2)
    for cl in range(len(classes)):
        ind = np.where(Y_test2 == classes[cl])
        Y_test3[ind] = int(cl)


    Y_train = OneHotEncoder(n_values=n_classes).fit_transform(Y_train3.reshape(-1, 1)).toarray()
    Y_test = OneHotEncoder(n_values=n_classes).fit_transform(Y_test3.reshape(-1, 1)).toarray()

    TestIndexes = np.random.permutation(X_train.shape[0])

    X_train = X_train[TestIndexes , ...]
    Y_train = Y_train[TestIndexes , ...]


    return X_train, Y_train, X_test, Y_test

def BatchesList(num_TrainData,batch_size):

    NumBatches = int(num_TrainData/batch_size)
    List = []
    for ind in range(0,NumBatches+1):
        List = np.append(List,np.array(batch_size)*ind)

    if num_TrainData > batch_size*NumBatches:
        List = np.append(List,np.array(num_TrainData-1))

    return List

def Batch(Data,Label,List,ind):

    a1 = int(List[ind])
    a2 = int(List[ind+1])
    data   = Data[a1:a2,:,:]
    labels = Label[a1:a2,:]

    return data, labels


TrainData, TrainLabel, TestData, TestLabel = ReadingData()

mode = 'Training' # 'NotTraining'
if mode == 'Training':
    TrainData, TrainLabel, TestData, TestLabel = ReadingData()

    # TrainData = mnist.train.images
    # TrainLabel = mnist.train.labels
    # TrainData = ConvertingData(TrainData)
    #
    # TestData = mnist.test.images
    # TestLabel = mnist.test.labels
    # TestData = ConvertingData(TestData)

    num_TrainData = TrainData.shape[0]
    BatchesEndPointsTrain = BatchesList(num_TrainData,batch_size)
    BatchesEndPointsTest = BatchesList(TestData.shape[0],batch_size)

batch_x, batch_y = Batch(TrainData, TrainLabel, BatchesEndPointsTrain, 1)

# batch_x.shape

def conv3d(x,W):
    return tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding = 'SAME') # the strides is because the conv will take pixel by pixel and move

def maxpool3d(x):
    #                       size of window      movement of window
    return tf.nn.max_pool3d(x,ksize=[1,2,2,3,1],strides=[1,2,2,1,1],padding='SAME') # padding will take 2*2
    # window each step and move 2 pixel at a time to not have any ovelapping

def convolutional_network_network(x):
    a = int(WIDTH2/4)
    b = int(HEIGHT2/4)

    # N_num = [5, 5 , 32 , 64]
    N_num = [5, 5 , 20 , 15 , 10]
    weights = {'W_conv1':tf.Variable(tf.random_normal([N_num[0],N_num[1],3,1,N_num[2]])),  # 5*5 convolution 1 input and will produce 32 features/output
               'W_conv2':tf.Variable(tf.random_normal([N_num[0],N_num[1],3,N_num[2],N_num[3]])), # 5*5 convolution 32 input 64 output
               #'W_conv3':tf.Variable(tf.random_normal([N_num[0],N_num[1],3,N_num[3],N_num[4]])),
               'W_fc':tf.Variable(tf.random_normal([a*b*3*N_num[3],fcNumNeurons])),  # we started
               'W_fc2':tf.Variable(tf.random_normal([fcNumNeurons,fcNumNeurons2])),
               'W_fc3':tf.Variable(tf.random_normal([fcNumNeurons2,fcNumNeurons3])),
               #  by WIDTH*HEIGHT image but now it converted to a feature map of a*b size, 64 is the number of features came from earlier
               # layer and also 1024 is the number of nodes  so the input is a*b*64 and the output is 1024
               'out':tf.Variable(tf.random_normal([fcNumNeurons3,n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([N_num[2]])),
              'b_conv2':tf.Variable(tf.random_normal([N_num[3]])),
              #'b_conv3':tf.Variable(tf.random_normal([N_num[4]])),
              'b_fc':tf.Variable(tf.random_normal([fcNumNeurons])),
              'b_fc2':tf.Variable(tf.random_normal([fcNumNeurons2])),
              'b_fc3':tf.Variable(tf.random_normal([fcNumNeurons3])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x,shape=[-1,HEIGHT2,WIDTH2,3,1])  # we are reshaping a flat WIDTH*HEIGHT vector to a a*b image
    ##############x = np.resize(x,[a,b])
    conv1 = conv3d(x,weights['W_conv1']) + biases['b_conv1']
    conv1 = maxpool3d(conv1)

    conv2 = conv3d(conv1,weights['W_conv2']) + biases['b_conv2']
    conv2 = maxpool3d(conv2)

    #conv3 = conv3d(conv2,weights['W_conv3'])
    #conv3 = maxpool3d(conv3)

    fc = tf.reshape(conv2,[-1,b*a*3*N_num[3]])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc,keep_rate)

    fc2 = tf.nn.relu(tf.matmul(fc,weights['W_fc2'])+biases['b_fc2'])
    fc3 = tf.nn.relu(tf.matmul(fc2,weights['W_fc3'])+biases['b_fc3'])

    Output = tf.matmul(fc3,weights['out']) + biases['out']

    return Output,fc

def train_neural_network(x):
    prediction,fc = convolutional_network_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    predicted_label = tf.nn.softmax(prediction)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()
    # saverFc = tf.train.Saver({"fc": fc})

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
    # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for ind in range(len(BatchesEndPointsTrain)-1):
                batch_x, batch_y = Batch(TrainData, TrainLabel, BatchesEndPointsTrain, ind)
                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x , y: batch_y})
                epoch_loss += c
            print('Epoch',epoch+1,'completed out of',hm_epochs,'loss:',epoch_loss)

        save_path = saver.save(sess, Directory+'model' + '/model.ckpt')
        print("Model saved in file: %s" % save_path)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        A = tf.cast(correct,'float')
        accuracy = tf.reduce_mean(A)
        # Accuracy = accuracy.eval({x:TestData , y:TestLabel})

        span = BatchesList(TestData.shape[0],1000)
        cnnAccuracy = np.zeros((TestData.shape[0]))
        for ind in range(len(span)-1):
            batch_x, batch_y = Batch(TestData, TestLabel, span, ind)

            a = int(span[ind])
            b = int(span[ind+1])
            cnnAccuracy[a:b]  = A.eval({x:batch_x , y:batch_y})

        cnnAccuracy = np.mean(cnnAccuracy)
        print('CNN Accuracy:',cnnAccuracy)

    return Accuracy

def preLastLayereSaver(x):

    prediction,fc = convolutional_network_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    # optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(Directory+'/model.ckpt.meta')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, Directory+'model' + '/model.ckpt')

        ##### ---------------------------- saving FC layer Train Data --------------------------------------
        fcTrainData  = np.zeros((TrainData.shape[0],fcNumNeurons))
        fcTrainLabel = np.zeros((TrainLabel.shape[0],n_classes))
        for ind in range(len(BatchesEndPointsTrain)-1):
            batch_x, batch_y = Batch(TrainData, TrainLabel, BatchesEndPointsTrain, ind)
            preLastLayer = sess.run([fc],feed_dict = {x: batch_x})

            a = int(BatchesEndPointsTrain[ind])
            b = int(BatchesEndPointsTrain[ind+1])
            fcTrainData[a:b,:]  = preLastLayer[0]
            fcTrainLabel[a:b,:] = batch_y

        np.savez(Directory+'model' + '/fcTrainData',tuple(fcTrainData))
        np.savez(Directory+'model' + '/fcTrainLabel', tuple(fcTrainLabel))

        ##### ---------------------------- saving FC layer Test Data --------------------------------------
        fcTestData  = np.zeros((TestData.shape[0],fcNumNeurons))
        fcTestLabel = np.zeros((TestLabel.shape[0],n_classes))
        for ind in range(len(BatchesEndPointsTest)-1):
            batch_x, batch_y = Batch( TestData, TestLabel, BatchesEndPointsTest, ind)
            preLastLayer = sess.run([fc],feed_dict = {x: batch_x})

            a = int(BatchesEndPointsTest[ind])
            b = int(BatchesEndPointsTest[ind+1])
            fcTestData[a:b,:]  = preLastLayer[0]
            fcTestLabel[a:b,:] = batch_y

        np.savez(Directory+'model' + '/fcTestData',tuple(fcTestData))
        np.savez(Directory+'model' + '/fcTestLabel', tuple(fcTestLabel))

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        A = tf.cast(correct,'float')
        accuracy = tf.reduce_mean(A)
        # Accuracy = accuracy.eval({x:TestData , y:TestLabel})
        # print('CNN Accuracy:',Accuracy)

        span = BatchesList(TestData.shape[0],1000)
        Acc = np.zeros((1,len(span)))
        cnnAccuracy = np.zeros((TestData.shape[0]))
        for ind in range(len(span)-1):
            batch_x, batch_y = Batch(TestData, TestLabel, span, ind)

            a = int(span[ind])
            b = int(span[ind+1])
            cnnAccuracy[a:b]  = A.eval({x:batch_x , y:batch_y})

        cnnAccuracy = np.mean(cnnAccuracy)
        print('CNN Accuracy:',cnnAccuracy)


    return fcTrainData, fcTrainLabel , fcTestData, fcTestLabel

###########################################################################

def train_neural_networkANDpreLastLayereSaver(x):

    prediction,fc = convolutional_network_network(x)
    # this part in the video is (prediction,y)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    predicted_label = tf.nn.softmax(prediction)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()
    # saverFc = tf.train.Saver({"fc": fc})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # for ind in range(int(TrainData.shape[0]/batch_size)):
            for ind in range(len(BatchesEndPointsTrain)-1):
                batch_x, batch_y = Batch(TrainData, TrainLabel, BatchesEndPointsTrain, ind)
                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x , y: batch_y})
                epoch_loss += c
            print('Epoch',epoch+1,'completed out of',hm_epochs,'loss:',epoch_loss)

        save_path = saver.save(sess, Directory + 'model/model.ckpt')
        print("Model saved in file: %s" % save_path)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        A = tf.cast(correct,'float')
        accuracy = tf.reduce_mean(A)
        # CNN_Accuracy = accuracy.eval({x:TestData , y:TestLabel})

        span = BatchesList(TestData.shape[0],1000)
        Acc = np.zeros((1,len(span)))
        cnnAccuracy = np.zeros((TestData.shape[0]))
        for ind in range(len(span)-1):
            batch_x, batch_y = Batch(TestData, TestLabel, span, ind)

            a = int(span[ind])
            b = int(span[ind+1])
            cnnAccuracy[a:b]  = A.eval({x:batch_x , y:batch_y})

        cnnAccuracy = np.mean(cnnAccuracy)
        print('CNN Accuracy:',cnnAccuracy)


        ##### ---------------------------- saving FC layer Train Data --------------------------------------
        fcTrainData  = np.zeros((TrainData.shape[0],fcNumNeurons))
        fcTrainLabel = np.zeros((TrainLabel.shape[0],n_classes))
        # for ind in range(int(TrainData.shape[0]/batch_size)):
        for ind in range(len(BatchesEndPointsTrain)-1):
            batch_x, batch_y = Batch(TrainData, TrainLabel, BatchesEndPointsTrain, ind)
            preLastLayer = sess.run([fc],feed_dict = {x: batch_x})

            a = int(BatchesEndPointsTrain[ind])
            b = int(BatchesEndPointsTrain[ind+1])
            fcTrainData[a:b,:]  = preLastLayer[0]
            fcTrainLabel[a:b,:] = batch_y

            # fcTrainData.append(preLastLayer[0])
            # fcTrainLabel.append(batch_y)
        np.savez(Directory + 'model/fcTrainData',tuple(fcTrainData))
        np.savez(Directory + 'model/fcTrainLabel', tuple(fcTrainLabel))

        ##### ---------------------------- saving FC layer Test Data --------------------------------------
        fcTestData  = np.zeros((TestData.shape[0],fcNumNeurons))
        fcTestLabel = np.zeros((TestLabel.shape[0],n_classes))

        for ind in range(len(BatchesEndPointsTest)-1):
            batch_x, batch_y = Batch( TestData, TestLabel, BatchesEndPointsTest, ind)
            preLastLayer = sess.run([fc],feed_dict = {x: batch_x})

            L_end = min((ind+1)*batch_size , TestData.shape[0])

            a = int(BatchesEndPointsTest[ind])
            b = int(BatchesEndPointsTest[ind+1])
            fcTestData[a:b,:]  = preLastLayer[0]
            fcTestLabel[a:b,:] = batch_y

            # fcTestData.append(preLastLayer[0])
            # fcTestLabel.append(batch_y)
        np.savez(Directory+'model' + '/fcTestData',tuple(fcTestData))
        np.savez(Directory+'model' + '/fcTestLabel', tuple(fcTestLabel))

        # for i in range(TestLabel.shape[0]):
        #     p_label = sess.run([predicted_label], feed_dict = {x: [TestData[i]]})
        #     print(i, 'actual label:', np.where(TestLabel[i]>0)[0][0], 'predicted label:', np.where(p_label[0] > 0))
    return cnnAccuracy, fcTrainData, fcTrainLabel , fcTestData, fcTestLabel

def loadFClayer():
    TrainData = np.load(Directory+'model'+'/fcTrainData.npz')
    TrainLabel = np.load(Directory+'model'+'/fcTrainLabel.npz')

    TestData  = np.load(Directory+'model'+'/fcTestData.npz')
    TestLabel = np.load(Directory+'model'+'/fcTestLabel.npz')

    return TrainData['arr_0'],TrainLabel['arr_0'],TestData['arr_0'],TestLabel['arr_0']

def ReshapeData(Data,Label):
    # print(data)
    # print(data['arr_0'].shape)
    # print(label['arr_0'].shape)
    Data = np.swapaxes(Data['arr_0'],1,0)
    Data = np.reshape(Data,[Data.shape[0]*Data.shape[1],Data.shape[2]],'F')

    Label = np.swapaxes(Label['arr_0'],1,0)
    Label = np.reshape(Label,[Label.shape[0]*Label.shape[1],Label.shape[2]],'F')

    return Data , Label

def RandomForestClassifier_func(fcTrainData, fcTrainLabel , fcTestData, fcTestLabel,NumberOfEstimators):
    Classifier = RandomForestClassifier(n_estimators = NumberOfEstimators,random_state=True)

    fcTrainLabel2 = np.zeros((fcTrainLabel.shape[0],1),dtype=int)
    index = 0
    for i in range(fcTrainLabel.shape[0]):
        a = np.where(fcTrainLabel[i,:] == 1)[0]
        if len(a) != 0:
            fcTrainLabel2[i] = a
        else:
            index = index + 1
    print('train index',index)

    fcTrainLabel2 = np.ravel(fcTrainLabel2)

    fcTestLabel2 = np.zeros((fcTestLabel.shape[0],1),dtype=int)
    index = 0
    for i in range(fcTestLabel.shape[0]):
        a = np.where(fcTestLabel[i,:] == 1)[0]
        if len(a) != 0:
            fcTestLabel2[i] = a
        else:
            index = index + 1
    print('test index',index)

    fcTestLabel2 = np.ravel(fcTestLabel2)

    Classifier.fit(fcTrainData,fcTrainLabel2)
    pred = Classifier.predict(fcTestData)
    Accuracy = np.average(pred == fcTestLabel2)
    print('Accuracy:',Accuracy)

    # print(TestLabel2[np.where(pred != TestLabel2)[0]])
    # pd.crosstab(TestLabel2, pred, rownames=['actual'], colnames=['preds'])

    return Accuracy

###########################################################################

# if mode == 'NotTraining':

state = 'FullModel'
if state == 'FullModel':
    CNN_Accuracy, fcTrainData, fcTrainLabel , fcTestData, fcTestLabel = train_neural_networkANDpreLastLayereSaver(x)

    Accuracy = RandomForestClassifier_func(fcTrainData, fcTrainLabel, fcTestData, fcTestLabel , RandomForestNumEstimators)

elif state == 'separetelySaveModelAndThenRunRandomForest':
    fcTrainData, fcTrainLabel , fcTestData, fcTestLabel = loadFClayer()

    Accuracy = RandomForestClassifier_func(fcTrainData, fcTrainLabel, fcTestData, fcTestLabel , RandomForestNumEstimators)

elif state == 'saveCNNmodel':
    Accuracy = train_neural_network(x)

elif state == 'LoadCNNmodelANDsaveFC':
    fcTrainData, fcTrainLabel , fcTestData, fcTestLabel = preLastLayereSaver(x)

    Accuracy = RandomForestClassifier_func(fcTrainData, fcTrainLabel, fcTestData, fcTestLabel , RandomForestNumEstimators)

elif state == 'LoadFC':
    fcTrainData, fcTrainLabel , fcTestData, fcTestLabel = loadFClayer()

    print(fcTrainData.shape)
    print(fcTrainLabel.shape)
    print(fcTestData.shape)
    print(fcTestLabel.shape)
