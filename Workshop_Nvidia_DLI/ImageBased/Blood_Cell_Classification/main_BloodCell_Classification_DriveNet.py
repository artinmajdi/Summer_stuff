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

HEIGHT2 = int(240/downsample) # 60
WIDTH2 = int(320/downsample) # 80

Directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'


x = tf.placeholder('float',[None,HEIGHT2,WIDTH2,3])
y = tf.placeholder('float')


# def load_image(path):
#     im = tifffile.imread(path)
#     im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
#     # im = np.reshape(im,[-1,WIDTH*HEIGHT],order='F')
#     return im

# def ConvertingData(train_data):
#
#     sze = train_data.shape
#     train_data2 = np.zeros([sze[0],HEIGHT,WIDTH])
#
#     for i in range(sze[0]):
#         a = train_data[i, :]
#         a = np.reshape(a, [HEIGHT,WIDTH], 'F')
#         train_data2[i, :,:] = a
#
#     return train_data2

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



def CNN_testAccuracy(A):
    cnnAccuracy = []
    for ind in range(len(BatchesEndPointsTest)-1):
        batch_x, batch_y = Batch(BatchesEndPointsTest, ind, 'test')

        a = int(BatchesEndPointsTest[ind])
        b = int(BatchesEndPointsTest[ind+1])
        cnnAccuracy.append(A.eval({x:batch_x , y:batch_y}))

    cnnAccuracy = np.mean(np.asarray(cnnAccuracy))
    print('CNN Accuracy:',cnnAccuracy)

    return cnnAccuracy

def conv3d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME') # the strides is because the conv will take pixel by pixel and move
    # return tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding = 'SAME') # the strides is because the conv will take pixel by pixel and move

def maxpool3d(x):
    return tf.nn.max_pool2d(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # padding will take 2*2
    # return tf.nn.max_pool3d(x,ksize=[1,2,2,3,1],strides=[1,2,2,1,1],padding='SAME') # padding will take 2*2

def convolutional_network_network(x):
    a = int(WIDTH2/2)
    b = int(HEIGHT2/2)

    # N_num = [5, 5 , 32 , 64]
    N_num = [3, 3 , 32 , 64 , 10]
    weights = {'W_conv1':tf.Variable(tf.random_normal([N_num[0],N_num[1],3,1,N_num[2]])),  # 5*5 convolution 1 input and will produce 32 features/output
               'W_conv2':tf.Variable(tf.random_normal([N_num[0],N_num[1],3,N_num[2],N_num[3]])), # 5*5 convolution 32 input 64 output
               #'W_conv3':tf.Variable(tf.random_normal([N_num[0],N_num[1],3,N_num[3],N_num[4]])),
               'W_fc':tf.Variable(tf.random_normal([a*b*3*N_num[3],fcNumNeurons])),  # we started
               #'W_fc2':tf.Variable(tf.random_normal([fcNumNeurons,fcNumNeurons2])),
               #'W_fc3':tf.Variable(tf.random_normal([fcNumNeurons2,fcNumNeurons3])),
               #  by WIDTH*HEIGHT image but now it converted to a feature map of a*b size, 64 is the number of features came from earlier
               # layer and also 1024 is the number of nodes  so the input is a*b*64 and the output is 1024
               'out':tf.Variable(tf.random_normal([fcNumNeurons,n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([N_num[2]])),
              'b_conv2':tf.Variable(tf.random_normal([N_num[3]])),
              #'b_conv3':tf.Variable(tf.random_normal([N_num[4]])),
              'b_fc':tf.Variable(tf.random_normal([fcNumNeurons])),
              #'b_fc2':tf.Variable(tf.random_normal([fcNumNeurons2])),
              #'b_fc3':tf.Variable(tf.random_normal([fcNumNeurons3])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x,shape=[-1,HEIGHT2,WIDTH2,3,1])  # we are reshaping a flat WIDTH*HEIGHT vector to a a*b image
    ##############x = np.resize(x,[a,b])
    conv1 = conv3d(x,weights['W_conv1']) + biases['b_conv1']
    # conv1 = tf.nn.max_pool3d(conv1,ksize=[1,2,2,1,1],strides=[1,2,2,1,1],padding='SAME')
    conv1 = tf.nn.relu(conv1)
    # conv1 = maxpool3d(conv1)

    conv2 = conv3d(conv1,weights['W_conv2']) + biases['b_conv2']
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool3d(conv2,ksize=[1,2,2,1,1],strides=[1,2,2,1,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2,0.75)
    # conv2 = maxpool3d(conv2)

    #conv3 = conv3d(conv2,weights['W_conv3']) + biases['b_conv3']
    #conv3 = tf.nn.max_pool3d(conv3,ksize=[1,2,2,1,1],strides=[1,2,2,1,1],padding='SAME')

    fc = tf.reshape(conv2,[-1,b*a*3*N_num[3]])

    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc,0.5)

    # fc2 = tf.nn.relu(tf.matmul(fc,weights['W_fc2'])+biases['b_fc2'])
    # fc2 = tf.nn.dropout(fc2,keep_rate)
    # fc3 = tf.nn.relu(tf.matmul(fc2,weights['W_fc3']) + biases['b_fc3'])

    Output = tf.matmul(fc,weights['out']) + biases['out']
    Output = tf.nn.softmax(Output)

    return Output,fc

def train_neural_network(x):
    prediction,fc = convolutional_network_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    # predicted_label = tf.nn.softmax(prediction)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()
    # saverFc = tf.train.Saver({"fc": fc})

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
    # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for ind in range(len(BatchesEndPointsTrain)-1):

                batch_x, batch_y = Batch(BatchesEndPointsTrain, ind, 'train')
                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x , y: batch_y})
                epoch_loss += c
            print('Epoch',epoch+1,'completed out of',hm_epochs,'loss:',epoch_loss)

        save_path = saver.save(sess, Directory+'model' + '/model.ckpt')
        print("Model saved in file: %s" % save_path)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) # !! predicted_label
        A = tf.cast(correct,'float')
        accuracy = tf.reduce_mean(A)
        # Accuracy = accuracy.eval({x:TestData , y:TestLabel})

        # cnnAccuracy = CNN_testAccuracy(A)
        # # span = BatchesList(TestData.shape[0],1000)
        # cnnAccuracy = np.zeros((TestData.shape[0]))
        cnnAccuracy = []
        for ind in range(len(BatchesEndPointsTest)-1):
            batch_x, batch_y = Batch(BatchesEndPointsTest, ind, 'test')

            a = int(BatchesEndPointsTest[ind])
            b = int(BatchesEndPointsTest[ind+1])
            cnnAccuracy.append(A.eval({x:batch_x , y:batch_y}))

        cnnAccuracy = np.mean(np.asarray(cnnAccuracy))
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
            batch_x, batch_y = Batch(BatchesEndPointsTrain, ind, 'train')
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
            batch_x, batch_y = Batch(BatchesEndPointsTest, ind, 'test')
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

        # span = BatchesList(TestData.shape[0],1000)

        # cnnAccuracy = CNN_testAccuracy(A)
        cnnAccuracy = []
        for ind in range(len(BatchesEndPointsTest)-1):
            batch_x, batch_y = Batch(BatchesEndPointsTest, ind, 'test')

            a = int(BatchesEndPointsTest[ind])
            b = int(BatchesEndPointsTest[ind+1])
            cnnAccuracy.append(A.eval({x:batch_x , y:batch_y}))

        cnnAccuracy = np.mean(np.asarray(cnnAccuracy))
        print('CNN Accuracy:',cnnAccuracy)

    return fcTrainData, fcTrainLabel , fcTestData, fcTestLabel


dict_characters = {1:'EOSINOPHIL',2:'LYMPHOCYTE',3:'MONOCYTE',4:'NEUTROPHIL'}

def train_neural_networkANDpreLastLayereSaver(x):

    prediction,fc = convolutional_network_network(x)
    # prediction = tf.nn.softmax(prediction2)

    # this part in the video is (prediction,y)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = prediction) )
    # cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = prediction) )   # newly added
    # predicted_label = tf.nn.softmax(prediction)
    # predicted_label = prediction  # newly added


    # correct = tf.equal(tf.argmax(predicted_label,1),tf.argmax(y,1))
    # accuracyM = tf.cast(correct,'float')
    # accuracy = tf.reduce_mean(A)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()
    # saverFc = tf.train.Saver({"fc": fc})
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list=GPU_Num
    with tf.Session(config=config) as sess:
    # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(hm_epochs)):
            epoch_loss = 0
            # for ind in range(int(TrainData.shape[0]/batch_size)):
            accuracyM = []
            for ind in range(len(BatchesEndPointsTrain)-1):
                batch_x, batch_y = Batch(BatchesEndPointsTrain, ind, 'train')

                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x , y: batch_y})
                epoch_loss += c
                # accuracyM.append(accuracy)
            print('Epoch',epoch+1,'completed out of',hm_epochs,'loss:',epoch_loss)

        save_path = saver.save(sess, Directory + 'model/model.ckpt')
        # saver.restore(sess, Directory + "model/model.ckpt")
        print("Model saved in file: %s" % save_path)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

        B = tf.cast(prediction,'float')
        A = tf.cast(correct,'float')
        accuracy = tf.reduce_mean(A)
        # CNN_Accuracy = accuracy.eval({x:TestData , y:TestLabel})
        cnnAccuracy = []
        for ind in range(len(BatchesEndPointsTest)-1):
            batch_x, batch_y = Batch(BatchesEndPointsTest, ind, 'test')

            a = int(BatchesEndPointsTest[ind])
            b = int(BatchesEndPointsTest[ind+1])
            k = A.eval({x:batch_x , y:batch_y})
            pred = B.eval({x:batch_x})
            #a = np.sum(batch_y,axis=0)
            #b = np.sum(pred,axis=0)

            #dif = batch_y + -2*pred
            #print(dif)
            # print('label: ',a,' predic: ',b)
            if ind == 0:
                cnnAccuracy = k
                PredFul = pred
                Yorig = batch_y
            else:
                cnnAccuracy = np.concatenate((cnnAccuracy,k) ,axis=0)
                Yorig = np.concatenate((Yorig,batch_y) ,axis=0)
                PredFul = np.concatenate((PredFul,pred) ,axis=0)


        Y_pred_classes = np.argmax(PredFul,axis=1)
        Y_true = np.argmax(Yorig,axis=1)
        print('\n', sklearn.metrics.classification_report( Y_true, Y_pred_classes, target_names=list(dict_characters.values())), sep='')

        cnnAccuracyM = np.mean(cnnAccuracy)
        print('CNN Accuracy:',cnnAccuracyM)


        ##### ---------------------------- saving FC layer Train Data --------------------------------------
        fcTrainData  = np.zeros((imagesDirects['train_data'].shape[0],fcNumNeurons))
        fcTrainLabel = np.zeros((imagesDirects['train_data'].shape[0],n_classes))
        # for ind in range(int(TrainData.shape[0]/batch_size)):
        for ind in range(len(BatchesEndPointsTrain)-1):
            batch_x, batch_y = Batch(BatchesEndPointsTrain, ind, 'train')
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
        fcTestData  = np.zeros((imagesDirects['test_data'].shape[0],fcNumNeurons))
        fcTestLabel = np.zeros((imagesDirects['test_data'].shape[0],n_classes))

        for ind in range(len(BatchesEndPointsTest)-1):
            batch_x, batch_y = Batch(BatchesEndPointsTest, ind, 'test')
            preLastLayer = sess.run([fc],feed_dict = {x: batch_x})

            L_end = min((ind+1)*batch_size , imagesDirects['test_data'].shape[0])

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
    return cnnAccuracyM, fcTrainData, fcTrainLabel , fcTestData, fcTestLabel

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
