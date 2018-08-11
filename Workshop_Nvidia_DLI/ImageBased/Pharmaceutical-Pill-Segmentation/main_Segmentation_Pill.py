import matplotlib.pyplot as plt
from tf_unet import unet, image_util
import logging
import numpy as np
from skimage import filters
import os
from PIL import Image
import imageio

import SimpleITK


dir = '/media/groot/Seagate Backup Plus Drive/dataset/new/Pharmaceutical-Pill-Image-Dataset/1K-Pill-Image-DataSet/data/'

try:
    os.makedirs(dir+'Results/')
except:
    print('-----------')

gpuNum = '2'

padSize1 = 40
TestSizeSpan = 30
padSize = int(padSize1/2)
epoch_Num = 100


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
net = unet.Unet(layers=3, features_root=16, channels=3, n_class=2 , summaries=True) # , cost="dice_coefficient"


def DiceCoefficientCalculator(msk1,msk2):

    intersection = msk1 * msk2  # np.logical_and(msk1,msk2)
    DiceCoef = ( intersection.sum() * 2 ) / (msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

def Training(net):

    TrainData = image_util.ImageDataProvider(dir + "*.tif",shuffle_data=True)
    L = int(len(TrainData.data_files)/10)
    trainer = unet.Trainer(net, optimizer = "adam")
    path = trainer.train(TrainData, dir+'model', training_iters = L, epochs=epoch_Num, display_step=100 , GPU_Num=gpuNum) # , GPU_Num=gpuNum
    return path

def Testing(path , net):

    # TestData = image_util.ImageDataProvider(  dir + 'test_padded/*.tif')
    TestData = image_util.ImageDataProvider(  dir + '*.tif')
    Data , Label = TestData(len(TestData.data_files))
    filenamesOrig = TestData.data_files

    filenames = []
    for f in range(len(filenamesOrig)):
        fname = filenamesOrig[f]
        filenames.append(fname.split('data/')[1])


    sz = Data.shape
    Dice = np.zeros((Data.shape[0]))
    Dice_Lgc = np.zeros((Data.shape[0]))
    K = int(sz[0]/TestSizeSpan)

    # if K*TestSizeSpan < sz[0]:
    #     K = K + 1

    prediction2 = np.zeros((sz[0],sz[1]-padSize1,sz[2]-padSize1,sz[3]))
    filesnameWrt = []
    for k in range(K-1):
        fname = filenames[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0])]
        data = Data[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0]),...]
        label = Label[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0]),...]
        # path = dir+'model/model.ckpt'
        prediction2 = net.predict( path , data , GPU_Num=gpuNum)

        L = label.shape

        for l in range(L[0]):
            lbl = label[l,padSize:L[1]-padSize,padSize:L[2]-padSize,1]

            prediction = prediction2[l,...,0]
            try:
                Thresh_Mult = max(filters.threshold_otsu(prediction),0.2)
            except:
                Thresh_Mult = 0.2

            prediction_Logical = prediction > Thresh_Mult

            dice = DiceCoefficientCalculator( prediction , lbl )
            dice_Lgc = DiceCoefficientCalculator( prediction_Logical , lbl )
            filesnameWrt.append(fname[l].split('.tif')[0])

            if l%100 == 0:
                fnm = fname[l].split('.tif')[0]
                imageio.imwrite( dir + 'Results/' + fnm + '_pred.jpg', prediction*256 )

                imageio.imwrite( dir + 'Results/' + fnm + '_predLgc.jpg', np.asarray(prediction_Logical,dtype=float)*256 )


        Dice[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0])] = dice
        Dice_Lgc[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0])] = dice_Lgc

    return Dice_Lgc, Dice, filesnameWrt


path = Training(net)
# path =  dir + 'model/model.cpkt'
Dice_Lgc, Dice, filesnameWrt = Testing(path, net)

np.savetxt(dir + 'Results/Dice_Logical.txt', Dice_Lgc , fmt='%1.2f' , delimiter=' ')
np.savetxt(dir + 'Results/Dice.txt', Dice , fmt='%1.2f' , delimiter=' ')
np.savetxt(dir + 'Results/filesnameWrtFull.txt', filesnameWrt , fmt='%s' , delimiter=' ')



## -------------------------------------------
# # TestData = image_util.ImageDataProvider(  dir + 'test_padded/*.tif')
# TestData = image_util.ImageDataProvider(  dir + '*.tif')
# Data , Label = TestData(len(TestData.data_files))
# filenamesOrig = TestData.data_files
#
# filenames = []
# for f in range(len(filenamesOrig)):
#     fname = filenamesOrig[f]
#     filenames.append(fname.split('data/')[1])
#
#
# sz = Data.shape
# Dice = np.zeros((Data.shape[0]))
# Dice_Lgc = np.zeros((Data.shape[0]))
# K = int(sz[0]/TestSizeSpan)
#
# # if K*TestSizeSpan < sz[0]:
# #     K = K + 1
#
# prediction2 = np.zeros((sz[0],sz[1]-padSize1,sz[2]-padSize1,sz[3]))
# filesnameWrt = []
# # for k in range(K-1):
# k = 0
# fname = filenames[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0])]
# data = Data[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0]),...]
# label = Label[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0]),...]
# # path = dir+'model/model.ckpt'
# prediction2 = net.predict( path , data , GPU_Num=gpuNum)
#
# L = label.shape
#
# # for l in range(L[0]):
# l = 0
# lbl = label[l,padSize:L[1]-padSize,padSize:L[2]-padSize,1]
#
# prediction = prediction2[l,...,0]
# try:
#     Thresh_Mult = max(filters.threshold_otsu(prediction),0.2)
# except:
#     Thresh_Mult = 0.2
#
# prediction_Logical = prediction > Thresh_Mult
#
# dice = DiceCoefficientCalculator( prediction , lbl )
# dice_Lgc = DiceCoefficientCalculator( prediction_Logical , lbl )
# filesnameWrt.append(fname[l].split('.tif')[0])
#
# # if l%100 == 0:
# fnm = fname[l].split('.tif')[0]
# imageio.imwrite( dir + 'Results/' + fnm + '_pred.jpg', prediction*256 )
#
# plt.imshow(prediction_Logical,cmap='gray')
# prediction_Logical.min()
# imageio.imwrite( dir + 'Results/' + fnm + '_predLgc.jpg', np.asarray(prediction_Logical,dtype=float)*256 )
# #
