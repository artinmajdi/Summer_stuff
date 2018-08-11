import matplotlib.pyplot as plt
from tf_unet import unet, image_util
import logging
import numpy as np
from skimage import filters
import os
from PIL import Image
import imageio
dir = '/media/groot/Seagate Backup Plus Drive/dataset/new/Ultrasound-Nerve-Segmentation/'

try:
    os.makedirs(dir+'Results/')
except:
    print('-----------')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
gpuNum = '2'

padSize1 = 88
TestSizeSpan = 20
padSize = int(padSize1/2)
epoch_Num = 60
def DiceCoefficientCalculator(msk1,msk2):
    # print(type(msk1))
    # print(type(msk2))
    intersection = msk1 * msk2  # np.logical_and(msk1,msk2)
    # print(type(intersection))
    DiceCoef = ( intersection.sum() * 2 ) / (msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

def Training(net):

    TrainData = image_util.ImageDataProvider(dir + "train_padded/*.tif",shuffle_data=True)
    L = int(len(TrainData.data_files)/10)
    trainer = unet.Trainer(net, optimizer = "adam")
    path = trainer.train(TrainData, dir+'model', training_iters = L, epochs=epoch_Num, display_step=100 , GPU_Num=gpuNum) # , GPU_Num=gpuNum
    return path

def Testing(path , net):

    # TestData = image_util.ImageDataProvider(  dir + 'test_padded/*.tif')
    TestData = image_util.ImageDataProvider(  dir + 'train_padded/*.tif')
    Data , Label = TestData(len(TestData.data_files))
    filenamesOrig = TestData.data_files

    filenames = []
    for f in range(len(filenamesOrig)):
        fname = filenamesOrig[f]
        filenames.append(fname.split('test_padded/')[1])


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
            # print(type(lbl))
            # print(lbl.sum())
            # print(lbl.shape)
            # print(type(prediction))
            # print(prediction.sum())
            # print(prediction.shape)

            dice = DiceCoefficientCalculator( prediction , lbl )
            dice_Lgc = DiceCoefficientCalculator( prediction_Logical , lbl )
            filesnameWrt.append(fname[l].split('.tif')[0])

            if l%100 == 0:
                fnm = fname[l].split('.tif')[0]
                imageio.imwrite( dir + 'Results/' + fnm + '_pred.jpg', prediction*256 )
                imageio.imwrite( dir + 'Results/' + fnm + '_predLgc.jpg', prediction_Logical*256 )


        Dice[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0])] = dice
        Dice_Lgc[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0])] = dice_Lgc

    return Dice_Lgc, Dice, filesnameWrt

net = unet.Unet(layers=4, features_root=64, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"

# path = Training(net)
path =  dir + 'model/model.cpkt'
Dice_Lgc, Dice, filesnameWrt = Testing(path, net)

np.savetxt(dir + 'Results/Dice_Logical.txt', Dice_Lgc , fmt='%1.2f' , delimiter=' ')
np.savetxt(dir + 'Results/Dice.txt', Dice , fmt='%1.2f' , delimiter=' ')
np.savetxt(dir + 'Results/filesnameWrtFull.txt', filesnameWrt , fmt='%s' , delimiter=' ')


# TestData = image_util.ImageDataProvider(  dir + 'test_padded/*.tif')
# Data , Label = TestData(len(TestData.data_files))
# filenamesOrig = TestData.data_files
# TestSizeSpan = 100
# sz = Data.shape
#
# filenames = []
# for f in range(len(filenamesOrig)):
#     fname = filenamesOrig[f]
#     filenames.append(fname.split('test_padded/')[1])
#
# net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"
#
# k = 0
# data = Data[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0]),...]
# label = Label[(k*TestSizeSpan):min((k+1)*TestSizeSpan,sz[0]),...]
# path = dir+'model/model.cpkt'
# gpuNum = '2'
#
# prediction2 = net.predict( path , data , GPU_Num=gpuNum)
#
# L = label.shape
#
# l  = 0
# lbl = label[l,padSize:L[1]-padSize,padSize:L[2]-padSize,1]
#
# prediction = prediction2[l,...,0]
# try:
#     Thresh_Mult = max(filters.threshold_otsu(prediction),0.2)
# except:
#     Thresh_Mult = 0.5
#
# Thresh_Mult = filters.threshold_otsu(prediction )
#
#
#
# prediction_Logical = prediction > Thresh_Mult
# plt.imshow(prediction> 0.5,cmap='gray')
# plt.imshow(lbl,cmap='gray')
