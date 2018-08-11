import numpy as np
import os
import cv2
from time import time
import pickle
from PIL import Image
from skimage.color import rgb2gray
directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'

mode = 'train'  #  'train'  #
subClasses = os.listdir(directory + mode)
downsample = 2

a = time()
for i in range(len(subClasses)):
    print('--- ',subClasses[i],' ---')
    dirr = directory + mode + '/' + subClasses[i] + '/'

    files2 = os.listdir(dirr)
    files = []
    for f in range(len(files2)):
        le = len(files2[f])
        if files2[f][le-4:] == 'jpeg':
            files.append(files2[f])

    for j in range(len(files)): # ): len(files)):
        if j%50 == 0:
            print('j: ',j, ' time: ',time()-a)
            a = time()

            # if j != 0:
            #     print(Full_Image.shape)

        # im = Image.open(dirr + files[j])
        # im = np.asarray(im)
        im2 = cv2.imread(dirr + files[j])
        for rgb in range(3):
            # im = rgb2gray(im)
            im = im2[...,rgb]
            sz = im.shape
            HEIGHT = int(sz[0]/downsample)
            WIDTH = int(sz[1]/downsample)
            im = cv2.resize(im, dsize=(WIDTH,HEIGHT))
            if (j == 0) & (i == 0):
                Full_Image = np.zeros((1,HEIGHT,WIDTH))
                Full_Image[0,...] = im
                Full_Label = subClasses[i]
            else:
                Full_Image = np.concatenate((Full_Image,im[np.newaxis,...]), axis=0)
                Full_Label = np.append(Full_Label,subClasses[i])


    # print(type(i))
    # print(subClasses)
    # print(directory + mode + '_Data_',subClasses[i],'.pkl')
with open(directory + mode + '_Data_3channels.pkl','wb') as outputDir:
    pickle.dump(Full_Image , outputDir)

with open(directory + mode + '_Label_3channels.pkl','wb') as outputDir:
    pickle.dump(Full_Label , outputDir)

# a = time()
# im = Image.open(dirr + files[j])
# im = np.asarray(im)
# b = time()
# print(b-a)
# im = cv2.imread(dirr + files[j])
# c = time()
# print(c-b)
