import numpy as np
import os
import cv2
from time import time
import pickle
from PIL import Image

directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'

mode = 'test'  #  'train'  #
subClasses = os.listdir(directory + mode)

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
        if j%500 == 0:
            print('j: ',j, ' time: ',time()-a)
            a = time()

            if j != 0:
                print(Full_Image.shape)

        # im = Image.open(dirr + files[j])
        # im = np.asarray(im)
        im = cv2.imread(dirr + files[j])
        if (j == 0): # & (i == 0)
            sz = im.shape
            Full_Image = np.zeros((1,sz[0],sz[1],sz[2]))
            Full_Image[0,...] = im
            Full_Label = subClasses[i]
        else:
            Full_Image = np.concatenate((Full_Image,im[np.newaxis,...]), axis=0)
            Full_Label = np.append(Full_Label,subClasses[i])

    # print(type(i))
    # print(subClasses)
    # print(directory + mode + '_Data_',subClasses[i],'.pkl')
    with open(directory + mode + '_Data_'+subClasses[i]+'.pkl','wb') as outputDir:
        pickle.dump(Full_Image , outputDir)

    with open(directory + mode + '_Label_'+subClasses[i]+'.pkl','wb') as outputDir:
        pickle.dump(Full_Label , outputDir)

# a = time()
# im = Image.open(dirr + files[j])
# im = np.asarray(im)
# b = time()
# print(b-a)
# im = cv2.imread(dirr + files[j])
# c = time()
# print(c-b)
