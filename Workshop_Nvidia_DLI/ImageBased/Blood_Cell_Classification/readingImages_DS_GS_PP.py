import numpy as np
import os
import cv2
from time import time
import pickle
from PIL import Image
from skimage.color import rgb2gray
import multiprocessing as mp
directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'

output = mp.Queue()

mode = 'train'  #  'train'  #
subClasses = os.listdir(directory + mode)
downsample = 2

def main(I):
    subClassesI = subClasses[I]
    print('--- ',subClassesI,' ---')
    dirr = directory + mode + '/' + subClassesI + '/'

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
        im = cv2.imread(dirr + files[j])
        im = rgb2gray(im)
        sz = im.shape
        HEIGHT = int(sz[0]/downsample)
        WIDTH = int(sz[1]/downsample)
        im = cv2.resize(im, dsize=(WIDTH,HEIGHT))
        if (j == 0):
            Full_Image = np.zeros((1,HEIGHT,WIDTH))
            Full_Image[0,...] = im
            Full_Label = subClasses[i]
        else:
            Full_Image = np.concatenate((Full_Image,im[np.newaxis,...]), axis=0)
            Full_Label = np.append(Full_Label,subClasses[i])

    return Full_Image, Full_Label


a = time()

processes = [mp.Process(target=main, args=(I)) for I in range(len(subClasses))]

for p in processes:
    p.start()

for p in processes:
    p.join()

results = [output.get() for p in processes]
print(results)
#
# with open(directory + mode + '_Data.pkl','wb') as outputDir:
#     pickle.dump(Full_Image , outputDir)
#
# with open(directory + mode + '_Label.pkl','wb') as outputDir:
#     pickle.dump(Full_Label , outputDir)


# a = time()
# im = Image.open(dirr + files[j])
# im = np.asarray(im)
# b = time()
# print(b-a)
# im = cv2.imread(dirr + files[j])
# c = time()
# print(c-b)




###### ------------------------------------------------------------

    ## for python2
    # for SbFlds in subFolders:
    #     processes = [mp.Process(target=main_Part, args=(SbFlds,TestName))]

    ## for python3

    # if sFi < tt:
    #     processes = [mp.Process(target=main_Part, args=(SbFlds,TestName)) for SbFlds in subFolders[divider*sFi:divider*(sFi+1)]]
    # elif (Remdr != 0) & (sFi == tt):
    #         processes = [mp.Process(target=main_Part, args=(SbFlds,TestName)) for SbFlds in subFolders[divider*(sFi):]]
    #
    # print(processes)
    #
    # for p in processes:
    #     p.start()
    #
    # for p in processes:
    #     p.join()
