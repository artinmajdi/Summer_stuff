import numpy as np
import matplotlib.pyplot as plt
# import Image
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter
import matplotlib.pyplot as plt
import cv2
import imageio
from skimage import transform
from scipy.misc import imrotate
dir = '/media/groot/Seagate Backup Plus Drive/dataset/new/Pharmaceutical-Pill-Image-Dataset/1K-Pill-Image-DataSet/'
downsample = 1

def myRotate(im,angle):
    im2 = transform.rotate(im,angle,'nearest')

    sz = im.shape
    sz2 = im2.shape

    shftX = int((sz2[0]-sz[0])/2)
    shftY = int((sz2[1]-sz[1])/2)

    K1 = (sz2[0]-sz[0])%2
    K2 = (sz2[1]-sz[1])%2

    if (K1 == 0) & (K2 == 0):
        im2 = im2[shftX:sz2[0]-shftX , shftY:sz2[1]-shftY , ...]
    elif (K1 == 0) & (K2 != 0):
        im2 = im2[shftX:sz2[0]-shftX , shftY:sz2[1]-shftY-1 , ...]
    elif (K1 != 0) & (K2 == 0):
        im2 = im2[shftX:sz2[0]-shftX-1 , shftY:sz2[1]-shftY , ...]
    else:
        im2 = im2[shftX:sz2[0]-shftX-1 , shftY:sz2[1]-shftY-1 , ...]


    return im2

def OrganizerData():
    try:
        os.makedirs(dir + 'data')
    except:
        'fg'

    for ind in range(254):
        name = '('+str(ind)+')'
        try:
            im = cv2.imread(dir + 'original/' + name + 'r30.jpg')

            im = im[:216,:216,:]
            mask = cv2.imread(dir + 'original/' + name + 'outline.jpg')
            mask = mask[:216,:216,0]

            pad2 = 40
            pad = int(pad2/2)
            maskP = np.pad(mask,((pad,pad),(pad,pad)),'maximum')
            imP = np.pad(im,((pad,pad),(pad,pad),(0,0)),'maximum')

            mask = myRotate(mask,-27)

            tifffile.imsave(dir + 'data/' + 'r30_' + str(ind) + '.tif' , imP)
            tifffile.imsave(dir + 'data/' + 'r30_' + str(ind) + '_mask.tif' , maskP)
        except:
            print(ind)

OrganizerData()
