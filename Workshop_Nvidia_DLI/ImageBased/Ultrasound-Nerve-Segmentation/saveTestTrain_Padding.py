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


dir = '/media/groot/Seagate Backup Plus Drive/dataset/new/Ultrasound-Nerve-Segmentation/'
downsample = 1


def makeDir(mde):
    try:
        mode2 = dir + mde + '_padded/'
        os.makedirs(mode2)
    except:
        print('----')

def main(mode, List):

    makeDir(mode)
    i = 0
    for lst in List:

        i = i + 1
        if i%500 == 0:
            print(i, lst)

        ed = len(lst)
        mask = tifffile.imread(dir + 'train/' + lst)
        im = tifffile.imread(dir + 'train/' + lst[:ed-9] + '.tif')
        sz = im.shape
        pad2 = 88

        if downsample != 1:
            sz2 = np.asarray(sz)
            sz2[0] = int(sz[0]/downsample)
            sz2[1] = int(sz[1]/downsample)
            im = cv2.resize(im,(sz2[1],sz2[0]))
            mask = cv2.resize(mask,(sz2[1],sz2[0]))
            pad2 = 44

        pad = int(pad2/2)
        maskP = np.pad(mask,((pad,pad),(pad,pad)),'constant')
        imP = np.pad(im,((pad,pad),(pad,pad)),'constant')

        tifffile.imsave(dir + mode + '_padded/' + lst , maskP)
        tifffile.imsave(dir + mode + '_padded/' + lst[:ed-9] + '.tif' , imP)

ListTemp = os.listdir(dir + 'train/')

List = []
for ls in ListTemp:
    ed = len(ls)
    if ls[ed-8:] == 'mask.tif':
        List.append(ls)
List = np.asarray(List)

rn = np.random.permutation(List.shape[0])
List = List[rn]

TrainPortion = int(List.shape[0]*0.7)

main('train', List[:TrainPortion])
main('test', List[TrainPortion:])

# main('train', List[:100])
# main('test', List[100:200])
