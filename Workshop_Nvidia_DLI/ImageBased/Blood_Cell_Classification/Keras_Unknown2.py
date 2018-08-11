!mkdir ./kaggle

from google.colab import files

files.upload()
dirr = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/'
# '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images'
# dirr = directory + 'dataset2-master/images/'

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from keras.applications import vgg16
from keras.preprocessing import image

samples = 9957
batch_size = 16

#run predictions:
generator = image.ImageDataGenerator(
        rescale = 1./255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

dataset = generator.flow_from_directory(
    shuffle = True,
    batch_size = 32,
    target_size = (80, 80),
    directory = dirr + 'train' # dataset2-master/images/train
)

def model():
    model = Sequential()
    model.add(Conv2D(80, (3,3), strides = (1, 1), activation = 'relu', input_shape = (80, 80, 3)))
    model.add(Conv2D(64, (3,3), strides = (1, 1), activation = 'relu', input_shape = (80, 80, 3)))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(64, (3,3), strides = (1,1), activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

    return model

nn = model()
nn.fit_generator(dataset, steps_per_epoch = None, epochs = 30, verbose = 1)
nn.save('Model.h5')
