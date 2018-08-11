from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import pickle
from skimage.color import rgb2gray
from resizeimage import resizeimage
import cv2

tf.logging.set_verbosity(tf.logging.INFO)
num_classes = 4

imSz = [28,28] # [240,320]

def readingDataset(mode):

    directory = '/media/groot/Seagate Backup Plus Drive/dataset/new/BloodCell_Images/blood-cells/dataset2-master/images/'
    with open(directory + mode + '_Data_20Per.pkl','rb') as outputDir:
        FullIm = pickle.load(outputDir)

    sz = FullIm.shape
    FullIm_Gray = np.zeros((sz[0],imSz[0],imSz[1],1))
    for i in range(sz[0]):
        im = FullIm[i,...].reshape((sz[1],sz[2],sz[3]))
        im = rgb2gray(im)
        FullIm_Gray[i,...,0] = cv2.resize(im, (28, 28))


    with open(directory + mode + '_Label_20Per.pkl','rb') as outputDir:
        FullLb = pickle.load(outputDir)

    FullLb2 = np.zeros((len(FullLb)))
    classes = np.unique(FullLb)
    for cl in range(len(classes)):
        ind = np.where(FullLb == classes[cl])
        FullLb2[ind] = int(cl)

    return FullIm_Gray, FullLb2


def cnn_model_fn(features, labels, mode):

    # with tf.device('/device:GPU:0'):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel

    # input_layer = tf.reshape(features["x"], [-1, imSz[0], imSz[1], 1])
    input_layer = features["x"]
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
    inputs = input_layer,
    filters = 32,
    kernel_size=[5, 5],
    padding="same",
    activation = tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
    inputs = pool1,
    filters=64,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1,int(imSz[0]/4)*int(imSz[1]/4)*64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions) # , config=tf.contrib.learn.RunConfig(session_config=config)) # , config=config

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op) # , config=tf.contrib.learn.RunConfig(session_config=config)) # , config=config

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    A = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops) # , config=tf.contrib.learn.RunConfig(session_config=config))
    return A

def main(unused_argv):

    # with tf.device('/device:GPU:0'):

    train_data, train_Label = readingDataset('train')
    eval_data , eval_Label  = readingDataset('test')


    train_data = np.asarray(train_data, dtype=np.float32)
    eval_data = np.asarray(eval_data, dtype=np.float32)
    train_labels = np.asarray(train_Label, dtype=np.int32)
    eval_labels = np.asarray(eval_Label, dtype=np.int32)

    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_dataM = mnist.train.images
    # train_dataM
    # train_data

    # train_data = train_data.reshape(-1,imSz[0],imSz[1],1)
    # # type(train_data)
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images
    # eval_data = eval_data.reshape(-1,imSz[0],imSz[1],1)
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the EstimatorSpec

    mnist_classifier = tf.estimator.Estimator(
    model_fn = cnn_model_fn, model_dir="/media/groot/Seagate Backup Plus Drive/code/CNN_MNIST") # , config=tf.contrib.learn.RunConfig(session_config=config)) #, config=config "/tmp/mnist_convnet_model")

    # set up logging for predictions
    # log the values in the "softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook( tensors=tensors_to_log, every_n_iter=500 )


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True)

    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000,
        # hooks = [logging_hook]
        )

    # Evaluate the model and prints results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs=1,
        shuffle = False)

    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)

# if __name__ == "__main__":
# t0 = time.time()

config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True

# myGraph = tf.Graph()
#sess = tf.Session() # config=config) #  , graph=myGraph.as_default())
#sess.run(main(1))
App = tf.app.run()
# t1 = time.time()

# print(t1-t0)
