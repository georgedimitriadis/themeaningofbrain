import tensorflow as tf

import numpy as np

tf.enable_eager_execution()

assert tf.executing_eagerly()


class ZCA(object):
    """
    Simple ZCA aka Mahalanobis transformation class made in TensorFlow.
    The code was largely ported from Keras ImageDataGenerator
    """

    def __init__(self, epsilon=1e-5, dtype='float64'):

        """epsilon is the normalization constant, dtype refers to the data type used in the computation.
         WARNING: the default precision is set to float64 as i have found that when computing the mean tensorflow'
         and numpy results can differ by a substantial amount.
         Usage: fit method computes the principal components and should be called first,
                compute method returns the actual transformed tensor
         NOTE : The input to both methods must be a 4D tensor.
        """

        assert dtype is 'float32' or 'float64', "precision must be float32 or float64"

        self.epsilon = epsilon
        self.dtype = dtype
        self.princ_comp = None
        self.mean = None

    def _featurewise_center(self, images_tensor):

        if self.mean is None:
            self.mean, _ = tf.nn.moments(images_tensor, axes=(0, 1, 2))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[2] = images_tensor.shape[3]
            self.mean = tf.reshape(self.mean, broadcast_shape)

        norm_images = tf.subtract(images_tensor, self.mean)

        return norm_images

    def fit(self, images_tensor):

        assert images_tensor.shape[3], "The input should be a 4D tensor"

        if images_tensor.dtype is not self.dtype:  # numerical error for float32

            images_tensor = tf.cast(images_tensor, self.dtype)

        images_tensor = self._featurewise_center(images_tensor)

        flat = tf.reshape(images_tensor, (-1, np.prod(images_tensor.shape[1:].as_list())))
        sigma = tf.div(tf.matmul(tf.transpose(flat), flat), tf.cast(flat.shape[0], self.dtype))
        s, u, _ = tf.svd(sigma)
        s_inv = tf.div(tf.cast(1, self.dtype), (tf.sqrt(tf.add(s[tf.newaxis], self.epsilon))))
        self.princ_comp = tf.matmul(tf.multiply(u, s_inv), tf.transpose(u))

    def compute(self, images_tensor):

        assert images_tensor.shape[3], "The input should be a 4D tensor"

        assert self.princ_comp is not None, "Fit method should be called first"

        if images_tensor.dtype is not self.dtype:
            images_tensor = tf.cast(images_tensor, self.dtype)

        images_tensors = self._featurewise_center(images_tensor)

        flatx = tf.cast(tf.reshape(images_tensors, (-1, np.prod(images_tensors.shape[1:]))), self.dtype)
        whitex = tf.matmul(flatx, self.princ_comp)
        x = tf.reshape(whitex, images_tensors.shape)

        return x

# Example of use
'''
def main():
    from keras.datasets import mnist

    import matplotlib.pyplot as plt

    train_set, test_set = mnist.load_data()
    x_train, y_train = train_set

    zca1 = ZCA(epsilon=1e-5, dtype='float64')

    # input should be a 4D tensor

    x_train = x_train.reshape(*x_train.shape, 1)
    zca1.fit(x_train)
    x_train_transf = zca1.compute(x_train)

    # reshaping to 28*28 and casting to uint8 for plotting

    x_train_transf = tf.reshape(x_train_transf, x_train_transf.shape[0:3])


    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(x_train_transf[i],
                  cmap='binary'
                  )

        xlabel = "True: %d" % y_train[i]
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


if __name__ == '__main__':
main()
'''

import sys
sys.path.append(r"E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\ExperimentSpecificCode\_2018_Chronic_Neuroseeker_TouchingLight\_2018_04_AK_33p1\NNs\Used_code")
from tensorflow_zca import ZCA